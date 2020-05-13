import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models.resnet as resnet
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
from anchors import Anchors
from utils import RegressionTransform
import losses


NUM_ANCHOR = 3
IN_CHANNELS = 512
FPN_NUM = 4


def load_model(path_to_model, device=None):
    model = create_retinaface()

    # Load trained model
    pre_state_dict = torch.load(path_to_model)
    retina_dict = model.state_dict()
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    model.load_state_dict(pretrained_dict)

    if device is not None:
        model = model.to(device)
    model.eval()

    return model


class ContextModule(nn.Module):
    def __init__(self, in_channels=256):
        super(ContextModule, self).__init__()
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.det_context_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.det_context_conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // 2)
        )
        self.det_context_conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.det_context_conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // 2)
        )
        self.det_concat_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.det_conv1(x)
        x_ = self.det_context_conv1(x)
        x2 = self.det_context_conv2(x_)
        x3_ = self.det_context_conv3_1(x_)
        x3 = self.det_context_conv3_2(x3_)

        out = torch.cat((x1, x2, x3), 1)
        act_out = self.det_concat_relu(out)

        return act_out


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_blocks = nn.ModuleList()
        self.context_blocks = nn.ModuleList()
        self.aggr_blocks = nn.ModuleList()
        for i, in_channels in enumerate(in_channels_list):
            if in_channels == 0:
                continue
            lateral_block_module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            aggr_block_module = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            context_block_module = ContextModule(out_channels)
            self.lateral_blocks.append(lateral_block_module)
            self.context_blocks.append(context_block_module)
            if i > 0:
                self.aggr_blocks.append(aggr_block_module)

        # initialize params of fpn layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.lateral_blocks[-1](x[-1])
        results = []
        results.append(self.context_blocks[-1](last_inner))
        for feature, lateral_block, context_block, aggr_block in zip(
                x[:-1][::-1], self.lateral_blocks[:-1][::-1], self.context_blocks[:-1][::-1], self.aggr_blocks[::-1]
        ):
            if not lateral_block:
                continue
            lateral_feature = lateral_block(feature)
            feat_shape = lateral_feature.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = lateral_feature + inner_top_down
            last_inner = aggr_block(last_inner)
            results.insert(0, context_block(last_inner))

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class ClassHead(nn.Module):
    def __init__(self, inchannels=IN_CHANNELS, num_anchors=NUM_ANCHOR):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)
        self.dropout = nn.Dropout(0.1, inplace=True)
        self.output_act = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = self.dropout(self.conv1x1(x))
        out = out.permute(0, 2, 3, 1)
        b, h, w, c = out.shape
        out = out.view(b, h, w, self.num_anchors, 2)
        out = self.output_act(out)

        return out.contiguous().view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=IN_CHANNELS, num_anchors=NUM_ANCHOR):
        super(BboxHead, self).__init__()
        self.predict_amount = 4
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * self.predict_amount, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, self.predict_amount)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=IN_CHANNELS, num_anchors=NUM_ANCHOR):
        super(LandmarkHead, self).__init__()
        self.predict_amount = 10
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * self.predict_amount, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, self.predict_amount)


class RetinaFace(nn.Module):
    def __init__(self, backbone, return_layers, anchor_nums=NUM_ANCHOR):
        super(RetinaFace, self).__init__()
        # if backbone_name == 'resnet50':
        #     self.backbone = resnet.resnet50(pretrained)
        # self.backbone = resnet.__dict__[backbone_name](pretrained=pretrained)
        # self.return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
        assert backbone, 'Backbone can not be none!'
        assert len(return_layers) > 0, 'There must be at least one return layers'
        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
        # in_channels_stage2 = 256
        in_channels_stage2 = 64
        in_channels_list = [
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = 256
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels)
        self.ClassHead = self._make_class_head()
        self.BboxHead = self._make_bbox_head()
        self.LandmarkHead = self._make_landmark_head()
        self.anchors = Anchors()
        self.regressBoxes = RegressionTransform()
        self.losslayer = losses.LossLayer()

    @staticmethod
    def _make_class_head(fpn_num=FPN_NUM, inchannels=IN_CHANNELS, num_anchor=NUM_ANCHOR):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, num_anchor))
        return classhead

    @staticmethod
    def _make_bbox_head(fpn_num=FPN_NUM, inchannels=IN_CHANNELS, num_anchor=NUM_ANCHOR):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, num_anchor))
        return bboxhead

    @staticmethod
    def _make_landmark_head(fpn_num=FPN_NUM, inchannels=IN_CHANNELS, num_anchor=NUM_ANCHOR):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, num_anchor))
        return landmarkhead

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        out = self.body(img_batch)
        features = self.fpn(out)

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features.values())], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features.values())],
                                    dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features.values())], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.losslayer(classifications, bbox_regressions, ldm_regressions, anchors, annotations)
        else:
            bboxes, landmarks = self.regressBoxes(anchors, bbox_regressions, ldm_regressions, img_batch)

            return classifications, bboxes, landmarks


def create_retinaface(backbone_name='resnet18', anchors_num=NUM_ANCHOR, pretrained=True):
    backbone = resnet.__dict__[backbone_name](pretrained=pretrained)
    # freeze layer1
    for name, parameter in backbone.named_parameters():
        # if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #     parameter.requires_grad_(False)
        if name == 'conv1.weight':
            # print('freeze first conv layer...')
            parameter.requires_grad_(False)

    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
    model = RetinaFace(backbone, return_layers, anchor_nums=anchors_num)

    return model
