import os
from typing import Any

import cv2.cv2 as cv2
import dlib
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import scipy.misc


IMAGE_SIZE = 128


class Model:
    def __init__(self, device=None, num_classes: int = 2):
        self.est = None
        self.device = device
        self.num_classes = num_classes

    def _predict(self, data):
        return self.est(data).data.max(1, keepdim=True)[1].cpu().numpy().ravel()

    def predict(self, image):
        return self._predict(image)

    def load_est(self, path):
        model_name = path.split('/')[-1][:-4]
        possible_names = ['ResNeXt-101-32x8d', 'WideResNet-101-2', 'WideResNet-50-2', 'ResNet-152',
                          'Densenet-161', 'ResNeXt-50-32x4d', 'ResNet-101', 'Densenet-201', 'ResNet-50',
                          'Densenet-169', 'Densenet-121', 'ResNet-34', 'ResNet-18']
    
        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()
        
            def forward(self, x):
                return x
    
        model = None
    
        if model_name in possible_names:
            if model_name == 'ResNet-18':
                model = torchvision.models.resnet18(pretrained=False)
                model.pool0 = Identity()
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif model_name == 'ResNet-34':
                model = torchvision.models.resnet34(pretrained=False)
                model.pool0 = Identity()
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif model_name == 'ResNet-50':
                model = torchvision.models.resnet50(pretrained=False)
                model.pool0 = Identity()
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif model_name == 'ResNet-101':
                model = torchvision.models.resnet101(pretrained=False)
                model.pool0 = Identity()
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif model_name == 'ResNet-152':
                model = torchvision.models.resnet152(pretrained=False)
                model.pool0 = Identity()
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif model_name == 'ResNeXt-50-32x4d':
                model = torchvision.models.resnext50_32x4d(pretrained=False)
                model.pool0 = Identity()
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif model_name == 'ResNeXt-101-32x8d':
                model = torchvision.models.resnext101_32x8d(pretrained=False)
                model.pool0 = Identity()
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif model_name == 'WideResNet-50-2':
                model = torchvision.models.wide_resnet50_2(pretrained=False)
                model.pool0 = Identity()
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif model_name == 'WideResNet-101-2':
                model = torchvision.models.wide_resnet101_2(pretrained=False)
                model.pool0 = Identity()
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif model_name == 'Densenet-121':
                model = torchvision.models.densenet121(pretrained=False)
                model.features.pool0 = Identity()
                model.fc = nn.Linear(model.classifier.in_features, self.num_classes)
            elif model_name == 'Densenet-169':
                model = torchvision.models.densenet169(pretrained=False)
                model.features.pool0 = Identity()
                model.fc = nn.Linear(model.classifier.in_features, self.num_classes)
            elif model_name == 'Densenet-201':
                model = torchvision.models.densenet201(pretrained=False)
                model.features.pool0 = Identity()
                model.fc = nn.Linear(model.classifier.in_features, self.num_classes)
            elif model_name == 'Densenet-161':
                model = torchvision.models.densenet161(pretrained=False)
                model.features.pool0 = Identity()
                model.fc = nn.Linear(model.classifier.in_features, self.num_classes)
    
        self.est = model
        self.est.load_state_dict(torch.load(path))
        if self.device is not None:
            self.est.to(self.device)
        self.est.eval()
        return self


class LivenessChecker:
    def __init__(self, path: str = 'Densenet-121.pth', device=None):
        self.model = Model(device).load_est(path)

    def check_liveness(self, image):
        return self.model.predict(image)


class FaceHandler:
    def __init__(self, args):
        self.args = args
        self.predictor = dlib.shape_predictor(args.get('predictor_path'))
        self.detector = dlib.cnn_face_detection_model_v1(args.get('detector_path'))
        self.detect_face = args.get('detect_face')
        self.save_path = args.get('save_path')
        self.left_eye = args.get('left_eye')
        self.face_width = args.get('face_width')
        self.face_height = args.get('face_height')
        self.eyes_slice = 5

        if self.face_width is None:
            self.face_width = 256
        if self.face_height is None:
            self.face_height = self.face_width
        if self.left_eye is None:
            self.left_eye = 0.35, 0.35

    def eyes_pos(self, shape):
        coords = np.empty((shape.num_parts, 2), dtype=int)

        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        return coords[self.eyes_slice + 1:], coords[:self.eyes_slice]

    def align(self, image, rect):
        shape = self.predictor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), rect)
        leftEyePts, rightEyePts = self.eyes_pos(shape)

        leftEyeCenter = leftEyePts.mean(axis=0)
        rightEyeCenter = rightEyePts.mean(axis=0)

        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]

        angle = np.degrees(np.arctan2(dY, dX)) - 180

        desiredRightEyeX = 1.0 - self.left_eye[0]

        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.left_eye[0]) * self.face_width
        scale = desiredDist / dist

        eyesCenter = (leftEyeCenter[0] + rightEyeCenter[0]) // 2, (leftEyeCenter[1] + rightEyeCenter[1]) // 2

        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        tX = self.face_width * 0.5
        tY = self.face_height * self.left_eye[1]

        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        size = self.face_width, self.face_height

        return Image.fromarray(cv2.warpAffine(image, M, size, flags=cv2.INTER_CUBIC), 'RGB')
    
    def process(self, image, args):
        human = args.get('human')
        total = args.get('total')
        ext = args.get('ext')
    
        rects = dlib.rectangles()
    
        if self.detect_face:
            detected = self.detector(image, 0)
            rects.extend([d.rect for d in detected])
        else:
            rects.append(dlib.rectangle(0, 0, image.shape[0], image.shape[1]))
    
        for i, rect in enumerate(rects):
            x1, y1 = rect.left(), rect.top()
            x2, y2 = rect.right() + 1, rect.bottom() + 1
        
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 < 0:
                x2 = 0
            if y2 < 0:
                y2 = 0
        
            if x2 - x1 > 0 and y2 - y1 > 0:
                self.align(image, rect.rect).save(
                    os.path.join(self.save_path, '_'.join(['face', str(int(human)), str(total + i)])) + '.' + ext
                )
    
        return total + len(rects)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, mode: str = None) -> None:
        super(ImageDataset, self).__init__()
        self.df = dataframe
        self.mode = mode
        image_size = 100

        transforms_list = []

        if self.mode == 'train':
            transforms_list.extend([
                transforms.RandomChoice([
                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                    transforms.RandomAffine(degrees=20, translate=(0.05, 0.05), scale=(0.7, 1.3), shear=10),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomGrayscale(0.05)
                ])
            ])
        
        transforms_list.extend([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transforms = transforms.Compose(transforms_list)
        self.targets = dataframe['target'].values
    
    def __getitem__(self, index: int) -> Any:
        filename = self.df['filename'].values[index]
        sample = Image.open(filename)
        assert sample.mode == 'RGB'
        image = self.transforms(sample)
        if self.mode == 'debug':
            return image, self.df['target'].values[index], filename
        return image, self.df['target'].values[index]

    def __len__(self) -> int:
        return self.df.shape[0]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filenames: np.array) -> None:
        super(Dataset, self).__init__()
        self.filenames = filenames
        self.detected_filenames = []

        transforms_list = [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ]

        self.transforms = transforms.Compose(transforms_list)

        self.detector = dlib.get_frontal_face_detector()  # our detector

    def detect_faces(self) -> None:
        for i in range(self.__len__()):
            image = self.__getitem__(i)
            dets = self.detector(image, 0)

            for j, d in enumerate(dets):
                top = max(d.top(), 0)
                bot = min(d.bottom(), IMAGE_SIZE)
                left = max(d.left(), 0)
                right = min(d.right(), IMAGE_SIZE)
                image = image[top:bot, left:right]

                path = "detected/{}_{}.jpeg".format(i, j)
                cv2.imwrite(path, image)

                self.detected_filenames.append(path)

    def __getitem__(self, index: int) -> np.array:
        filename = self.filenames[index]
        sample = cv2.resize(cv2.imread(filename), (IMAGE_SIZE, IMAGE_SIZE))
        return sample

    def __iter__(self):
        yield from self.filenames

    def __len__(self) -> int:
        return len(self.filenames)
