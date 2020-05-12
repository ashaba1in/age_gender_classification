import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch import multiprocessing
from torch.utils.data import DataLoader
import torch.nn.functional as F
import skimage.transform
import skimage.color
import numpy as np
import skimage.io
import skimage
import random
import torch
import os
from typing import Tuple, Union

IMAGE_SIZE = 256


def collater(data):
    batch_size = len(data)

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]

    # batch images
    height = imgs[0].shape[0]
    width = imgs[0].shape[1]
    assert height == width, 'Input width must eqs height'

    input_size = width
    batched_imgs = torch.zeros(batch_size, height, width, 3)

    for i in range(batch_size):
        img = imgs[i]
        batched_imgs[i, :] = img

    # batch annotations
    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        if annots[0].shape[1] > 4:
            annot_padded = torch.ones((len(annots), max_num_annots, 14)) * -1
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
        else:
            annot_padded = torch.ones((len(annots), max_num_annots, 4)) * -1
            # print('annot~~~~~~~~~~~~~~~~~~,',annots)
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        if annots[0].shape[1] > 4:
            annot_padded = torch.ones((len(annots), 1, 14)) * -1
        else:
            annot_padded = torch.ones((len(annots), 1, 4)) * -1

    batched_imgs = batched_imgs.permute(0, 3, 1, 2)

    return {'img': batched_imgs, 'annot': annot_padded}


class RandomCroper(object):
    def __call__(self, sample, input_size=IMAGE_SIZE):
        image, annots = sample['img'], sample['annot']
        rows, cols, _ = image.shape

        smallest_side = min(rows, cols)
        longest_side = max(rows, cols)
        scale = random.uniform(0.3, 1)
        short_size = int(smallest_side * scale)
        start_short_upscale = smallest_side - short_size
        start_long_upscale = longest_side - short_size
        crop_short = random.randint(0, start_short_upscale)
        crop_long = random.randint(0, start_long_upscale)
        crop_y = 0
        crop_x = 0
        if smallest_side == rows:
            crop_y = crop_short
            crop_x = crop_long
        else:
            crop_x = crop_short
            crop_y = crop_long
        # crop        
        cropped_img = image[crop_y:crop_y + short_size, crop_x:crop_x + short_size]
        # resize
        new_image = skimage.transform.resize(cropped_img, (input_size, input_size))

        # why normalized from 255 to 1 after skimage.transform?????????
        new_image = new_image * 255

        # relocate bbox
        annots[:, 0] = annots[:, 0] - crop_x
        annots[:, 1] = annots[:, 1] - crop_y
        annots[:, 2] = annots[:, 2] - crop_x
        annots[:, 3] = annots[:, 3] - crop_y

        # relocate landmarks
        if annots.shape[1] > 4:
            # l_mask = annots[:,4]!=-1
            l_mask = annots[:, 4] > 0
            annots[l_mask, 4] = annots[l_mask, 4] - crop_x
            annots[l_mask, 5] = annots[l_mask, 5] - crop_y
            annots[l_mask, 6] = annots[l_mask, 6] - crop_x
            annots[l_mask, 7] = annots[l_mask, 7] - crop_y
            annots[l_mask, 8] = annots[l_mask, 8] - crop_x
            annots[l_mask, 9] = annots[l_mask, 9] - crop_y
            annots[l_mask, 10] = annots[l_mask, 10] - crop_x
            annots[l_mask, 11] = annots[l_mask, 11] - crop_y
            annots[l_mask, 12] = annots[l_mask, 12] - crop_x
            annots[l_mask, 13] = annots[l_mask, 13] - crop_y

        # scale annotations
        resize_scale = input_size / short_size
        annots[:, :4] = annots[:, :4] * resize_scale
        if annots.shape[1] > 4:
            annots[l_mask, 4:] = annots[l_mask, 4:] * resize_scale

        # remove faces center not in image afer crop
        center_x = (annots[:, 0] + annots[:, 2]) / 2
        center_y = (annots[:, 1] + annots[:, 3]) / 2

        mask_x = (center_x[:, ] > 0) & (center_x[:, ] < input_size)
        mask_y = (center_y[:, ] > 0) & (center_y[:, ] < input_size)

        mask = mask_x & mask_y

        # clip bbox
        annots[:, :4] = annots[:, :4].clip(0, input_size)

        # clip landmarks
        if annots.shape[1] > 4:
            annots[l_mask, 4:] = annots[l_mask, 4:].clip(0, input_size)

        annots = annots[mask]

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots)}


class RandomFlip(object):
    def __call__(self, sample, input_size=IMAGE_SIZE, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']

            # flip image
            image = torch.flip(image, [1])

            image = image.numpy()
            annots = annots.numpy()

            # relocate bboxes
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            x_tmp = x1.copy()
            annots[:, 0] = input_size - x2
            annots[:, 2] = input_size - x_tmp

            # relocate landmarks
            # l_mask = annots[:, 4]!=-1
            l_mask = annots[:, 4] > 0
            annots[l_mask, 4::2] = input_size - annots[l_mask, 4::2]
            l_tmp = annots.copy()
            annots[l_mask, 4:6] = l_tmp[l_mask, 6:8]
            annots[l_mask, 6:8] = l_tmp[l_mask, 4:6]
            annots[l_mask, 10:12] = l_tmp[l_mask, 12:]
            annots[l_mask, 12:] = l_tmp[l_mask, 10:12]

            image = torch.from_numpy(image)
            annots = torch.from_numpy(annots)

            sample = {'img': image, 'annot': annots}

        return sample


class Resizer(object):
    def __init__(self, input_size=IMAGE_SIZE):
        self.input_size = input_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        rows, cols, _ = image.shape
        long_side = max(rows, cols)
        scale = self.input_size / long_side

        # resize image
        resized_image = skimage.transform.resize(image, (
            int(rows * self.input_size / long_side), int(cols * self.input_size / long_side)))
        resized_image = resized_image * 255

        assert (resized_image.shape[0] == self.input_size or resized_image.shape[1] == self.input_size), \
            'resized image size not {}'.format(self.input_size)

        if annots.shape[1] > 4:
            annots = annots * scale
        else:
            annots[:, :4] = annots[:, :4] * scale

        return {'img': resized_image, 'annot': annots}


class PadToSquare(object):
    def __init__(self, input_size=IMAGE_SIZE):
        self.input_size = input_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        rows, cols, _ = image.shape
        dim_diff = np.abs(rows - cols)

        # relocate bbox annotations
        if rows == self.input_size:
            diff = self.input_size - cols
            annots[:, 0] = annots[:, 0] + diff / 2
            annots[:, 2] = annots[:, 2] + diff / 2
        elif cols == self.input_size:
            diff = self.input_size - rows
            annots[:, 1] = annots[:, 1] + diff / 2
            annots[:, 3] = annots[:, 3] + diff / 2
        if annots.shape[1] > 4:
            ldm_mask = annots[:, 4] > 0
            if rows == self.input_size:
                diff = self.input_size - cols
                annots[ldm_mask, 4::2] = annots[ldm_mask, 4::2] + diff / 2
            elif cols == self.input_size:
                diff = self.input_size - rows
                annots[ldm_mask, 5::2] = annots[ldm_mask, 5::2] + diff / 2

        # pad image
        img = torch.from_numpy(image)
        img = img.permute(2, 0, 1)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = (0, 0, pad1, pad2) if rows <= cols else (pad1, pad2, 0, 0)

        padded_img = F.pad(img, pad, "constant", value=0)
        padded_img = padded_img.permute(1, 2, 0)

        annots = torch.from_numpy(annots)

        return {'img': padded_img, 'annot': annots}


def get_data(path_to_labels) -> Tuple[np.ndarray, np.ndarray]:
    with open(path_to_labels, 'r') as f:
        lines = f.readlines()
    is_first = True
    filenames = []
    labels = []
    labels_on_photo = []
    for line in lines:
        line = line.rstrip()
        if line.startswith('#'):
            if is_first:
                is_first = False
            else:
                labels.append(np.array(labels_on_photo).copy())
                labels_on_photo.clear()
            img_path = os.path.join(path_to_labels.replace('label.txt', 'images'), line[2:])
            filenames.append(img_path)
        else:
            line = line.split(' ')
            label = np.array([float(x) for x in line][:18])
            labels_on_photo.append(label)

    labels.append(np.array(labels_on_photo).copy())

    return np.array(filenames), np.array(labels)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, txt_path, transform=None, flip=False):
        self.transform = transform
        self.flip = flip
        self.batch_count = 0
        self.img_paths, self.labels = get_data(txt_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = skimage.io.imread(self.img_paths[index])
        labels = self.labels[index]

        if len(labels) == 0:
            return labels

        # x1, y1, w1, h1 -> x1, y1, x2, y2
        labels[:, 2:4] += labels[:, 0:2]

        if labels.shape[1] > 4:
            # landmarks
            labels[:, 4] = labels[:, 4]  # l0_x
            labels[:, 5] = labels[:, 5]  # l0_y
            labels[:, 6] = labels[:, 7]  # l1_x
            labels[:, 7] = labels[:, 8]  # l1_y
            labels[:, 8] = labels[:, 10]  # l2_x
            labels[:, 9] = labels[:, 11]  # l2_y
            labels[:, 10] = labels[:, 13]  # l3_x
            labels[:, 11] = labels[:, 14]  # l3_y
            labels[:, 12] = labels[:, 16]  # l4_x
            labels[:, 13] = labels[:, 17]  # l4_y

        sample = {'img': img, 'annot': labels[:, :14]}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def load_data(path: str,
              batch_size: int = 32,
              num_workers: int = multiprocessing.cpu_count(),
              use_gpu: bool = True,
              split_train_test: bool = False) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    if split_train_test:
        dataset = Dataset(path, transform=transforms.Compose([Resizer(), PadToSquare()]))
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.05)
        dataloader_train = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size,
                                      collate_fn=collater, shuffle=True, pin_memory=use_gpu)
        dataloader_test = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size,
                                     collate_fn=collater, shuffle=True, pin_memory=use_gpu)
        return dataloader_train, dataloader_test
    else:
        dataset = Dataset(path, transform=transforms.Compose([Resizer(), PadToSquare()]))
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size,
                                collate_fn=collater, shuffle=True, pin_memory=use_gpu)
        return dataloader
