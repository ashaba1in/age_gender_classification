import json
import multiprocessing
import os
from datetime import datetime
from typing import (
    Any,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_config():
    with open('technical/config.json', 'r') as _:
        return json.load(_)


config = get_config()

IMAGE_SIZE = config['image_size']


class AgeGender:
    def __init__(
            self,
            model_name: str,
            device=None,
            load_pretrained: bool = False,
            path_pretrained: str = None
    ):
        self.device = device
        self.zoo_pretrained = config['pretrained']
        self.path_pretrained = path_pretrained
        self.possible_names = config['model_names']

        self._est = None

        if model_name in self.possible_names:
            class Identity(nn.Module):
                def __init__(self):
                    super(Identity, self).__init__()

                @staticmethod
                def forward(x):
                    return x

            class DoubleOutput(nn.Module):
                def __init__(self, in_features: int):
                    super(DoubleOutput, self).__init__()
                    self.fc11 = nn.Linear(in_features, in_features // 2)
                    self.fc12 = nn.Linear(in_features // 2, config['num_classes_age'])
                    self.relu = nn.ReLU(inplace=False)
                    self.fc21 = nn.Linear(in_features, in_features // 2)
                    self.fc22 = nn.Linear(in_features // 2, config['num_classes_gender'])

                    nn.init.normal_(self.fc11.weight)
                    nn.init.normal_(self.fc12.weight)
                    nn.init.normal_(self.fc21.weight)
                    nn.init.normal_(self.fc22.weight)

                    nn.init.uniform_(self.fc11.bias)
                    nn.init.uniform_(self.fc12.bias)
                    nn.init.uniform_(self.fc21.bias)
                    nn.init.uniform_(self.fc22.bias)

                def forward(self, x):
                    return self.fc12(self.relu(self.fc11(x))), self.fc22(self.relu(self.fc21(x)))

            if model_name == 'ResNet-18':
                self._est = torchvision.models.resnet18(pretrained=self.zoo_pretrained)
                self._est.pool0 = Identity()
                self._est.fc = DoubleOutput(self._est.fc.in_features)
            elif model_name == 'ResNet-152':
                self._est = torchvision.models.resnet152(pretrained=self.zoo_pretrained)
                self._est.pool0 = Identity()
                self._est.fc = DoubleOutput(self._est.fc.in_features)
            elif model_name == 'ResNeXt-101-32x8d':
                self._est = torchvision.models.resnext101_32x8d(pretrained=self.zoo_pretrained)
                self._est.pool0 = Identity()
                self._est.fc = DoubleOutput(self._est.fc.in_features)
            elif model_name == 'Densenet-121':
                self._est = torchvision.models.densenet121(pretrained=self.zoo_pretrained)
                self._est.features.pool0 = Identity()
                self._est.classifier = DoubleOutput(self._est.classifier.in_features)
            else:
                print(
                    'Not supported model {}, try one from list {}'.format(
                        model_name,
                        [
                            'ResNet-18',
                            'ResNet-152',
                            'ResNeXt-101-32x8d',
                            'Densenet-121'
                        ]
                    )
                )
        else:
            print('AgeGender name should be from list {}'.format(self.possible_names))

        if load_pretrained:
            self._load_est(os.path.join(self.path_pretrained, '{}.pth'.format(model_name)))

        self._est.to(self.device)

    def model(self):
        return self._est

    def _load_est(self, path):
        model_name = path.split('/')[-1][:-4]

        if model_name in self.possible_names:
            try:
                self._est.load_state_dict(torch.load(path))
            except RuntimeError:
                print('Path should lead to correct state dict file')
                return self

            if self.device is not None:
                self._est.to(self.device)
            self._est.eval()
            return self
        else:
            print('Path should lead to correct state dict file of model from list {}'.format(self.possible_names))
            return self


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, mode: str = None) -> None:
        super(ImageDataset, self).__init__()
        self.df = dataframe
        self.mode = mode
        self.image_size = IMAGE_SIZE

        transforms_list = []

        if self.mode == 'train':
            transforms_list.extend([
                transforms.RandomChoice([
                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                    transforms.RandomAffine(degrees=20, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=10),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomGrayscale(0.05)
                ])
            ])

        transforms_list.extend([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transforms = transforms.Compose(transforms_list)

        self.targets_age = dataframe['target_age'].values
        self.targets_gender = dataframe['target_gender'].values

    def __getitem__(self, index: int) -> Any:
        filename = self.df['filename'].values[index]
        image = Image.open(filename)
        assert image.mode == 'RGB'

        image = self.transforms(image)

        return image, self.targets_age[index], self.targets_gender[index]

    def __len__(self) -> int:
        return self.df.shape[0]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path: str) -> None:
        super(Dataset, self).__init__()
        self.image_path = image_path
        self.image_paths = None
        self.detected = pd.DataFrame(columns=['face_path', 'parent_image_path', 'age', 'gender'])
        self.detected_path = os.path.join(config['detected_save_path'], 'detected/{}_{}.jpeg')

        self.detector = lambda x: torch.Tensor  # replace with proper face detector
        self.is_fake = lambda x: torch.Tensor  # replace with proper validator
        self.age_gender_model = lambda x: torch.Tensor  # torch based model for predicting age and gender

    def collect_images(self) -> None:
        self.image_paths = get_data(self.image_path, collect_targets=False)

    def detect_faces(self) -> None:
        for i, image_path in enumerate(self.image_paths):
            image = self.read_image(image_path)

            faces = [image]  # here u need to detect faces from picture

            for j, face in enumerate(faces):
                path = self.detected_path.format(i, j)

                face.save(fp=path)

                self.detected = self.detected.append(
                    {
                        'image_path': image_path,
                        'face_path': path
                    },
                    ignore_index=True
                )

    def remove_fake_faces(self) -> None:
        drop = []

        for i, face_path in zip(self.detected.index, self.detected.face_path):
            image = self.read_image(face_path)
            if self.is_fake(image):
                os.remove(face_path)
                drop.append(i)

        self.detected.drop(drop, axis=0, inplace=True)

    def classify_faces_age_gender(self) -> None:
        age = torch.LongTensor()
        gender = torch.LongTensor()
        loader = load_data(self.detected.face_path, mode='inference')
        for images in loader:
            age_pred, gender_pred = self.age_gender_model(images)

            age_pred = age_pred.data.max(1, keepdim=True)[1]
            gender_pred = gender_pred.data.max(1, keepdim=True)[1]

            age = torch.cat((age, age_pred), dim=0)
            gender = torch.cat((gender, gender_pred), dim=0)

        self.detected.age = age.numpy()
        self.detected.gender = gender.numpy()

    @staticmethod
    def read_image(path, img_size=IMAGE_SIZE):
        sample = Image.open(path)
        assert sample.mode == 'RGB'
        return sample.resize((img_size, img_size))


def calc_age(taken, date_of_birth):
    date_of_birth = datetime.fromordinal(max(int(date_of_birth) - 366, 1))

    if date_of_birth.month < 7:
        return taken - date_of_birth.year
    else:
        return taken - date_of_birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)[db][0, 0]
    full_path = meta['full_path'][0]
    date_of_birth = meta['dob'][0]
    gender = meta['gender'][0]
    photo_taken = meta['photo_taken'][0]
    face_score = meta['face_score'][0]
    second_face_score = meta['second_face_score'][0]
    age = [calc_age(photo_taken[i], date_of_birth[i]) for i in range(len(date_of_birth))]

    return full_path, age, gender, face_score, second_face_score


def rename_imdb_wiki(path):
    for base in ['wiki', 'imdb']:
        total = 0
        global_path = os.path.join(path, '{}_crop'.format(base))
        full_path = os.path.join(path, '{}_crop/{}.mat'.format(base, base))
        full_path, age, gender, face_score, second_face_score = get_meta(full_path, base)
        for i in tqdm(range(len(age))):
            bad = False
            if face_score[i] < 1.:  # noize
                bad = True
            if (not np.isnan(second_face_score[i])) and second_face_score[i] > 0.:  # multiple faces
                bad = True
            if not (0 <= age[i] <= 100):  # age
                bad = True
            if np.isnan(gender[i]):  # gender
                bad = True

            filename = os.path.join(global_path, full_path[i][0])

            if Image.open(filename).mode != 'RGB':
                bad = True

            if bad:
                os.remove(filename)
            else:
                dir_name = os.path.dirname(filename)
                file_name = '{}_{}_{}'.format(i, age[i], int(gender[i])) + os.path.splitext(filename)[1]
                os.rename(src=filename, dst=os.path.join(dir_name, file_name))
                total += 1
        print('Total {} {}'.format(base, total))


def get_data(path: str, collect_targets: bool = True) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    filenames = []
    target_age = []
    target_gender = []

    for root, _, files in os.walk(path):
        for _file in files:
            filenames.append(os.path.join(root, _file))
            if collect_targets:
                age, gender = tuple(map(int, _file.split('_')[1:3]))
                target_age.append(age)
                target_gender.append(gender)

    if collect_targets:
        return np.array(filenames), np.array(target_age), np.array(target_gender)
    return np.array(filenames)


def load_data(
        path: str,
        batch_size: int = config['batch_size'],
        num_workers: int = multiprocessing.cpu_count(),
        use_gpu: bool = config['use_gpu'],
        mode: str = 'train',
        split_train_test: bool = False
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    x, y_age, y_gender = get_data(path)

    if split_train_test:

        test_idx = []
        test_size = 0.2

        for age in range(config['num_classes_age']):
            for gender in range(config['num_classes_gender']):
                mask = (y_age == age) & (y_gender == gender)
                idx = np.random.choice(
                    np.where(mask)[0],
                    size=int(np.sum(mask) * test_size),
                    replace=False
                )
                test_idx.extend(idx)

        test_idx = np.array(test_idx)
        train_idx = np.ones(len(x), dtype=np.bool)
        train_idx[test_idx] = 0

        x_train, y_age_train, y_gender_train = x[train_idx], y_age[train_idx], y_gender[train_idx]
        x_test, y_age_test, y_gender_test = x[test_idx], y_age[test_idx], y_gender[test_idx]

        train = pd.DataFrame(data={
            'filename': x_train,
            'target_age': y_age_train,
            'target_gender': y_gender_train
        })

        train = DataLoader(
            ImageDataset(train, mode='train'),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_gpu
        )

        test = pd.DataFrame(data={
            'filename': x_test,
            'target_age': y_age_test,
            'target_gender': y_gender_test
        })

        test = DataLoader(
            ImageDataset(test, mode='test'),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_gpu
        )

        return train, test
    else:
        df = pd.DataFrame(data={
            'filename': x,
            'target_age': y_age,
            'target_gender': y_gender
        })

        loader = DataLoader(
            ImageDataset(df, mode=mode),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_gpu
        )
    return loader


def print_log(msg: str, debug: bool = config['logging']):
    if debug:
        print(msg)

