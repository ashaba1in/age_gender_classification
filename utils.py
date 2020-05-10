import json
import multiprocessing
import os
import shutil
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
CROP_SIZE = config['crop_size']

NUM_CLASSES_AGE = config['num_classes_age']
NUM_CLASSES_GENDER = config['num_classes_gender']
MAX_AGE = config['max_age']
AGE_SHIFT = config['age_shift']

FREEZE_LAYERS = config['freeze']


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
            class DoubleOutput(nn.Module):
                def __init__(self, in_features: int):
                    super(DoubleOutput, self).__init__()
                    self.fc1 = nn.Linear(in_features, in_features)
                    self.fc12 = nn.Linear(in_features, config['num_classes_age'])
                    self.fc2 = nn.Linear(in_features, in_features // 2)
                    self.fc22 = nn.Linear(in_features // 2, config['num_classes_gender'])
                    self.act = nn.ReLU()
                    self.dr = nn.Dropout(p=config['dropout'])

                def forward(self, x):
                    age = self.fc12(self.dr(self.act(self.fc1(self.dr(x)))))
                    gender = self.fc22(self.dr(self.act(self.fc2(self.dr(x)))))

                    return age, gender

            if model_name == 'ResNet-18':
                self._est = torchvision.models.resnet18(pretrained=self.zoo_pretrained)
                self._est.fc = DoubleOutput(self._est.fc.in_features)
                if FREEZE_LAYERS:
                    self._est.conv1.requires_grad = False
            elif model_name == 'ShuffleNet':
                self._est = torchvision.models.shufflenet_v2_x0_5(pretrained=self.zoo_pretrained)
                self._est.fc = DoubleOutput(self._est.fc.in_features)
            elif model_name == 'MobileNet_v2':
                self._est = torchvision.models.mobilenet_v2(pretrained=self.zoo_pretrained)
                self._est.classifier = DoubleOutput(self._est.classifier[1].in_features)
            elif model_name == 'Densenet-121':
                self._est = torchvision.models.densenet121(pretrained=self.zoo_pretrained)
                self._est.classifier = DoubleOutput(self._est.classifier.in_features)
            else:
                print(
                    'Not supported model {}, try one from list {}'.format(
                        model_name,
                        [
                            'ShuffleNet',
                            'MobileNet_v2',
                            'ResNet-18',
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


def rand_init_layer(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, np.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict, mode: str = None) -> None:
        super(ImageDataset, self).__init__()
        self.data = data
        self.mode = mode
        self.image_size = IMAGE_SIZE
        self.crop_size = CROP_SIZE

        transforms_list = []

        if self.mode == 'train':
            transforms_list.extend([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomResizedCrop(size=self.crop_size, scale=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.1),
                transforms.RandomGrayscale(p=0.1)
            ])
        else:
            transforms_list.extend([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.CenterCrop((self.crop_size, self.crop_size)),
            ])

        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transforms = transforms.Compose(transforms_list)

        self.targets_age = self.data['target_age']
        self.targets_gender = self.data['target_gender']

    def __getitem__(self, index: int) -> Any:
        filename = self.data['filenames'][index]
        image = Image.open(filename)

        image = self.transforms(image)

        if self.mode == 'inference':
            return image
        return image, self.targets_age[index], self.targets_gender[index]

    def __len__(self) -> int:
        return len(self.data['filenames'])


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
        self.image_paths = get_data(self.image_path, collect_targets=False, db='')

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
        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')
        return sample.resize((img_size, img_size))


class ServerSetter:
    def __init__(self):
        self.remove = os.remove
        self.rename = os.rename
        self.chdir = os.chdir
        self.mkdir = os.mkdir
        self.move = shutil.move
        self.exec = os.system
        self.join = os.path.join
        self.paths = config['paths']
        self.links = config['links']

    def init_directories(self):
        for path_name in self.paths.keys():
            try:
                self.mkdir(self.paths[path_name])
            except FileExistsError:
                pass

    def download_unpack_data(self):
        os.chdir(self.paths['base_path'])
        for db in self.links.keys():
            link = self.links[db]
            print_log('Processing db {}'.format(db))
            self.exec('wget {}'.format(link))
            print_log('db {} downloaded...'.format(db))
            self.exec('tar xf {}_crop.tar --no-same-owner'.format(db))
            print_log('db {} unpacked...'.format(db))

    def prepare_data(self):
        self._delete_noise()

    def _delete_noise(self):
        for db in self.links.keys():
            total = 0
            self.chdir(self.join(self.paths['base_path'], '{}_crop'.format(db)))
            full_path, age, gender, face_score, second_face_score = self._get_meta('{}.mat'.format(db), db)
            for i in tqdm(range(len(age)), desc=db):
                bad = False
                if face_score[i] < 1.:  # noise
                    bad = True
                elif (not np.isnan(second_face_score[i])) and second_face_score[i] > 0.:  # multiple faces
                    bad = True
                elif not (0 <= age[i] <= MAX_AGE):  # age
                    bad = True
                elif np.isnan(gender[i]):  # gender
                    bad = True

                if bad:
                    self.remove(full_path[i][0])
                else:
                    self.rename(src=full_path[i][0], dst='{}_{}_{}_.jpg'.format(i, age[i], int(gender[i])))
                    total += 1
            self.remove('{}.mat'.format(db))
            print('extracted {} images from {}'.format(total, db))
            self.move(src='{}_crop'.format(db), dst=self.paths['train_path'])

    def _get_meta(self, mat_path, db):
        meta = loadmat(mat_path)[db][0, 0]
        full_path = meta['full_path'][0]
        date_of_birth = meta['dob'][0]
        gender = meta['gender'][0]
        photo_taken = meta['photo_taken'][0]
        face_score = meta['face_score'][0]
        second_face_score = meta['second_face_score'][0]
        age = [self._calc_age(photo_taken[i], date_of_birth[i]) for i in range(len(date_of_birth))]

        return full_path, age, gender, face_score, second_face_score

    @staticmethod
    def _calc_age(taken, date_of_birth):
        date_of_birth = datetime.fromordinal(max(int(date_of_birth) - 366, 1))

        if date_of_birth.month < 7:
            return taken - date_of_birth.year
        else:
            return taken - date_of_birth.year - 1


def get_data(path: str, db: str, collect_targets: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    filenames = []
    target_age = []
    target_gender = []

    for root, _, files in os.walk(path):
        for filename in files:
            filenames.append(os.path.join(root, filename))
            if collect_targets:
                if db == 'imdb_wiki':
                    age, gender = tuple(map(int, filename.split('_')[1:3]))
                else:
                    age, gender = tuple(map(int, filename.split('/')[-1].split('_')[0:2]))
                    gender = 1 - gender

                age_vector = np.zeros(NUM_CLASSES_AGE, dtype=np.float32)
                gender_vector = np.zeros(NUM_CLASSES_GENDER, dtype=np.float32)

                for i in range(-AGE_SHIFT, AGE_SHIFT + 1):
                    if 0 <= age + i <= MAX_AGE:
                        age_vector[age + i] = 1. / (1. + abs(i)) ** 2

                gender_vector[gender] = 1.

                target_age.append(age_vector)
                target_gender.append(gender_vector)

    if collect_targets:
        return np.array(filenames), np.array(target_age), np.array(target_gender)
    return np.array(filenames), np.empty(0), np.empty(0)


def load_data(
        path: str,
        batch_size: int = config['batch_size'],
        mode: str = 'train',
        db: str = 'imdb_wiki',
        collect_targets: bool = True
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    x, y_age, y_gender = get_data(path, collect_targets=collect_targets, db=db)

    data = {
        'filenames': x,
        'target_age': y_age,
        'target_gender': y_gender
    }

    loader = DataLoader(
        ImageDataset(data, mode=mode),
        batch_size=batch_size,
        shuffle=mode == 'train',
        num_workers=multiprocessing.cpu_count(),
        pin_memory=config['use_gpu']
    )

    return loader


def print_log(msg: str, debug: bool = config['logging']):
    if debug:
        print(msg)
