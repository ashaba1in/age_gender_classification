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
import torch.utils.data
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
                transforms.Resize(size=(self.image_size, self.image_size)),
                transforms.RandomCrop(size=(self.crop_size, self.crop_size)),
                transforms.RandomHorizontalFlip()
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
        image = Image.open(filename).convert('RGB')

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

    def gaussian_prob(true, var=AGE_SHIFT):
        x = np.arange(0, MAX_AGE + 1, 1)
        p = np.exp(-np.square(x - true) / (2 * var ** 2)) / (var * (2 * np.pi ** 0.5))
        return p / p.sum()

    age_vectors = [
        gaussian_prob(x) for x in range(0, MAX_AGE + 1)
    ]

    for root, _, files in os.walk(path):
        for filename in files:
            filenames.append(os.path.join(root, filename))
            if collect_targets:
                if db == 'imdb_wiki':
                    try:
                        age, gender = tuple(map(int, filename.split('_')[1:3]))
                    except ValueError:
                        continue
                else:
                    age, gender = tuple(map(int, filename.split('/')[-1].split('_')[0:2]))
                    gender = 1 - gender

                target_age.append(age_vectors[age])
                target_gender.append(gender)

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
