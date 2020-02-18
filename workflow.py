import argparse
from classes import Dataset
import pandas as pd
import os
import numpy as np
import magic
import multiprocessing
import dlib

from typing import Tuple
from torch.utils.data import DataLoader
from torch.utils.data import dataset


BATCH_SIZE = 4096
NUM_WORKERS = multiprocessing.cpu_count()


def get_images(path) -> np.ndarray:
    filenames = []
    for root, _, files in os.walk(path):
        for _file in files:
            if magic.from_file(os.path.join(root, _file), mime=True).split('/')[0] == 'image':
                filenames.append(os.path.join(root, _file))

    return np.array(filenames)


def load_data(path: str = 'faces/') -> Dataset:
    filenames = get_images(path)
    return Dataset(filenames)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--image_path', type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = load_data(args.image_path)
    data.detect_faces()


if __name__ == "__main__":
    main()
