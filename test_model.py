import argparse
import multiprocessing
import os
import time

import magic
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes import ImageDataset, LivenessChecker

IMAGE_SIZE = 100
PADDING_SIZE = 4
BATCH_SIZE = 256
NUM_CLASSES = 2
NUM_WORKERS = multiprocessing.cpu_count()


def get_argv():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--image_path', type=str, default='images')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--count_time', type=bool, default=True)
    
    argv = parser.parse_args()
    return argv


def get_filenames(path):
    filenames = []
    for root, _, files in os.walk(path):
        for _file in files:
            if magic.from_file(os.path.join(root, _file), mime=True).split('/')[0] == 'image':
                filenames.append(os.path.join(root, _file))
    return filenames


def get_data(filenames, train=True):
    images = []
    target = []
    for filename in filenames:
        images.append(filename)
        target.append(int(filename.split('_')[1]))
    if train:
        return np.array(images), np.array(target)
    else:
        return np.array(images)


def check_mistakes(pred, labels):
    mask_real = labels == 1
    mask_fake = labels == 0
    
    correct_fake = np.sum(pred[mask_fake] == labels[mask_fake])
    correct_real = np.sum(pred[mask_real] == labels[mask_real])
    
    return mask_fake.sum() - correct_fake, mask_real.sum() - correct_real


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.benchmark = True

    argv = get_argv()
    checker = LivenessChecker(path='models_data/{}.pth'.format(argv.model_name), device=device)
    
    global_start = 0
    if argv.count_time:
        global_start = time.perf_counter()
    mistakes_human1 = 0
    mistakes_human0 = 0
    total_human0 = 0
    total_human1 = 0
    x, y = get_data(get_filenames(argv.image_path))
    dataset = ImageDataset(pd.DataFrame(data={'filename': x, 'target': y}), mode='test')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    for images, labels in tqdm(loader, disable=True):
        total_human0 += len(labels) - labels.sum()
        total_human1 += labels.sum()
    
        images, labels = images.to(device), labels.to(device)
    
        pred = checker.check_liveness(images)
        mistakes = check_mistakes(pred, labels.cpu().numpy())
        mistakes_human0 += mistakes[0]
        mistakes_human1 += mistakes[1]
    
    print('total pictures {} with {:.5%} correct'.format(
        total_human0 + total_human1,
        1 - (mistakes_human0 + mistakes_human1) / (total_human0 + total_human1)))
    
    if total_human1 != 0:
        print('human correct {:.5%}'.format(1 - mistakes_human1 / total_human1))
    if total_human0 != 0:
        print('not human correct {:.5%}'.format(1 - mistakes_human0 / total_human0))
    
    if argv.count_time:
        print('total time: {:.5f} seconds'.format(time.perf_counter() - global_start))
    
    pd.DataFrame({'model': argv.model_name,
                  'real': 100 * mistakes_human1 / total_human1,
                  'fake': 100 * mistakes_human0 / total_human1,
                  'total_time': time.perf_counter() - global_start}).to_csv('results.csv', mode='a', header=False)


if __name__ == '__main__':
    main()
