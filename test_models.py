import argparse
import multiprocessing
import os
import time
from typing import Any

import magic
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
from checker import LivenessChecker
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

IMAGE_SIZE = 100
PADDING_SIZE = 4
BATCH_SIZE = 256
NUM_CLASSES = 2
NUM_WORKERS = multiprocessing.cpu_count()


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe
        
        transforms_list = [
            transforms.Resize((IMAGE_SIZE - PADDING_SIZE, IMAGE_SIZE - PADDING_SIZE)),
            transforms.Pad(PADDING_SIZE, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        
        self.transforms = transforms.Compose(transforms_list)
        self.targets = dataframe['target'].values
    
    def __getitem__(self, index: int) -> Any:
        filename = self.df['filename'].values[index]
        sample = Image.open(filename)
        assert sample.mode == 'RGB'
        image = self.transforms(sample)
        
        return image, self.df['target'].values[index], filename
    
    def __len__(self) -> int:
        return self.df.shape[0]


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
    checker = LivenessChecker(path1='model_{}.pth'.format(argv.model_name), debug=True, device=device)
    
    global_start = 0
    if argv.count_time:
        global_start = time.perf_counter()
    mistakes_human1 = 0
    mistakes_human0 = 0
    total_human0 = 0
    total_human1 = 0
    mistakes_mem0 = 0
    mistakes_mem1 = 0
    mistakes_ad0 = 0
    mistakes_ad1 = 0
    x, y = get_data(get_filenames(argv.image_path))
    dataset = ImageDataset(pd.DataFrame(data={'filename': x, 'target': y}))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    for images, labels, filenames in tqdm(loader, disable=True):
        total_human0 += len(labels) - labels.sum()
        total_human1 += labels.sum()
        
        images, labels = images.to(device), labels.to(device)
        
        pred_mem, _ = checker.check_liveness(images)
        # mistakes_ad = check_mistakes(pred_ads, labels.numpy())
        mistakes_mem = check_mistakes(pred_mem, labels.cpu().numpy())
        # mistakes_total = check_mistakes(np.logical_and(pred_mem, pred_ads), labels.numpy())
        mistakes_total = mistakes_mem
        # mistakes_ad0 += mistakes_ad[0]
        # mistakes_ad1 += mistakes_ad[1]
        mistakes_mem0 += mistakes_mem[0]
        mistakes_mem1 += mistakes_mem[1]
        mistakes_human0 += mistakes_total[0]
        mistakes_human1 += mistakes_total[1]
    
    print('total pictures {} with {:.5%} correct'.format(
        total_human0 + total_human1,
        1 - (mistakes_human0 + mistakes_human1) / (total_human0 + total_human1)))
    
    if total_human1 != 0:
        print('human correct {:.5%}'.format(1 - mistakes_human1 / total_human1))
    if total_human0 != 0:
        print('not human correct {:.5%}'.format(1 - mistakes_human0 / total_human0))
    
    print('ad model {} mistakes out of {} on fake'.format(mistakes_ad0, total_human0))
    print('ad model {} mistakes out of {} on real'.format(mistakes_ad1, total_human1))
    print('mem model {} mistakes out of {} on fake'.format(mistakes_mem0, total_human0))
    print('mem model {} mistakes out of {} on real'.format(mistakes_mem1, total_human1))
    
    if argv.count_time:
        print('total time: {:.5f} seconds'.format(time.perf_counter() - global_start))
    
    pd.DataFrame({'model': argv.model_name,
                  'fake_ad': 100 * mistakes_ad0 / total_human0,
                  'fake_mem': 100 * mistakes_mem0 / total_human0,
                  'real_ad': 100 * mistakes_ad1 / total_human1,
                  'real_mem': 100 * mistakes_mem1 / total_human1,
                  'total_time': time.perf_counter() - global_start}).to_csv('results.csv', mode='a', header=False)


if __name__ == '__main__':
    main()
