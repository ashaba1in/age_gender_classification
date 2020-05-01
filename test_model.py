import argparse
import gc
import multiprocessing
import time

import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.multiprocessing
from tqdm import tqdm

from utils import (
	load_data,
	Model,
)

IMAGE_SIZE = 100
PADDING_SIZE = 4
BATCH_SIZE = 4096
NUM_CLASSES = 2
NUM_WORKERS = multiprocessing.cpu_count()

USE_GPU = True

DEVICE = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
cudnn.benchmark = True


def get_argv():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--image_path', type=str, default='images')
	parser.add_argument('--model_name', type=str)
	parser.add_argument('--batch_size', type=int)
	parser.add_argument('--count_time', type=bool, default=True)
	
	argv = parser.parse_args()
	return argv


def main():
	argv = get_argv()
	
	global BATCH_SIZE
	BATCH_SIZE = argv.batch_size
	
	model_name = argv.model_name
	
	model = Model(model_name=model_name, device=DEVICE, num_classes=NUM_CLASSES)
	model.load_est('models_data/{}.pth'.format(model_name))
	model = model.model()
	
	global_start = 0
	
	if argv.count_time:
		global_start = time.perf_counter()
	
	loader = load_data(
		path=argv.image_path,
		BATCH_SIZE=BATCH_SIZE,
		NUM_WORKERS=NUM_WORKERS,
		USE_GPU=USE_GPU
	)
	gc.collect()
	
	correct = np.zeros(NUM_CLASSES)
	
	for images, labels in tqdm(loader, disable=True):
		images, labels = images.to(DEVICE), labels.to(DEVICE)
		
		pred = model(images).data.max(1, keepdim=True)[1]
		for i in range(NUM_CLASSES):
			mask = labels == i
			correct[i] += pred[mask].eq(labels[mask].data.view_as(pred[mask])).cpu().sum()
	
	counts = np.zeros(NUM_CLASSES)
	
	for i in range(NUM_CLASSES):
		counts[i] += np.sum(loader.dataset.targets == i)
	
	print('total pictures {} with {:.5%} correct'.format(
		len(loader.dataset),
		np.sum(correct / counts)))
	
	for i in range(NUM_CLASSES):
		print('CLASS {} correct {:.5%}'.format(i, correct[i] / counts[i]))
	
	if argv.count_time:
		print('total time: {:.5f} seconds'.format(time.perf_counter() - global_start))
	
	dct = {
		'model': argv.model_name,
		'total_time': time.perf_counter() - global_start
	}
	
	for i in range(NUM_CLASSES):
		dct['CLASS {}'.format(i)] = correct[i] / counts[i]
	
	pd.DataFrame(dct).to_csv('results.csv', mode='a', header=False)


if __name__ == '__main__':
	main()
