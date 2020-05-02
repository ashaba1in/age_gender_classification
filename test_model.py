import gc
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.multiprocessing
from tqdm import tqdm

from utils import (
	get_config,
	load_data,
	Model,
	print_log,
)

config = get_config()

IMAGE_PATH = config['image_test_path']
NUM_CLASSES = config['num_classes']
NUM_WORKERS = multiprocessing.cpu_count()

USE_GPU = config['use_gpu']

DEVICE = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
cudnn.benchmark = True

MODELS_PATH = config['models_path']


def test(data_loader, model):
	correct = np.zeros(NUM_CLASSES)
	
	for images, labels in tqdm(data_loader, disable=True):
		images, labels = images.to(DEVICE), labels.to(DEVICE)
		
		pred = model(images).data.max(1, keepdim=True)[1]
		for i in range(NUM_CLASSES):
			mask = labels == i
			correct[i] += pred[mask].eq(labels[mask].data.view_as(pred[mask])).cpu().sum()
	
	counts = np.zeros(NUM_CLASSES)
	
	for i in range(NUM_CLASSES):
		counts[i] += np.sum(data_loader.dataset.targets == i)
	
	return correct / counts


def main(model_name):
	with open(os.path.join(MODELS_PATH, 'BATCH_{}.txt'.format(model_name)), 'r') as _:
		BATCH_SIZE = int(_.read())
	
	model = Model(
		model_name=model_name,
		device=DEVICE,
		num_classes=NUM_CLASSES,
		load_pretrained=True,
		path_pretrained=MODELS_PATH
	).model()
	
	global_start = time.perf_counter()
	
	loader = load_data(path=IMAGE_PATH, batch_size=BATCH_SIZE)
	gc.collect()
	
	results = test(loader, model)
	
	print_log('total pictures {} with {:.5%} correct'.format(
		len(loader.dataset),
		np.sum(results)))
	
	for i in range(NUM_CLASSES):
		print_log('CLASS {} correct {:.5%}'.format(i, results[i]))
	
	print_log('total time: {:.5f} seconds'.format(time.perf_counter() - global_start))
	
	dct = {
		'model': model_name,
		'total_time': time.perf_counter() - global_start
	}
	
	for i in range(NUM_CLASSES):
		dct['CLASS_{}'.format(i)] = results[i]
	
	pd.DataFrame(dct).to_csv('results.csv', mode='a', header=False)


if __name__ == '__main__':
	main(sys.argv[1])
