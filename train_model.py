import gc
import os

import matplotlib

matplotlib.use('Agg')

import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import torch
import torch.multiprocessing
import torch.backends.cudnn as cudnn
import torch.nn.functional as f
from utils import (
	load_data,
	CenterLoss,
	Model,
	get_config,
	print_log,
)

sns.set(style='darkgrid')

config = get_config()

IMAGE_PATH = config['image_train_path']

LEARNING_RATE = config['learning_rate']
LR_STEP = config['lr_step']
LR_FACTOR = config['lr_factor']
BATCH_SIZE = config['batch_size']
NUM_CLASSES = config['num_classes']
NUM_EPOCHS = config['num_epochs']

USE_GPU = config['use_gpu']

DEVICE = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
cudnn.benchmark = True

MODELS_PATH = config['models_path']


def train(data_loader, model, optimizer, criterion):
	model.train()
	for images, labels in data_loader:
		images, labels = images.to(DEVICE), labels.to(DEVICE)
		
		optimizer.zero_grad()
		
		outputs = model(images)
		
		loss = criterion(outputs, labels)
		
		loss.backward()
		
		optimizer.step()


def evaluate(data_loader, model):
	model.eval()
	loss = 0.0
	correct = np.zeros(NUM_CLASSES, dtype=int)
	
	with torch.no_grad():
		for images, labels in data_loader:
			images, labels = images.to(DEVICE), labels.to(DEVICE)
			
			outputs = model(images)
			
			loss += f.cross_entropy(outputs, labels, reduction='sum').item()
			
			pred = outputs.data.max(1, keepdim=True)[1]
			
			for i in range(NUM_CLASSES):
				mask = labels == i
				correct[i] += pred[mask].eq(labels[mask].data.view_as(pred[mask])).cpu().sum()
	
	loss /= len(data_loader.dataset)
	
	counts = np.zeros(NUM_CLASSES)
	
	for i in range(NUM_CLASSES):
		counts[i] += np.sum(data_loader.dataset.targets == i)
	
	return loss, *(correct / counts)


def main(model_name):
	model = Model(model_name=model_name, device=DEVICE, num_classes=NUM_CLASSES).model()
	global BATCH_SIZE
	
	criterion = CenterLoss(num_classes=NUM_CLASSES, feat_dim=2, use_gpu=USE_GPU)
	
	parameters = list(model.parameters()) + list(criterion.parameters())
	
	optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_FACTOR)
	
	history = []
	epoch = 0
	bar = tqdm(total=NUM_EPOCHS)
	
	loader = load_data(path=IMAGE_PATH, batch_size=BATCH_SIZE)
	gc.collect()
	
	while epoch < NUM_EPOCHS:
		try:
			train(loader, model, optimizer, criterion)
			epoch += 1
		except RuntimeError:
			print_log('Batch size {} too large, trying {}'.format(BATCH_SIZE, BATCH_SIZE // 2))
			BATCH_SIZE //= 2
			loader = load_data(path=IMAGE_PATH, batch_size=BATCH_SIZE)
			gc.collect()
			continue
		
		history.append(evaluate(loader, model))
		
		lr_scheduler.step(epoch)
		torch.save(model.state_dict(), os.path.join(MODELS_PATH, '{}.pth'.format(model_name)))
		bar.update(1)
	
	history = np.array(history)
	
	plt.figure(figsize=(15, 15))
	plt.title('loss model {}'.format(model_name))
	plt.plot(history[:, 0], marker='.')
	plt.savefig(os.path.join(MODELS_PATH, 'loss_{}.png'.format(model_name)))
	
	plt.figure(figsize=(15, 15))
	plt.title('accuracy on fake and real')
	plt.plot(history[:, 2], marker='.', label='real')
	plt.plot(history[:, 1], marker='.', label='fake')
	plt.legend()
	plt.savefig(os.path.join(MODELS_PATH, 'accuracy_{}.png'.format(model_name)))
	
	with open(os.path.join(MODELS_PATH, 'BATCH_{}.txt'.format(model_name)), 'w') as _:
		_.write(str(BATCH_SIZE))


if __name__ == '__main__':
	main(sys.argv[1])
