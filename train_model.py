import argparse
import gc
import multiprocessing

import matplotlib

matplotlib.use('Agg')

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
)

sns.set(style='darkgrid')

IMAGE_SIZE = 100
PADDING_SIZE = 4

LEARNING_RATE = 1e-4
LR_STEP = 16
LR_FACTOR = 0.1
BATCH_SIZE = 4096
NUM_CLASSES = 2
NUM_WORKERS = multiprocessing.cpu_count()
NUM_EPOCHS = 2 ** 6

USE_GPU = True

DEVICE = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
cudnn.benchmark = True


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
	
	return loss, correct / counts


def get_argv():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--image_path', type=str)
	parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
	parser.add_argument('--model_name', type=str)
	
	argv = parser.parse_args()
	return argv


def main():
	argv = get_argv()
	model_name = argv.model_name
	model = Model(model_name=model_name, device=DEVICE, num_classes=NUM_CLASSES).model()
	global BATCH_SIZE
	
	loader = load_data(
		path=argv.image_path,
		BATCH_SIZE=BATCH_SIZE,
		NUM_WORKERS=NUM_WORKERS,
		USE_GPU=USE_GPU
	)
	gc.collect()
	
	criterion = CenterLoss(num_classes=NUM_CLASSES, feat_dim=2, use_gpu=USE_GPU)
	
	parameters = list(model.parameters()) + list(criterion.parameters())
	
	optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_FACTOR)
	
	history = []
	epoch = 0
	bar = tqdm(total=argv.epochs)
	
	while epoch < argv.epochs:
		try:
			train(loader, model, optimizer, criterion)
			epoch += 1
		except RuntimeError:
			print('Batch size {} too large :D'.format(BATCH_SIZE))
			BATCH_SIZE //= 2
			loader = load_data(
				path=argv.image_path,
				BATCH_SIZE=BATCH_SIZE,
				NUM_WORKERS=NUM_WORKERS,
				USE_GPU=USE_GPU
			)
			gc.collect()
			continue
		
		history.append(evaluate(loader, model))
		
		lr_scheduler.step(epoch)
		torch.save(model.state_dict(), 'models_data/{}.pth'.format(model_name))
		bar.update(1)
	
	history = np.array(history)
	
	plt.figure(figsize=(15, 15))
	plt.title('loss model {}'.format(model_name))
	plt.plot(history[:, 0], marker='.')
	plt.savefig('models_data/loss_{}.png'.format(model_name))
	
	plt.figure(figsize=(15, 15))
	plt.title('accuracy on fake and real')
	plt.plot(history[:, 2], marker='.', label='real')
	plt.plot(history[:, 1], marker='.', label='fake')
	plt.legend()
	plt.savefig('models_data/accuracy_{}.png'.format(model_name))
	
	torch.save(model.state_dict(), 'models_data/{}.pth'.format(model_name))
	
	with open('models_data/{}_BATCH.txt'.format(model_name), 'w') as f:
		f.write(str(BATCH_SIZE))


if __name__ == '__main__':
	main()
