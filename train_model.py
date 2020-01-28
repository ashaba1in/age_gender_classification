import argparse
import gc
import multiprocessing
import os
from typing import Tuple

import magic
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.multiprocessing
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as f
import torchvision
from torch.utils.data import DataLoader
from classes import ImageDataset

sns.set(style='darkgrid')

IMAGE_SIZE = 100
PADDING_SIZE = 4

LEARNING_RATE = 1e-4
LR_STEP = 16
LR_FACTOR = 0.1
BATCH_SIZE = 64
NUM_CLASSES = 2
NUM_WORKERS = multiprocessing.cpu_count()
NUM_EPOCHS = 2 ** 6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
cudnn.benchmark = True


def get_data(path) -> Tuple[np.ndarray, np.ndarray]:
    filenames = []
    target = []
    for root, _, files in os.walk(path):
        for _file in files:
            if magic.from_file(os.path.join(root, _file), mime=True).split('/')[0] == 'image':
                filenames.append(os.path.join(root, _file))
                target.append(int(filenames[-1].split('_')[1]))
    return np.array(filenames), np.array(target)


def load_data(path: str = 'faces/') -> DataLoader:
    x, y = get_data(path)
    
    df = pd.DataFrame(data={'filename': x, 'target': y})
    
    loader = DataLoader(ImageDataset(df, mode='train'), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    return loader


def train(data_loader, model, optimizer, criterion):
    model.train()
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()


def evaluate(data_loader, model):
    model.eval()
    loss = 0.0
    correct_fake = 0
    correct_real = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)

            loss += f.cross_entropy(outputs, labels, reduction='sum').item()
            
            pred = outputs.data.max(1, keepdim=True)[1]
            
            mask_real = labels == 1
            mask_fake = labels == 0
            
            correct_fake += pred[mask_fake].eq(labels[mask_fake].data.view_as(pred[mask_fake])).cpu().sum()
            correct_real += pred[mask_real].eq(labels[mask_real].data.view_as(pred[mask_real])).cpu().sum()
    
    loss /= len(data_loader.dataset)
    
    total_fake = np.sum(data_loader.dataset.targets == 0)
    total_real = np.sum(data_loader.dataset.targets == 1)
    
    return loss, 100. * correct_fake / total_fake, 100. * correct_real / total_real


def get_argv():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--model_name', type=str)
    
    argv = parser.parse_args()
    return argv


def main():
    argv = get_argv()
    model_name = argv.model_name
    model = None
    
    possible_names = ['ResNeXt-101-32x8d', 'WideResNet-101-2', 'WideResNet-50-2', 'ResNet-152', 'Densenet-161',
                      'ResNeXt-50-32x4d', 'ResNet-101', 'Densenet-201', 'ResNet-50', 'Densenet-169',
                      'Densenet-121', 'ResNet-34', 'ResNet-18']
    
    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()
        
        def forward(self, x):
            return x
    
    if model_name in possible_names:
        if model_name == 'ResNet-18':
            model = torchvision.models.resnet18(pretrained=True)
            model.pool0 = Identity()
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'ResNet-34':
            model = torchvision.models.resnet34(pretrained=True)
            model.pool0 = Identity()
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'ResNet-50':
            model = torchvision.models.resnet50(pretrained=True)
            model.pool0 = Identity()
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'ResNet-101':
            model = torchvision.models.resnet101(pretrained=True)
            model.pool0 = Identity()
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'ResNet-152':
            model = torchvision.models.resnet152(pretrained=True)
            model.pool0 = Identity()
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'ResNeXt-50-32x4d':
            model = torchvision.models.resnext50_32x4d(pretrained=True)
            model.pool0 = Identity()
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'ResNeXt-101-32x8d':
            model = torchvision.models.resnext101_32x8d(pretrained=True)
            model.pool0 = Identity()
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'WideResNet-50-2':
            model = torchvision.models.wide_resnet50_2(pretrained=True)
            model.pool0 = Identity()
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'WideResNet-101-2':
            model = torchvision.models.wide_resnet101_2(pretrained=True)
            model.pool0 = Identity()
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'Densenet-121':
            model = torchvision.models.densenet121(pretrained=True)
            model.features.pool0 = Identity()
            model.fc = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        elif model_name == 'Densenet-169':
            model = torchvision.models.densenet169(pretrained=True)
            model.features.pool0 = Identity()
            model.fc = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        elif model_name == 'Densenet-201':
            model = torchvision.models.densenet201(pretrained=True)
            model.features.pool0 = Identity()
            model.fc = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        elif model_name == 'Densenet-161':
            model = torchvision.models.densenet161(pretrained=True)
            model.features.pool0 = Identity()
            model.fc = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    
    loader = load_data(argv.image_path)
    gc.collect()
    
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP,
                                                   gamma=LR_FACTOR)
    
    history = []
    
    for epoch in range(argv.epochs):
        train(loader, model, optimizer, criterion)
        history.append(evaluate(loader, model))
        lr_scheduler.step(epoch)
        torch.save(model.state_dict(), 'models_data/{}.pth'.format(model_name))
    
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


if __name__ == '__main__':
    main()
