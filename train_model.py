import argparse
import gc
import multiprocessing
import os
from typing import Any, Tuple

import magic
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader

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


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe
        
        transforms_list = [
            transforms.RandomChoice([
                transforms.RandomPerspective(),
                transforms.RandomAffine(degrees=30, translate=(0.05, 0.05),
                                        scale=(0.6, 1.4), shear=10)
            ])
        ]
        
        transforms_list.extend([
            transforms.Resize((IMAGE_SIZE - PADDING_SIZE, IMAGE_SIZE - PADDING_SIZE)),
            transforms.Pad(PADDING_SIZE, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.transforms = transforms.Compose(transforms_list)
        self.targets = dataframe['target'].values
    
    def __getitem__(self, index: int) -> Any:
        filename = self.df['filename'].values[index]
        sample = Image.open(filename)
        assert sample.mode == 'RGB'
        image = self.transforms(sample)
        return image, self.df['target'].values[index]
    
    def __len__(self) -> int:
        return self.df.shape[0]


def get_filenames(path) -> Tuple[np.ndarray, np.ndarray]:
    filenames = []
    target = []
    for root, _, files in os.walk(path):
        for _file in files:
            if magic.from_file(os.path.join(root, _file), mime=True).split('/')[0] == 'image':
                filenames.append(os.path.join(root, _file))
                target.append(int(filenames[-1].split('_')[1]))
    return np.array(filenames), np.array(target)


def load_data(path: str = 'faces/') -> DataLoader:
    x, y = get_filenames(path)
    
    df = pd.DataFrame(data={'filename': x, 'target': y})
    
    loader = DataLoader(ImageDataset(df), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
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
            
            loss += F.cross_entropy(outputs, labels, reduction='sum').item()
            
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
    
    parser.add_argument('--image_path', type=str, default='faces/')
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
    if model_name in possible_names:
        if model_name == 'ResNet-18':
            model = torchvision.models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'ResNet-34':
            model = torchvision.models.resnet34(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'ResNet-50':
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'ResNet-101':
            model = torchvision.models.resnet101(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'ResNet-152':
            model = torchvision.models.resnet152(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'ResNeXt-50-32x4d':
            model = torchvision.models.resnext50_32x4d(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'ResNeXt-101-32x8d':
            model = torchvision.models.resnext101_32x8d(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'WideResNet-50-2':
            model = torchvision.models.wide_resnet50_2(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'WideResNet-101-2':
            model = torchvision.models.wide_resnet101_2(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'Densenet-121':
            model = torchvision.models.densenet121(pretrained=True)
            model.fc = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        elif model_name == 'Densenet-169':
            model = torchvision.models.densenet169(pretrained=True)
            model.fc = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        elif model_name == 'Densenet-201':
            model = torchvision.models.densenet201(pretrained=True)
            model.fc = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        elif model_name == 'Densenet-161':
            model = torchvision.models.densenet161(pretrained=True)
            model.fc = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    
    loader = load_data(argv.image_path)
    gc.collect()
    
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP,
                                                   gamma=LR_FACTOR)
    
    history = []
    
    for epoch in range(NUM_EPOCHS):
        train(loader, model, optimizer, criterion)
        history.append(evaluate(loader, model))
        lr_scheduler.step(epoch)
    
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
