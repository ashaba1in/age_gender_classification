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
    AgeGender,
    get_config,
)

sns.set(style='darkgrid')

config = get_config()

IMAGE_PATH = config['image_train_path']
MODELS_PATH = config['models_path']

LEARNING_RATE = config['learning_rate']

NUM_CLASSES_AGE = config['num_classes_age']
NUM_CLASSES_GENDER = config['num_classes_gender']

BATCH_SIZE = config['batch_size']
NUM_EPOCHS = config['num_epochs']
USE_GPU = config['use_gpu']

DEVICE = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
cudnn.benchmark = True


def train(data_loader, model, optimizer, criterions):
    criterion_age, criterion_gender = criterions

    model.train()

    for images, labels_age, labels_gender in data_loader:
        images, labels_age, labels_gender = images.to(DEVICE), labels_age.to(DEVICE), labels_gender.to(DEVICE)

        optimizer.zero_grad()

        outputs_age, outputs_gender = model(images)

        loss_model = criterion_age(outputs_age, labels_age)
        loss_model += criterion_gender(outputs_gender, labels_gender)

        loss_model.backward()

        optimizer.step()


def evaluate(data_loader, model):
    model.eval()
    losses = np.zeros(2, dtype=np.float32)
    correct_age = np.zeros(NUM_CLASSES_AGE, dtype=int)
    correct_gender = np.zeros(NUM_CLASSES_GENDER, dtype=int)

    with torch.no_grad():
        for images, labels_age, labels_gender in data_loader:
            images, labels_age, labels_gender = images.to(DEVICE), labels_age.to(DEVICE), labels_gender.to(DEVICE)

            output_age, output_gender = model(images)

            losses[0] += f.cross_entropy(output_age, labels_age, reduction='sum').item()
            losses[1] += f.cross_entropy(output_gender, labels_gender, reduction='sum').item()

            pred_age = output_age.data.max(1, keepdim=True)[1]
            for i in range(NUM_CLASSES_AGE):
                mask = labels_age == i
                correct_age[i] += pred_age[mask].eq(
                    labels_age[mask].data.view_as(pred_age[mask])).cpu().sum()

            pred_gender = output_gender.data.max(1, keepdim=True)[1]
            for i in range(NUM_CLASSES_GENDER):
                mask = labels_gender == i
                correct_gender[i] += pred_gender[mask].eq(
                    labels_gender[mask].data.view_as(pred_gender[mask])).cpu().sum()

    losses /= len(data_loader.dataset)

    counts_age = np.zeros(NUM_CLASSES_AGE)
    for i in range(NUM_CLASSES_AGE):
        counts_age[i] += np.sum(data_loader.dataset.targets_age == i)

    counts_gender = np.zeros(NUM_CLASSES_GENDER)
    for i in range(NUM_CLASSES_GENDER):
        counts_gender[i] += np.sum(data_loader.dataset.targets_gender == i)

    return losses, correct_age / counts_age, correct_gender / counts_gender


def main(model_name):
    model = AgeGender(model_name=model_name, device=DEVICE).model()
    global BATCH_SIZE

    criterion_age = torch.nn.CrossEntropyLoss()
    criterion_gender = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        'loss': [],
        'age': [],
        'gender': []
    }

    epoch = 0
    bar = tqdm(total=NUM_EPOCHS)

    loader = load_data(path=IMAGE_PATH, batch_size=BATCH_SIZE)
    gc.collect()

    while epoch < NUM_EPOCHS:
        try:
            train(loader, model, optimizer, [criterion_age, criterion_gender])
            epoch += 1
        except RuntimeError as e:
            if 'CUDA' in str(e):
                print('BATCH SIZE {} too big, trying {}'.format(BATCH_SIZE, BATCH_SIZE // 2))
                BATCH_SIZE //= 2
                loader = load_data(path=IMAGE_PATH, batch_size=BATCH_SIZE)
                gc.collect()
                continue
            else:
                raise e

        inference = evaluate(loader, model)
        history['loss'].append(inference[0])
        history['age'].append(inference[1])
        history['gender'].append(inference[2])

        torch.save(model.state_dict(), os.path.join(MODELS_PATH, '{}.pth'.format(model_name)))
        bar.update(1)

    for key in history.keys():
        history[key] = np.array(history[key])

    plt.figure(figsize=(20, 20))
    plt.title('loss model {}'.format(model_name))
    plt.plot(history['loss'][:, 0], marker='.', label='age')
    plt.plot(history['loss'][:, 1], marker='.', label='gender')
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, 'loss_{}.png'.format(model_name)))

    plt.figure(figsize=(20, 20))
    plt.title('accuracy age model {}'.format(model_name))
    for i in range(NUM_CLASSES_AGE):
        plt.plot(history['age'][:, i], marker='.', label='class {}'.format(i))
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, 'accuracy_age_{}.png'.format(model_name)))

    plt.figure(figsize=(20, 20))
    plt.title('accuracy gender model {}'.format(model_name))
    for i in range(NUM_CLASSES_GENDER):
        plt.plot(history['gender'][:, i], marker='.', label='class {}'.format(i))
    plt.legend()
    plt.savefig(os.path.join(MODELS_PATH, 'accuracy_gender_{}.png'.format(model_name)))

    with open(os.path.join(MODELS_PATH, 'BATCH_{}.txt'.format(model_name)), 'w') as _:
        _.write(str(BATCH_SIZE))


if __name__ == '__main__':
    main(sys.argv[1])
