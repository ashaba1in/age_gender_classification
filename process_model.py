import gc
import os
import time

import matplotlib
import pandas as pd

matplotlib.use('Agg')

import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import torch
import torch.multiprocessing
import torch.backends.cudnn as cudnn
from utils import (
    load_data,
    AgeGender,
    get_config,
)

sns.set(style='darkgrid')

config = get_config()

IMAGE_PATH = config['images_path']
MODELS_PATH = config['models_path']
GRAPHS_PATH = config['graphs_path']

LEARNING_RATE = config['learning_rate']
WEIGHT_DECAY = config['l2']

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


def evaluate(data_loader, model, criterions):
    model.eval()
    losses = np.zeros(2, dtype=np.float32)
    mae_age = 0.
    correct_gender = 0.

    criterion_age, criterion_gender = criterions

    with torch.no_grad():
        for images, labels_age, labels_gender in data_loader:
            images, labels_age, labels_gender = images.to(DEVICE), labels_age.to(DEVICE), labels_gender.to(DEVICE)

            output_age, output_gender = model(images)

            losses[0] += criterion_age(output_age, labels_age).item()
            losses[1] += criterion_gender(output_gender, labels_gender).item()

            pred_age = output_age.data.max(1, keepdim=True)[1]
            mae_age += torch.sum(torch.abs(pred_age - labels_age.data.view_as(pred_age)))

            pred_gender = output_gender.data.max(1, keepdim=True)[1]
            correct_gender += torch.sum(pred_gender.eq(labels_gender.data.view_as(pred_gender)).cpu())

    losses /= len(data_loader.dataset)
    mae_age /= len(data_loader.dataset)
    correct_gender /= len(data_loader.dataset)

    return losses, mae_age, correct_gender


def main(model_name):
    model = AgeGender(model_name=model_name, device=DEVICE).model()
    global BATCH_SIZE

    criterion_age = torch.nn.CrossEntropyLoss()
    criterion_gender = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = {
        'loss_train': [],
        'age_train': [],
        'gender_train': [],
        'loss_test': [],
        'age_test': [],
        'gender_test': []
    }

    epoch = 0
    bar = tqdm(total=NUM_EPOCHS)

    loader_train, loader_test = load_data(path=IMAGE_PATH, batch_size=BATCH_SIZE, split_train_test=True)
    gc.collect()

    while epoch < NUM_EPOCHS:
        try:
            train(loader_train, model, optimizer, [criterion_age, criterion_gender])
            epoch += 1
        except RuntimeError as e:
            if 'CUDA' in str(e):
                print('BATCH SIZE {} too big, trying {}'.format(BATCH_SIZE, BATCH_SIZE // 2))
                BATCH_SIZE //= 2
                loader_train, loader_test = load_data(path=IMAGE_PATH, batch_size=BATCH_SIZE, split_train_test=True)
                gc.collect()
                continue
            else:
                print(e)
                return

        inference_train = evaluate(loader_train, model, [criterion_age, criterion_gender])
        history['loss_train'].append(inference_train[0])
        history['age_train'].append(inference_train[1])
        history['gender_train'].append(inference_train[2])

        inference_test = evaluate(loader_test, model, [criterion_age, criterion_gender])
        history['loss_test'].append(inference_test[0])
        history['age_test'].append(inference_test[1])
        history['gender_test'].append(inference_test[2])

        torch.save(model.state_dict(), os.path.join(MODELS_PATH, '{}.pth'.format(model_name)))
        bar.update(1)

    for key in history.keys():
        history[key] = np.array(history[key])

    plt.figure(figsize=(20, 20))
    plt.title('loss age model {}'.format(model_name))
    plt.plot(history['loss_train'][:, 0], marker='.', label='train')
    plt.plot(history['loss_test'][:, 0], marker='.', label='test')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_PATH, 'loss_age_{}.png'.format(model_name)))

    plt.figure(figsize=(20, 20))
    plt.title('loss gender model {}'.format(model_name))
    plt.plot(history['loss_train'][:, 1], marker='.', label='train')
    plt.plot(history['loss_test'][:, 1], marker='.', label='test')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_PATH, 'loss_gender_{}.png'.format(model_name)))

    plt.figure(figsize=(20, 20))
    plt.title('MAE age model {}'.format(model_name))
    plt.plot(history['age_train'], marker='.', label='train')
    plt.plot(history['age_test'], marker='.', label='test')
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_PATH, 'MAE_age_{}.png'.format(model_name)))

    plt.figure(figsize=(20, 20))
    plt.title('accuracy gender model {}'.format(model_name))
    plt.plot(history['gender_train'], marker='.', label='train')
    plt.plot(history['gender_test'], marker='.', label='test')
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_PATH, 'accuracy_gender_{}.png'.format(model_name)))

    del loader_train, loader_test

    torch.cuda.empty_cache()

    gc.collect()

    loader = load_data(path=IMAGE_PATH, batch_size=BATCH_SIZE)

    start = time.perf_counter()

    _, age_mae, gender_accuracy = evaluate(loader, model, [criterion_age, criterion_gender])

    df = {
        'time': time.perf_counter() - start,
        'age_mae': age_mae,
        'gender_accuracy': gender_accuracy
    }

    pd.DataFrame(data=df, index=[model_name]).to_csv(os.path.join(MODELS_PATH, 'results.csv'), mode='a', header=False)


if __name__ == '__main__':
    main(sys.argv[1])
