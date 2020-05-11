import gc
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')

import sys

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.autonotebook import tqdm
import torch
from torch import (
    save,
    sum,
    abs,
)
import torch.multiprocessing
import torch.backends.cudnn as cudnn
from utils import (
    load_data,
    get_config,
    print_log,
)

from model import Model

sns.set(style='darkgrid')

config = get_config()

paths = config['paths']

TRAIN_PATH = paths['train_path']
TEST_PATH = paths['test_path']
MODELS_PATH = paths['models_path']
GRAPHS_PATH = paths['graphs_path']

LEARNING_RATE = config['learning_rate']
WEIGHT_DECAY = config['l2']

NUM_CLASSES_AGE = config['num_classes_age']
NUM_CLASSES_GENDER = config['num_classes_gender']
AGE_SHIFT = config['age_shift']

BATCH_SIZE = config['batch_size']
NUM_EPOCHS = config['num_epochs']
USE_GPU = config['use_gpu']
LOGGING = config['logging']

DEVICE = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')
cudnn.benchmark = True

bar_epochs = None
bar_train = None
bar_inference = None

age_multiplier = torch.from_numpy(np.arange(config['num_classes_age'])).to(DEVICE)


def update_bars(msg: str, len1: int = None, len2: int = None):
    if not LOGGING:
        return
    global bar_epochs, bar_train, bar_inference
    if msg == 'init':
        bar_epochs = tqdm(total=NUM_EPOCHS, desc='EPOCHS', position=0, file=sys.stdout)
        bar_train = tqdm(total=len1, desc='TRAIN', position=1, file=sys.stdout)
        bar_inference = tqdm(total=len1 + len2, desc='INFERENCE', position=2, file=sys.stdout)
    if msg == 'epochs':
        bar_epochs.update(1)
    if msg == 'train':
        bar_train.update(1)
    if msg == 'inference':
        bar_inference.update(1)
    if msg == 'reset_train':
        bar_train.reset()
    if msg == 'reset_inference':
        bar_inference.reset()


def train(data_loader, model, optimizer, criterions):
    criterion_age, criterion_gender = criterions

    model.train()

    for images, labels_age, labels_gender in data_loader:
        images, labels_age, labels_gender = images.to(DEVICE), labels_age.to(DEVICE), labels_gender.to(DEVICE)

        optimizer.zero_grad()

        output_age, output_gender = model(images)

        loss = criterion_age(output_age.double(), labels_age.double())
        loss += criterion_gender(output_gender, labels_gender)

        loss.backward()

        optimizer.step()

        update_bars('train')


def evaluate(data_loader, model, criterions):
    model.eval()

    loss = 0.
    cs_age = 0.
    mae_age = 0.
    correct_gender = 0.

    criterion_age, criterion_gender = criterions

    with torch.no_grad():
        for images, labels_age, labels_gender in data_loader:
            images, labels_age, labels_gender = images.to(DEVICE), labels_age.to(DEVICE), labels_gender.to(DEVICE)

            output_age, output_gender = model(images)

            loss += criterion_age(output_age.double(), labels_age.double())
            loss += criterion_gender(output_gender, labels_gender)

            pred_age = sum(
                output_age * age_multiplier,
                dim=1,
                keepdim=True
            )
            true_age = labels_age.data.max(1, keepdim=True)[1]

            shift = abs(pred_age - true_age).cpu()

            cs_age += sum(shift <= AGE_SHIFT)

            mae_age += sum(shift)

            pred_gender = output_gender.data.max(1, keepdim=True)[1]
            correct_gender += sum(pred_gender.eq(labels_gender.data.view_as(pred_gender)).cpu())

            update_bars('inference')

    loss /= len(data_loader.dataset)
    cs_age /= len(data_loader.dataset)
    mae_age /= len(data_loader.dataset)
    correct_gender /= len(data_loader.dataset)

    return loss, [cs_age, mae_age], correct_gender


def plot_results(history, model_name):
    plt.figure(figsize=(16, 16))
    plt.title('loss model {}'.format(model_name))
    plt.plot(history['loss_train'], marker='.', label='train')
    plt.plot(history['loss_test'], marker='.', label='test')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_PATH, 'loss_{}.png'.format(model_name)))

    plt.figure(figsize=(16, 16))
    plt.title('CS age model {}'.format(model_name))
    plt.plot(history['age_train_cs'], marker='.', label='train')
    plt.plot(history['age_test_cs'], marker='.', label='test')
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_PATH, 'accuracy_bucket_age_{}.png'.format(model_name)))

    plt.figure(figsize=(16, 16))
    plt.title('MAE age model {}'.format(model_name))
    plt.plot(history['age_train_mae'], marker='.', label='train')
    plt.plot(history['age_test_mae'], marker='.', label='test')
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_PATH, 'MAE_age_{}.png'.format(model_name)))

    plt.figure(figsize=(16, 16))
    plt.title('accuracy gender model {}'.format(model_name))
    plt.plot(history['gender_train'], marker='.', label='train')
    plt.plot(history['gender_test'], marker='.', label='test')
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_PATH, 'accuracy_gender_{}.png'.format(model_name)))

    plt.close('all')


def main(model_name):
    global BATCH_SIZE
    print_log('Using DEVICE {}'.format(DEVICE))
    model = Model().to(DEVICE)

    criterion_age = torch.nn.BCELoss(reduction='sum')
    criterion_gender = torch.nn.CrossEntropyLoss(reduction='sum')

    loader_train = load_data(path=TRAIN_PATH, mode='train')
    loader_test = load_data(path=TEST_PATH, mode='test')
    gc.collect()

    print_log('Data loaded... train {} test {}'.format(len(loader_train.dataset), len(loader_test.dataset)))

    update_bars('init', len1=len(loader_train), len2=len(loader_test))

    epoch = 0
    current_best_loss = np.full(2, fill_value=np.inf)

    history = {
        'loss_train': [],
        'age_train_cs': [],
        'age_train_mae': [],
        'gender_train': [],
        'loss_test': [],
        'age_test_cs': [],
        'age_test_mae': [],
        'gender_test': []
    }

    while epoch < NUM_EPOCHS:
        try:
            model.freeze_backbone(epoch)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )
            update_bars('reset_train')
            update_bars('reset_inference')
            train(loader_train, model, optimizer, [criterion_age, criterion_gender])
            epoch += 1
        except RuntimeError as e:
            if 'CUDA' in str(e):
                print_log('BATCH SIZE {} too big, trying {}'.format(BATCH_SIZE, BATCH_SIZE // 2))
                BATCH_SIZE //= 2
                loader_train = load_data(path=TRAIN_PATH, batch_size=BATCH_SIZE, mode='train')
                loader_test = load_data(path=TEST_PATH, batch_size=BATCH_SIZE, mode='test')
                gc.collect()
                continue
            else:
                raise e

        update_bars('reset_inference')

        inference_train = evaluate(loader_train, model, [criterion_age, criterion_gender])
        inference_test = evaluate(loader_test, model, [criterion_age, criterion_gender])

        history['loss_train'].append(inference_train[0])
        history['loss_test'].append(inference_test[0])

        history['age_train_cs'].append(inference_train[1][0])
        history['age_test_cs'].append(inference_test[1][0])

        history['age_train_mae'].append(inference_train[1][1])
        history['age_test_mae'].append(inference_test[1][1])

        history['gender_train'].append(inference_train[2])
        history['gender_test'].append(inference_test[2])

        if current_best_loss[0] >= inference_train[0] and current_best_loss[1] >= inference_test[0]:
            save(model.state_dict(), os.path.join(MODELS_PATH, '{}_{}.pth'.format(model_name, epoch)))

        if not epoch % 10:
            save(model.state_dict(), os.path.join(MODELS_PATH, '{}.pth'.format(model_name)))

        current_best_loss[0] = min(inference_train[0], current_best_loss[0])
        current_best_loss[1] = min(inference_test[0], current_best_loss[1])

        plot_results(history, model_name)

        update_bars('epochs')


if __name__ == '__main__':
    main(sys.argv[1])
