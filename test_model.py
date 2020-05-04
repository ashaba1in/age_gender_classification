import gc
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.multiprocessing

from utils import (
    AgeGender,
    get_config,
    load_data,
    print_log,
)

config = get_config()

IMAGE_PATH = config['image_test_path']
NUM_CLASSES_AGE = config['num_classes_age']
NUM_CLASSES_GENDER = config['num_classes_gender']
NUM_WORKERS = multiprocessing.cpu_count()

USE_GPU = config['use_gpu']

DEVICE = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
cudnn.benchmark = True

MODELS_PATH = config['models_path']


def test(data_loader, model):
    model.eval()
    correct_age = np.zeros(NUM_CLASSES_AGE, dtype=int)
    correct_gender = np.zeros(NUM_CLASSES_GENDER, dtype=int)

    with torch.no_grad():
        for images, labels_age, labels_gender in data_loader:
            images, labels_age, labels_gender = images.to(DEVICE), labels_age.to(DEVICE), labels_gender.to(DEVICE)

            output_age, output_gender = model(images)

            pred_age = output_age.data.max(1, keepdim=True)[1]
            for i in range(NUM_CLASSES_AGE):
                mask = labels_age == i
                correct_age[i] += pred_age[mask].eq(
                    labels_age[mask].data.view_as(pred_age[mask])).cpu().sum()

            pred_gender = output_age.data.max(1, keepdim=True)[1]
            for i in range(NUM_CLASSES_GENDER):
                mask = labels_gender == i
                correct_gender[i] += pred_gender[mask].eq(
                    labels_gender[mask].data.view_as(pred_gender[mask])).cpu().sum()

    counts_age = np.zeros(NUM_CLASSES_AGE)
    for i in range(NUM_CLASSES_AGE):
        counts_age[i] += np.sum(data_loader.dataset.targets_age == i)

    counts_gender = np.zeros(NUM_CLASSES_GENDER)
    for i in range(NUM_CLASSES_GENDER):
        counts_gender[i] += np.sum(data_loader.dataset.targets_gender == i)

    classes = [correct_age / counts_age, correct_gender / counts_gender]
    accuracy_age = np.sum(correct_age) / np.sum(counts_age)
    accuracy_gender = np.sum(correct_gender) / np.sum(counts_gender)

    return classes, accuracy_age, accuracy_gender


def main(model_name):
    with open(os.path.join(MODELS_PATH, 'BATCH_{}.txt'.format(model_name)), 'r') as _:
        BATCH_SIZE = int(_.read())

    model = AgeGender(
        model_name=model_name,
        device=DEVICE,
        load_pretrained=True,
        path_pretrained=MODELS_PATH
    ).model()

    global_start = time.perf_counter()

    loader = load_data(path=IMAGE_PATH, batch_size=BATCH_SIZE)
    gc.collect()

    classes, accuracy_age, accuracy_gender = np.array(test(loader, model))

    for i in range(NUM_CLASSES_AGE):
        print_log('AGE CLASS {} correct {:.5%}'.format(i, classes[0][i]))

    for i in range(NUM_CLASSES_GENDER):
        print_log('GENDER CLASS {} correct {:.5%}'.format(i, classes[1][i]))

    print_log('total time: {:.5f} seconds'.format(time.perf_counter() - global_start))

    dct = {
        'total_time': time.perf_counter() - global_start,
        'accuracy_AGE': accuracy_age,
        'accuracy_GENDER': accuracy_gender
    }

    pd.DataFrame(data=dct, index=[model_name]).to_csv(os.path.join(MODELS_PATH, 'results.csv'), mode='a', header=False)


if __name__ == '__main__':
    main(sys.argv[1])
