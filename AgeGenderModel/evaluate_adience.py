import os

import matplotlib

matplotlib.use('Agg')

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')
import torch
import torch.backends.cudnn as cudnn
from torch.nn.functional import sigmoid
from torch import (
    cat,
    mean,
)
from tqdm import tqdm

from model import Model
from utils import (
    get_config,
    load_adience,
)

config = get_config()

BATCH_SIZE = config['batch_size']
USE_GPU = config['use_gpu']
DEVICE = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')
cudnn.benchmark = True

paths = config['paths']

MODELS_PATH = paths['models_path']
IMAGE_PATH = paths['adience']
GRAPHS_PATH = paths['graphs_path']
model_name = config['model_names'][0]

mapping = {
    0: [0, 2],
    1: [4, 6],
    2: [8, 12],
    3: [15, 20],
    4: [25, 32],
    5: [38, 43],
    6: [48, 53],
    7: [60, 100]
}

milestones = np.array([0, 2, 4, 6, 8, 12, 15, 20, 25, 32, 38, 43, 48, 53, 60, 100])

age_multiplier = torch.from_numpy(np.arange(config['num_classes_age'])).to(DEVICE)


def get_group_by_real_age(age):
    diff = milestones - age
    idx1 = np.searchsorted(diff, 0)
    idx2 = max(idx1 - 1, 0)

    diff = np.abs(diff)

    return idx1 // 2 + np.argmax(diff[[idx1, idx2]]) - 1 + (diff[idx1] == diff[idx2])


def evaluate_adience(data_loader, model):
    model.eval()

    pred_groups = []
    pred_gender = torch.LongTensor()

    with torch.no_grad():
        for images, _, _ in tqdm(data_loader):
            images = images.to(DEVICE)

            output_age, output_gender = model(images)

            output_age = (sigmoid(output_age) > 0.5).float()

            pred_age = age_multiplier * output_age

            pred_age = mean(
                pred_age[output_age > 0],
                dim=1,
                keepdim=True
            ).cpu().numpy()

            for age in pred_age:
                pred_groups.append(get_group_by_real_age(age))

            output_gender = output_gender.data.max(1, keepdim=True)[1].cpu()
            pred_gender = cat((pred_gender, output_gender), dim=0)

    return np.array(pred_groups), pred_gender.numpy().ravel()


def main():
    model = Model(
        load_pretrained=True,
        path_pretrained=os.path.join(MODELS_PATH, '{}.pth'.format(model_name))
    ).to(DEVICE)

    exact = np.zeros(8)
    one_off = np.zeros(8)
    gender = np.zeros(8)

    total_len = np.zeros(8)

    for i in range(5):
        loader = load_adience(path=os.path.join(IMAGE_PATH, 'fold{}'.format(i)), batch_size=BATCH_SIZE, mode='test')

        groups, genders = evaluate_adience(loader, model)
        true_groups, true_genders = loader.dataset.targets_age, loader.dataset.targets_gender
        for j in range(8):
            mask = true_groups == j
            total_len[j] += np.sum(mask)

            exact[j] += np.sum(groups[mask] == true_groups[mask])
            one_off[j] += np.sum(np.abs(groups[mask] - true_groups[mask]) <= 1)
            gender[j] += np.sum(genders[mask] != true_genders[mask])

    exact /= total_len
    one_off /= total_len
    gender /= total_len

    correct_groups = list(mapping.values())

    for i, pe in enumerate(correct_groups):
        correct_groups[i] = '({}, {})'.format(pe[0], pe[1])

    plt.figure(figsize=(12, 12))
    plt.title('Exact age accuracy для всех возрастных групп')
    plt.bar(correct_groups, exact)
    plt.savefig(os.path.join(GRAPHS_PATH, 'exact_{}.png'.format(model_name)))

    plt.figure(figsize=(12, 12))
    plt.title('One-off age accuracy для всех возрастных групп')
    plt.bar(correct_groups, one_off)
    plt.savefig(os.path.join(GRAPHS_PATH, 'one-off_{}.png'.format(model_name)))

    plt.figure(figsize=(12, 12))
    plt.title('Gender accuracy для всех возрастных групп')
    plt.bar(correct_groups, gender)
    plt.savefig(os.path.join(GRAPHS_PATH, 'gender_{}.png'.format(model_name)))

    print('Exact age accuracy: {:.3f} +/- {:.3f}'.format(np.mean(exact), sem(exact)))
    print('One-off age accuracy: {:.3f} +/- {:.3f}'.format(np.mean(one_off), sem(one_off)))
    print('Gender accuracy: {:.3f} +/- {:.3f}'.format(np.mean(gender), sem(gender)))


if __name__ == '__main__':
    main()
