import argparse
import os
import time

import cv2
import joblib
import magic
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

WHITE = 255.
BLACK = 0.
MAX_IMG_SIZE = 300


class Model:
    def __init__(self, estimator):
        self.est = estimator

    def predict(self, image: np.ndarray):
        return self.est.predict(image)

    def probabilities(self, image: np.ndarray):
        return self.est.predict_proba(image)[:, 1]

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.est = self.est.fit(x, y)
        return self

    def load_est(self, path: str = '../model.pkl'):
        self.est = joblib.load(path)
        return self

    def save_est(self, path: str = '../model.pkl'):
        joblib.dump(self.est, path)
        return self


def parse(string: str):
    return list(map(int, string[:-1].split('\t')))


def resize_image(image: np.ndarray):
    k = image.shape[0] / image.shape[1]
    if image.shape[0] > MAX_IMG_SIZE or image.shape[1] > MAX_IMG_SIZE:
        if k >= 1:
            return cv2.resize(image, (MAX_IMG_SIZE, int(MAX_IMG_SIZE * k)))
        else:
            return cv2.resize(image, (int(MAX_IMG_SIZE * k), MAX_IMG_SIZE))
    return image


def transform_image_to_raw(image_name: str):
    return [resize_image(cv2.imread(image_name)).reshape(-1, 3), cv2.imread(image_name).shape[:2]]


def transform_probabilities_to_img(raw: np.ndarray, shape: tuple):
    return np.repeat((raw * WHITE).reshape(-1, 1), 3, axis=1).reshape([shape[0], shape[1], 3])


def transform_predictions_to_img(raw: np.ndarray, shape: tuple):
    mask = raw == 0
    return np.repeat((mask * WHITE + ~mask * BLACK).reshape(-1, 1), 3, axis=1).reshape([shape[0], shape[1], 3])


def get_argv():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train_pixels', type=bool, default=False)
    parser.add_argument('--image_path', type=str, default='images')
    parser.add_argument('--show_images', type=bool, default=False)
    parser.add_argument('--count_time', type=bool, default=True)
    parser.add_argument('--pred_threshold', type=float)
    parser.add_argument('--count_stats_pred', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)
    
    argv = parser.parse_args()
    return argv


def get_filenames(path: str):
    filenames = []
    for root, _, files in os.walk(path):
        for _file in files:
            filenames.append(os.path.join(root, _file))
    return filenames


def main():
    argv = get_argv()
    if argv.train_pixels:
        with open('../Skin_NonSkin.txt', 'r') as input_file:
            data = np.array([parse(line) for line in tqdm(input_file.readlines(), disable=True)])

        x_train, x_test, y_train, y_test = train_test_split(data[:, [0, 1, 2]], 2 - data[:, 3], test_size=0.5)
        
        model = Model(GradientBoostingClassifier()).fit(x_train, y_train)
        print('Accuracy for test: {:.4f}'.format(accuracy_score(y_true=y_test, y_pred=model.predict(x_test))))
        print('ROC-AUC for test: {:.4f}'.format(roc_auc_score(y_true=y_test, y_score=model.probabilities(x_test))))
        model.save_est()
    elif argv.test:
        global_start = 0
        if argv.count_time:
            global_start = time.perf_counter()
        mistakes_human1 = 0
        mistakes_human0 = 0
        total_human0 = 0
        total_human1 = 0
        prediction_threshold = argv.pred_threshold
        filenames = get_filenames(argv.image_path)
        model = Model(GradientBoostingClassifier()).load_est()
        for filename in tqdm(filenames, disable=True):
            if magic.from_file(filename, mime=True).split('/')[0] == 'image':
                human = int(filename.split('_')[1])
                total_human0 += 1 - human
                total_human1 += human
                if argv.count_time and (total_human0 + total_human1) % 1000 == 0:
                    print('total pictures {} with {:.5%} correct'.format(
                        total_human0 + total_human1,
                        1 - (mistakes_human0 + mistakes_human1) / (total_human0 + total_human1)))
                    print('human correct {:.5%}'.format(1 - mistakes_human1 / total_human1))
                    print('not human correct {:.5%}'.format(1 - mistakes_human0 / total_human0))
                    print('total time: {:.5f} seconds'.format(time.perf_counter() - global_start))
                raw, shape = transform_image_to_raw(filename)
                predictions = model.predict(raw)

                if (human and predictions.mean() < prediction_threshold) or (
                        not human and predictions.mean() > prediction_threshold):
                    print(filename)
                    if human:
                        mistakes_human1 += 1
                    else:
                        mistakes_human0 += 1
                
                if argv.show_images:
                    probabilities = model.probabilities(raw)

                    probabilities_image = transform_probabilities_to_img(probabilities, shape)
                    predictions_image = transform_predictions_to_img(predictions, shape)

                    cv2.imshow('original', cv2.imread(filename))
                    cv2.imshow('probabilities', probabilities_image)
                    cv2.imshow('predictions', predictions_image)

                    cv2.waitKey()
                    cv2.destroyAllWindows()

        print('total pictures {} with {:.5%} correct'.format(
            total_human0 + total_human1,
            1 - (mistakes_human0 + mistakes_human1) / (total_human0 + total_human1)))
        print('human correct {:.5%}'.format(1 - mistakes_human1 / total_human1))
        print('not human correct {:.5%}'.format(1 - mistakes_human0 / total_human0))
        if argv.count_time:
            print('total time: {:.5f} seconds'.format(time.perf_counter() - global_start))
    elif argv.count_stats_pred:
        global_start = 0
        if argv.count_time:
            global_start = time.perf_counter()
        predictions = []
        filenames = get_filenames(argv.image_path)
        model = Model(GradientBoostingClassifier()).load_est()
        for filename in tqdm(filenames, disable=True):
            if magic.from_file(filename, mime=True).split('/')[0] == 'image':
                raw, shape = transform_image_to_raw(filename)
                predictions.append(model.predict(raw).mean())
                if argv.count_time and len(predictions) % 1000 == 0:
                    print('Mean prediction on {} images: {:.4f}, variance: {:.4f}'.format(len(predictions),
                                                                                          np.mean(predictions),
                                                                                          np.var(predictions, ddof=1)))
                    print('total time: {:.5f} seconds'.format(time.perf_counter() - global_start))

        print('Mean prediction on {} images: {:.4f}, variance: {:.4f}'.format(len(predictions), np.mean(predictions),
                                                                              np.var(predictions, ddof=1)))
        print('99% quantile: {:4f}'.format(np.percentile(np.array(predictions), 1)))
        if argv.count_time:
            print('total time: {:.5f} seconds'.format(time.perf_counter() - global_start))


if __name__ == '__main__':
    main()
