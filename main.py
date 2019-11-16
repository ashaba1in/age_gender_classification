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


class Model:
    def __init__(self, estimator):
        self.est = estimator
    
    def predict(self, image):
        return self.est.predict(image)
    
    def probabilities(self, image):
        return self.est.predict_proba(image)[:, 0]
    
    def fit(self, x, y):
        print((y == 1).sum())
        print((y == 0).sum())
        self.est = self.est.fit(x, y)
        return self
    
    def load_est(self, path='../model.pkl'):
        self.est = joblib.load(path)
        return self
    
    def save_est(self, path='../model.pkl'):
        joblib.dump(self.est, path)
        return self


def parse(string: str):
    return list(map(int, string[:-1].split('\t')))


def transform_image_to_raw(image_name):
    return [cv2.imread(image_name).reshape(-1, 3), cv2.imread(image_name).shape[:2]]


def transform_probabilities_to_img(raw, shape):
    print(test.pred())
    return np.repeat((raw * WHITE).reshape(-1, 1), 3, axis=1).reshape([shape[0], shape[1], 3])


def transform_predictions_to_img(raw, shape):
    mask = raw == 0
    return np.repeat((mask * WHITE + ~mask * BLACK).reshape(-1, 1), 3, axis=1).reshape([shape[0], shape[1], 3])


def get_argv():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--train_pixels', type=int, default=0, help='whether to train model or use')
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--show_images', type=int, default=0)
    parser.add_argument('--count_time', type=int, default=0)
    parser.add_argument('--pred_threshold', type=float, default=0.5)
    
    argv = parser.parse_args()
    return argv


def get_filenames(path):
    filenames = []
    for root, _, files in os.walk(path):
        for _file in files:
            filenames.append(os.path.join(root, _file))
    return filenames


def main():
    argv = get_argv()
    if argv.train_pixels:
        with open('Skin_NonSkin.txt', 'r') as input_file:
            data = np.array([parse(line) for line in tqdm(input_file.readlines(), disable=True)])
        
        x_train, x_test, y_train, y_test = train_test_split(data[:, [0, 1, 2]], data[:, 3] - 1, test_size=0.5)
        
        model = Model(GradientBoostingClassifier()).fit(x_train, y_train)
        print('Accuracy for test: {:.4f}'.format(accuracy_score(y_true=y_test, y_pred=model.predict(x_test))))
        print('ROC-AUC for test: {:.4f}'.format(roc_auc_score(y_true=y_test, y_score=model.probabilities(x_test))))
        model.save_est()
    else:
        global_start = 0
        if argv.count_time:
            global_start = time.perf_counter()
        total_true = 0
        total = 0
        prediction_threshold = argv.pred_threshold
        filenames = get_filenames(argv.image_dir)
        model = Model(GradientBoostingClassifier()).load_est()
        for filename in tqdm(filenames, disable=True):
            if magic.from_file(filename, mime=True).split('/')[0] == 'image':
                total += 1
                if argv.count_time and total % 1000 == 0:
                    print('total pictures {} with {:.5f}% correct'.format(total, total_true / total * 100))
                    print('total time: {:.5f} seconds'.format(time.perf_counter() - global_start))
                raw, shape = transform_image_to_raw(filename)
                predictions = model.predict(raw)
                if predictions.mean() > prediction_threshold:
                    total_true += 1
                if argv.show_images:
                    probabilities = model.probabilities(raw)
                    probabilities_image = transform_probabilities_to_img(probabilities, shape)
                    predictions_image = transform_predictions_to_img(predictions, shape)
                    cv2.imshow('original', cv2.imread(filename))
                    cv2.imshow('probabilities', probabilities_image)
                    cv2.imshow('predictions', predictions_image)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
        print('total pictures {} with {:.5f}% correct'.format(total, total_true / total * 100))
        if argv.count_time:
            print('total time: {:.5f} seconds'.format(time.perf_counter() - global_start))


if __name__ == '__main__':
    main()
