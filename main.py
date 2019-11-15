import argparse
import cv2
import joblib
import numpy as np
import magic
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


class Model:
    def __init__(self, estimator):
        self.est = estimator
    
    def predict(self, image):
        return self.est.predict(image)
    
    def probabilities(self, image):
        return self.est.predict_proba(image)[:, 1]
    
    def fit(self, x, y):
        self.est = self.est.fit(x, y)
        return self
    
    def load_est(self, path='model.pkl'):
        self.est = joblib.load(path)
    
    def save_est(self, path='model.pkl'):
        joblib.dump(self.est, path)


def parse(string: str):
    return list(map(int, string[:-1].split('\t')))


def transform_image(image_name):
    return cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB).ravel()


def get_argv():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--train_pixels', type=str, default='False', help='whether to train model or use')
    parser.add_argument('--image_dir', type=str, default='images')
    
    argv = parser.parse_args()
    return argv


def main():
    argv = get_argv()
    if argv.train_pixels == 'True':
        with open('Skin_NonSkin.txt', 'r') as input_file:
            data = np.array([parse(line) for line in tqdm(input_file.readlines(), disable=True)])
        
        x_train, x_test, y_train, y_test = train_test_split(data[:, [0, 1, 2]], data[:, 3], test_size=0.5)
        
        model = Model(GradientBoostingClassifier()).fit(x_train, y_train)
        print('Accuracy for test: {:.4f}'.format(accuracy_score(y_true=y_test, y_pred=model.predict(x_test))))
        print('ROC-AUC for test: {:.4f}'.format(roc_auc_score(y_true=y_test, y_score=model.probabilities(x_test))))
        model.save_est()
    else:
        directory = argv.image_dir
        mime = magic.Magic(mime=True)
        for root, dirs, files in os.walk(directory):
            for _dir in dirs:
                for _file in files:
                    filename = os.path.join(root, _dir,  _file)
                    print(magic.from_file(filename))
            for _file in files:
                filename = os.path.join(root, _file)
                print(mime.from_file(filename))


if __name__ == '__main__':
    main()
