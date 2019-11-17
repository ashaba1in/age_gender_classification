import argparse
import os

import cv2
import dlib
import magic
import numpy as np
from tqdm import tqdm


def get_argv():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--human', type=int)
    
    argv = parser.parse_args()
    return argv


def get_filenames(path: str):
    filenames = []
    for root, _, files in os.walk(path):
        for _file in files:
            filenames.append(os.path.join(root, _file))
    return sorted(filenames, key=lambda x: int(x.split('/')[-1]))


def save_face(save_path: str, ext: str, human: bool, enum: int, image: np.ndarray):
    cv2.imwrite(os.path.join(save_path, '_'.join(['face', str(int(human)), str(enum)])) + '.' + ext, image)


def main():
    detector = dlib.cnn_face_detection_model_v1("data/dlib_cnn_weight.dat")
    human = get_argv().human
    image_path = get_argv().image_path
    filenames = get_filenames(image_path)
    total_faces = 0
    for filename in tqdm(filenames, disable=True):
        print(filename)
        if magic.from_file(filename, mime=True).split('/')[0] == 'image':
            input_image = cv2.resize(cv2.imread(filename), (800, 800))
            detected = detector(input_image, 0)
            if len(detected):
                for pos, d in enumerate(detected):
                    x1, y1 = d.rect.left(), d.rect.top()
                    x2, y2 = d.rect.right() + 1, d.rect.bottom() + 1
                    save_face(os.path.splitext(filename)[0],
                              magic.from_file(filename, mime=True).split('/')[1],
                              human,
                              total_faces,
                              input_image[y1:y2 + 1, x1:x2 + 1, :])
                    total_faces += 1


if __name__ == '__main__':
    main()
