import argparse
import os

import cv2
import dlib
import magic
import numpy as np
from tqdm import tqdm

MAX_IMG_SIZE = 300


def get_argv():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--human', type=int)
    parser.add_argument('--margin', type=float, default=0.5)
    
    argv = parser.parse_args()
    return argv


def resize_image(image: np.ndarray):
    k = image.shape[0] / image.shape[1]
    if image.shape[0] > MAX_IMG_SIZE or image.shape[1] > MAX_IMG_SIZE:
        if k >= 1:
            return cv2.resize(image, (MAX_IMG_SIZE, int(MAX_IMG_SIZE * k)))
        else:
            return cv2.resize(image, (int(MAX_IMG_SIZE * k), MAX_IMG_SIZE))
    return image


def get_filenames(path: str):
    filenames = []
    for root, _, files in os.walk(path):
        for _file in files:
            filenames.append(os.path.join(root, _file))
    return filenames


def central_crop(image: np.ndarray, margin: float):
    assert margin > 0, 'Margin should be greater than 0'
    h, w = image.shape[0], image.shape[1]
    new_h, new_w = int(np.floor(h * margin)), int(np.floor(w * margin))
    return cv2.resize(image, (new_w, new_h))


def save_face(save_path: str, ext: str, human: bool, enum: int, image: np.ndarray):
    cv2.imwrite(os.path.join(save_path, '_'.join(['face', str(int(human)), str(enum)])) + '.' + ext, image)


def main():
    detector = dlib.cnn_face_detection_model_v1("data/dlib_cnn_weight.dat")
    human = get_argv().human
    margin = get_argv().margin
    image_path = get_argv().image_path
    filenames = get_filenames(image_path)
    save_path = get_argv().save_path
    if save_path[-1] != '/':
        save_path += '/'
    total_faces = 0
    for filename in tqdm(filenames, disable=True):
        if magic.from_file(filename, mime=True).split('/')[0] == 'image':
            input_image = resize_image(cv2.imread(filename))
            detected = detector(input_image, 0)
            if len(detected):
                for pos, d in enumerate(detected):
                    x1, y1 = d.rect.left(), d.rect.top()
                    x2, y2 = d.rect.right() + 1, d.rect.bottom() + 1
                    save_face(
                        save_path,
                        magic.from_file(filename, mime=True).split('/')[1],
                        human,
                        total_faces,
                        central_crop(input_image[y1:y2 + 1, x1:x2 + 1, :], margin)
                    )
                    total_faces += 1


if __name__ == '__main__':
    main()
