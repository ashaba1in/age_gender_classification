import argparse
import os

import cv2.cv2 as cv2
import dlib
import magic
import numpy as np
from tqdm import tqdm


def get_argv():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--human', type=int)
    
    return parser.parse_args()


def get_filenames(path: str) -> list:
    filenames = []
    for root, _, files in os.walk(path):
        for _file in files:
            filenames.append(os.path.join(root, _file))
    return filenames


def save_face(save_path: str, ext: str, human: bool, enum: int, image: np.ndarray) -> None:
    cv2.imwrite(os.path.join(save_path, '_'.join(['face', str(int(human)), str(enum)])) + '.' + ext, image)


def main():
    detector = dlib.cnn_face_detection_model_v1('data/dlib_cnn_weight.dat')
    human = get_argv().human
    image_path = get_argv().image_path
    filenames = get_filenames(image_path)
    save_path = get_argv().save_path
    if save_path[-1] != '/':
        save_path += '/'
    total_faces = 0
    for filename in tqdm(filenames):
        if magic.from_file(filename, mime=True).split('/')[0] == 'image':
            image = cv2.imread(filename)
            detected = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0)
            for pos, d in enumerate(detected):
                x1, y1 = d.rect.left(), d.rect.top()
                x2, y2 = d.rect.right() + 1, d.rect.bottom() + 1
                if x2 - x1 > 0 and y2 - y1 > 0:
                    save_face(
                        save_path,
                        magic.from_file(filename, mime=True).split('/')[1],
                        human,
                        total_faces,
                        image[y1:y2 + 1, x1:x2 + 1, :]
                    )
                    total_faces += 1
    
    print('Найдено {} лиц'.format(total_faces))


if __name__ == '__main__':
    main()
