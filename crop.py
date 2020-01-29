import argparse
import os

import magic
import numpy as np
from PIL import Image
from tqdm import tqdm

from classes import FaceHandler


def get_filenames(path):
    filenames = []
    for root, _, files in os.walk(path):
        for _file in files:
            filenames.append(os.path.join(root, _file))
    return filenames


def get_argv():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--predictor_path', type=str, default='data/eye.dat')
    parser.add_argument('--detector_path', type=str, default='data/face.dat')
    parser.add_argument('--detect_face', type=int)
    parser.add_argument('--face_width', type=int)
    parser.add_argument('--human', type=int)
    
    return parser.parse_args()


def construct_dict(argv):
    args = {}
    for arg in vars(argv):
        args[arg] = getattr(argv, arg)
    return args


def main():
    argv = get_argv()
    args = construct_dict(argv)
    fh = FaceHandler(args)
    desired_size = 1000
    
    human, image_path, save_path = argv.human, argv.image_path, argv.save_path
    
    filenames = get_filenames(image_path)
    print('Total files found {}'.format(len(filenames)))
    
    total = 0
    
    for filename in tqdm(filenames):
        if magic.from_file(filename, mime=True).split('/')[0] != 'image':
            pass

        im = Image.open(filename)

        ratio = float(desired_size) / max(im.size)
        new_size = tuple([int(x * ratio) for x in im.size])

        im = im.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new('RGB', (desired_size, desired_size))
        new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))

        image = np.asarray(new_im)

        if image.shape[0] == 0 or image.shape[1] == 0:
            pass

        tmp_args = {
            'human': args.get('human'),
            'total': total,
            'ext': magic.from_file(filename, mime=True).split('/')[1]
        }

        total = fh.process(image, tmp_args)
    
    print('Total faces {}'.format(total))


if __name__ == '__main__':
    main()
