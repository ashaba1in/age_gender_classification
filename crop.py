import argparse
import os

import cv2.cv2 as cv2
import dlib
import magic
import numpy as np
from PIL import Image
from tqdm import tqdm


class FaceHandler:
    def __init__(self, args):
        self.args = args
        self.predictor = dlib.shape_predictor(args.get('predictor_path'))
        self.detector = dlib.cnn_face_detection_model_v1(args.get('detector_path'))
        self.save_path = args.get('save_path')
        self.left_eye = args.get('left_eye')
        self.face_width = args.get('face_width')
        self.face_height = args.get('face_height')
        self.eyes_slice = 5
        
        if self.face_width is None:
            self.face_width = 256
        if self.face_height is None:
            self.face_height = self.face_width
        if self.left_eye is None:
            self.left_eye = 0.35, 0.35
    
    def eyes_pos(self, shape):
        coords = np.empty((shape.num_parts, 2), dtype=int)
        
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        
        return coords[self.eyes_slice + 1:], coords[:self.eyes_slice]
    
    def align(self, image, rect):
        shape = self.predictor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), rect)
        leftEyePts, rightEyePts = self.eyes_pos(shape)
        
        leftEyeCenter = leftEyePts.mean(axis=0)
        rightEyeCenter = rightEyePts.mean(axis=0)
        
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        desiredRightEyeX = 1.0 - self.left_eye[0]
        
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.left_eye[0]) * self.face_width
        scale = desiredDist / dist
        
        eyesCenter = (leftEyeCenter[0] + rightEyeCenter[0]) // 2, (leftEyeCenter[1] + rightEyeCenter[1]) // 2
        
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        
        tX = self.face_width * 0.5
        tY = self.face_height * self.left_eye[1]
        
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        size = self.face_width, self.face_height
        
        return Image.fromarray(cv2.warpAffine(image, M, size, flags=cv2.INTER_CUBIC), 'RGB')
    
    def process(self, image, args):
        human = args.get('human')
        total = args.get('total')
        ext = args.get('ext')
        rects = self.detector(image, 0)
        for i, rect in enumerate(rects):
            x1, y1 = rect.rect.left(), rect.rect.top()
            x2, y2 = rect.rect.right() + 1, rect.rect.bottom() + 1
            
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 < 0:
                x2 = 0
            if y2 < 0:
                y2 = 0
            
            if x2 - x1 > 0 and y2 - y1 > 0:
                self.align(image, rect.rect).save(
                    os.path.join(self.save_path, '_'.join(['face', str(int(human)), str(total + i)])) + '.' + ext
                )
        return total + len(rects)


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
    parser.add_argument('--detect_face', type=bool, default=True)
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
        old_size = im.size
        
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        
        im = im.resize(new_size, Image.ANTIALIAS)
        
        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))
        
        image = np.asarray(new_im)
        
        if image.shape[0] != 0 and image.shape[1] != 0:
            image = cv2.resize(image, (600, 600))
        else:
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
