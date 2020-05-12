import multiprocessing
import os
import sys
from glob import glob
from multiprocessing import Pool

import cv2
import dlib
from imutils.face_utils import FaceAligner as fa
from imutils.face_utils import rect_to_bb
from tqdm.autonotebook import tqdm

from utils import get_config

config = get_config()


class FaceAligner:
    def __init__(self):
        self.desiredFaceWidth = 256
        self.face_threshold = -0.4
        self.expand_margin = 0.4

        self.Path2ShapePred = os.path.join(config['paths']['models_path'], 'shape_predictor_68_face_landmarks.dat')

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.Path2ShapePred)
        self.fa = fa(self.predictor, desiredFaceWidth=self.desiredFaceWidth)

    def get(self, img):
        if type(img) == str:
            img = cv2.imread(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        upscale = 1 if img.shape[0] * img.shape[1] < 512 * 512 else 0

        rects, scores, _ = self.detector.run(img, upscale, self.face_threshold)

        exp_rects = []
        for rect in rects:
            x, y, w, h = rect_to_bb(rect)

            x = max(0, x)
            y = max(0, y)
            w = min(img.shape[1] - x, w)
            h = min(img.shape[0] - y, h)

            exp = min(int(w * self.expand_margin), x, img.shape[1] - x - w,
                      int(h * self.expand_margin), y, img.shape[0] - y - h)
            exp = max(0, exp)

            x, y = x - exp, y - exp
            w, h = w + 2 * exp, h + 2 * exp

            exp_rects.append(dlib.rectangle(x, y, x + w, y + h))

        aligned = [self.fa.align(img, gray, rect) for rect in exp_rects]

        return aligned


f = FaceAligner()


def align_one(filename):
    aligned = f.get(filename)
    if len(aligned):
        cv2.imwrite(filename, aligned[0])
    else:
        os.remove(filename)


def main(path):
    print(path)
    filenames = glob(path)
    print(len(filenames))

    pool = Pool(processes=multiprocessing.cpu_count())
    for _ in tqdm(pool.imap_unordered(align_one, filenames), total=len(filenames)):
        pass


if __name__ == '__main__':
    main(sys.argv[1])
