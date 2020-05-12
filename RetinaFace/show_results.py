import argparse
import torch
import torch.nn.functional as F
import numpy as np
from FaceDetectionModel import eval_widerface
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from torchvision_model import load_model

IMG_EXTENSIONS = ['jpeg', 'jpg', 'png']


def get_args():
    parser = argparse.ArgumentParser(description="Detect program for retinaface.")
    parser.add_argument('--data_path', type=str,
                        help='Path for image to detect')
    parser.add_argument('--model_path', type=str, help='Path for model')
    args = parser.parse_args()
    print(args)

    return args


def show_result(image, picked_boxes, picked_landmarks, picked_scores):
    fig, ax = plt.subplots(1, figsize=(15, 10))
    ax.imshow(image)

    # nose, left eye, right eye, left mouth, right mouth
    colors = ['b', 'g', 'r', 'c', 'y']
    for j, boxes in enumerate(picked_boxes):
        if boxes is not None:
            for box, landmark, score in zip(boxes, picked_landmarks[j], picked_scores[j]):
                rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                                 facecolor='none')
                ax.add_patch(rect)
                for i, color in enumerate(colors):
                    circle = Circle((landmark[2 * i], landmark[2 * i + 1]), radius=2, color=colors[i])
                    ax.add_patch(circle)

                plt.text(box[0], box[1], str(score.item())[:5])

    plt.show()


def yield_image(imgs_path) -> np.array:
    for path, folders, files in os.walk(imgs_path):
        for file in files:
            if file.split('.')[-1] in IMG_EXTENSIONS:
                img_path = os.path.join(path, file)
                img = np.array(Image.open(img_path))
                yield img


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def main():
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    for image in yield_image(args.data_path):
        input_img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        if device is not None:
            input_img = input_img.to(device)

        picked_boxes, picked_landmarks, picked_scores = eval_widerface.get_detections(input_img, model,
                                                                                      score_threshold=0.98,
                                                                                      iou_threshold=0.2)

        show_result(image, picked_boxes, picked_landmarks, picked_scores)


if __name__ == '__main__':
    main()
