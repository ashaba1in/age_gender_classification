import json
import multiprocessing
import os
from typing import (
	Any,
	Tuple,
)

import cv2.cv2 as cv2
import dlib
import magic
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader


def get_config():
	with open('config.json', 'r') as _:
		return json.load(_)


FEMALE = 0
MALE = 1

config = get_config()

IMAGE_SIZE = config['image_size']


class Model:
	def __init__(
			self,
			model_name: str,
			device=None,
			num_classes: int = 2,
			load_pretrained: bool = False,
			path_pretrained: str = None
	):
		self.device = device
		self.num_classes = num_classes
		self.path_pretrained = path_pretrained
		self.possible_names = config['model_names']
		
		self._est = None
		
		if model_name in self.possible_names:
			class Identity(nn.Module):
				def __init__(self):
					super(Identity, self).__init__()
				
				@staticmethod
				def forward(x):
					return x
			
			if model_name == 'ResNet-18':
				self._est = torchvision.models.resnet18(pretrained=False)
				self._est.pool0 = Identity()
				self._est.fc = nn.Linear(self._est.fc.in_features, self.num_classes)
			elif model_name == 'ResNet-34':
				self._est = torchvision.models.resnet34(pretrained=False)
				self._est.pool0 = Identity()
				self._est.fc = nn.Linear(self._est.fc.in_features, self.num_classes)
			elif model_name == 'ResNet-50':
				self._est = torchvision.models.resnet50(pretrained=False)
				self._est.pool0 = Identity()
				self._est.fc = nn.Linear(self._est.fc.in_features, self.num_classes)
			elif model_name == 'ResNet-101':
				self._est = torchvision.models.resnet101(pretrained=False)
				self._est.pool0 = Identity()
				self._est.fc = nn.Linear(self._est.fc.in_features, self.num_classes)
			elif model_name == 'ResNet-152':
				self._est = torchvision.models.resnet152(pretrained=False)
				self._est.pool0 = Identity()
				self._est.fc = nn.Linear(self._est.fc.in_features, self.num_classes)
			elif model_name == 'ResNeXt-50-32x4d':
				self._est = torchvision.models.resnext50_32x4d(pretrained=False)
				self._est.pool0 = Identity()
				self._est.fc = nn.Linear(self._est.fc.in_features, self.num_classes)
			elif model_name == 'ResNeXt-101-32x8d':
				self._est = torchvision.models.resnext101_32x8d(pretrained=False)
				self._est.pool0 = Identity()
				self._est.fc = nn.Linear(self._est.fc.in_features, self.num_classes)
			elif model_name == 'WideResNet-50-2':
				self._est = torchvision.models.wide_resnet50_2(pretrained=False)
				self._est.pool0 = Identity()
				self._est.fc = nn.Linear(self._est.fc.in_features, self.num_classes)
			elif model_name == 'WideResNet-101-2':
				self._est = torchvision.models.wide_resnet101_2(pretrained=False)
				self._est.pool0 = Identity()
				self._est.fc = nn.Linear(self._est.fc.in_features, self.num_classes)
			elif model_name == 'Densenet-121':
				self._est = torchvision.models.densenet121(pretrained=False)
				self._est.features.pool0 = Identity()
				self._est.fc = nn.Linear(self._est.classifier.in_features, self.num_classes)
			elif model_name == 'Densenet-169':
				self._est = torchvision.models.densenet169(pretrained=False)
				self._est.features.pool0 = Identity()
				self._est.fc = nn.Linear(self._est.classifier.in_features, self.num_classes)
			elif model_name == 'Densenet-201':
				self._est = torchvision.models.densenet201(pretrained=False)
				self._est.features.pool0 = Identity()
				self._est.fc = nn.Linear(self._est.classifier.in_features, self.num_classes)
			elif model_name == 'Densenet-161':
				self._est = torchvision.models.densenet161(pretrained=False)
				self._est.features.pool0 = Identity()
				self._est.fc = nn.Linear(self._est.classifier.in_features, self.num_classes)
		else:
			print('Model name should be from list {}'.format(self.possible_names))
		
		if load_pretrained:
			self._load_est(os.path.join(path_pretrained, model_name))
	
	def model(self):
		return self._est
	
	def _load_est(self, path):
		model_name = path.split('/')[-1][:-4]
		
		if model_name in self.possible_names:
			try:
				self._est.load_state_dict(torch.load(path))
			except RuntimeError:
				print('Path should lead to correct state dict file')
				return self
			
			if self.device is not None:
				self._est.to(self.device)
			self._est.eval()
			return self
		else:
			print('Path should lead to correct state dict file of model from list {}'.format(self.possible_names))
			return self


class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, dataframe: pd.DataFrame, mode: str = None) -> None:
		super(ImageDataset, self).__init__()
		self.df = dataframe
		self.mode = mode
		self.image_size = IMAGE_SIZE
		
		transforms_list = []
		
		if self.mode == 'train':
			transforms_list.extend([
				transforms.RandomChoice([
					transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
					transforms.RandomAffine(degrees=20, translate=(0.05, 0.05), scale=(0.7, 1.3), shear=10),
					transforms.RandomHorizontalFlip(),
					transforms.RandomGrayscale(0.05)
				])
			])
		
		transforms_list.extend([
			transforms.Resize((self.image_size, self.image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		
		self.transforms = transforms.Compose(transforms_list)
		self.targets = dataframe['target'].values
	
	def __getitem__(self, index: int) -> Any:
		filename = self.df['filename'].values[index]
		sample = Image.open(filename)
		assert sample.mode == 'RGB'
		image = self.transforms(sample)
		if self.mode == 'debug':
			return image, self.df['target'].values[index], filename
		return image, self.df['target'].values[index]
	
	def __len__(self) -> int:
		return self.df.shape[0]


class Dataset(torch.utils.data.Dataset):
	def __init__(self, image_paths: np.array) -> None:
		super(Dataset, self).__init__()
		self.image_paths = image_paths
		self.detected = pd.DataFrame(columns=['face_path', 'parent_image_path', 'age', 'gender'])
		self.detected_path = os.path.join(config['detected_save_path'], 'detected/{}_{}.jpeg')
		
		self.detector = lambda x: dlib.get_frontal_face_detector()(x, 0)  # our detector
		self.is_fake = lambda x: 0  # replace with proper validator
		self.classify_gender = lambda x: MALE  # i want this to get path and return gender
		self.classify_age = lambda x: 0  # i want this to get path and return age group

	def detect_faces(self) -> None:
		for i, image_path in enumerate(self.image_paths):
			image = self.read_image(image_path)
			detected = self.detector(image)
			
			for j, d in enumerate(detected):
				top = max(d.top(), 0)
				bot = min(d.bottom(), IMAGE_SIZE)
				left = max(d.left(), 0)
				right = min(d.right(), IMAGE_SIZE)
				image = image[top:bot, left:right]
				
				path = self.detected_path.format(i, j)
				cv2.imwrite(path, image)
				
				self.detected = self.detected.append(
					{
						'image_path': image_path,
						'face_path': path
					},
					ignore_index=True
				)
	
	def remove_fake_faces(self) -> None:
		drop = []
		
		for i, face_path in zip(self.detected.index, self.detected.face_path):
			image = self.read_image(face_path)
			if self.is_fake(image):
				os.remove(face_path)
				drop.append(i)
		
		self.detected.drop(drop, axis=0, inplace=True)
	
	def classify_faces_gender(self) -> None:
		self.detected.gender = np.vectorize(self.classify_gender)(self.detected.face_path)
	
	def classify_faces_age(self) -> None:
		self.detected.age = np.vectorize(self.classify_age)(self.detected.face_path)
	
	def __getitem__(self, index: int) -> np.array:
		image_path = self.image_paths[index]
		return cv2.resize(cv2.imread(image_path), (IMAGE_SIZE, IMAGE_SIZE))
	
	def __iter__(self):
		yield from self.image_paths
	
	def __len__(self) -> int:
		return len(self.image_paths)
	
	@staticmethod
	def read_image(path, img_size=IMAGE_SIZE):
		return cv2.resize(cv2.imread(path), (img_size, img_size))


class CenterLoss(nn.Module):
	def __init__(self, num_classes=2, feat_dim=2, use_gpu=True):
		super(CenterLoss, self).__init__()
		self.num_classes = num_classes
		self.feat_dim = feat_dim
		self.use_gpu = use_gpu
		
		if self.use_gpu:
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda(), requires_grad=True)
		else:
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim), requires_grad=True)
	
	def forward(self, x, labels):
		batch_size = x.size(0)
		distmatrix = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
		distmatrix += torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
		
		distmatrix.addmm_(x, self.centers.t(), beta=1, alpha=-2)
		
		classes = torch.arange(self.num_classes).long()
		if self.use_gpu:
			classes = classes.cuda()
		
		labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
		mask = labels.eq(classes.expand(batch_size, self.num_classes))
		
		dist = distmatrix * mask.float()
		loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
		
		return loss


def get_filenames(path) -> np.ndarray:
	filenames = []
	for root, _, files in os.walk(path):
		for _file in files:
			filenames.append(os.path.join(root, _file))
	return np.array(filenames)


def get_data(path) -> Tuple[np.ndarray, np.ndarray]:
	filenames = []
	target = []
	for root, _, files in os.walk(path):
		for _file in files:
			if magic.from_file(os.path.join(root, _file), mime=True).split('/')[0] == 'image':
				filenames.append(os.path.join(root, _file))
				target.append(int(filenames[-1].split('_')[1]))
	return np.array(filenames), np.array(target)


def load_data(
		path: str,
		batch_size: int = config['batch_size'],
		num_workers: int = multiprocessing.cpu_count(),
		use_gpu: bool = config['use_gpu'],
		mode: str = 'train'
) -> DataLoader:
	x, y = get_data(path)
	
	df = pd.DataFrame(data={
		'filename': x,
		'target': y
	})
	
	shuffle = mode == 'train'
	
	loader = DataLoader(
		ImageDataset(df, mode=mode),
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=use_gpu
	)
	
	return loader


class FaceHandler:
	def __init__(self, args):
		self.args = args
		self.predictor = dlib.shape_predictor(args.get('predictor_path'))
		self.detector = dlib.cnn_face_detection_model_v1(args.get('detector_path'))
		self.detect_face = args.get('detect_face')
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
		
		rects = dlib.rectangles()
		
		if self.detect_face:
			detected = self.detector(image, 0)
			rects.extend([d.rect for d in detected])
		else:
			rects.append(dlib.rectangle(0, 0, image.shape[0], image.shape[1]))
		
		for i, rect in enumerate(rects):
			x1, y1 = rect.left(), rect.top()
			x2, y2 = rect.right() + 1, rect.bottom() + 1
			
			if x1 < 0:
				x1 = 0
			if y1 < 0:
				y1 = 0
			if x2 < 0:
				x2 = 0
			if y2 < 0:
				y2 = 0
			
			if x2 - x1 > 0 and y2 - y1 > 0:
				self.align(image, rect).save(
					os.path.join(self.save_path, '_'.join(['face', str(int(human)), str(total + i)])) + '.' + ext
				)
		
		return total + len(rects)


def print_log(msg: str, debug: bool = True):
	if debug:
		print(msg)
