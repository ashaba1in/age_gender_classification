from typing import Any

import cv2.cv2 as cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage.feature import local_binary_pattern


class BaseModel:
    def __init__(self):
        self.est = None
    
    def check_model_implemented(self) -> None:
        if self.est is None:
            raise NotImplementedError('Model is not defined')
    
    def _predict(self, data) -> Any:
        return self.est.predict(data)
    
    def transform_data(self, data: np.ndarray) -> np.ndarray:
        pass
    
    def load_est(self, path):
        with open(path, 'rb') as file:
            self.est = joblib.load(file)
        return self
    
    def dump_est(self, path) -> None:
        with open(path, 'wb') as file:
            joblib.dump(self.est, file)


class ModelMem(BaseModel):
    def __init__(self, image_size: int = 64):
        super().__init__()
        transforms_list = []
        transforms_list.extend([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.transforms = transforms.Compose(transforms_list)
    
    def transform_data(self, image: np.ndarray) -> np.ndarray:
        return self.transforms(Image.fromarray(image))
    
    def _predict(self, data):
        pred = self.est(data[None, ...])
        return pred.cpu().data.max(1, keepdim=True)[1]
    
    def predict(self, image: np.ndarray):
        self.check_model_implemented()
        return self._predict(self.transform_data(image))
    
    def load_est(self, path):
        self.est = torchvision.models.resnet34()
        self.est.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.est.fc = nn.Linear(self.est.fc.in_features, 2)
        self.est.load_state_dict(torch.load(path))
        self.est.eval()
        return self


class ModelAds(BaseModel):
    def __init__(self, p: int = 8, image_size: int = 16):
        super().__init__()
        assert image_size > 0, 'Image size should be greater than 0'
        self.p = p
        self.image_size = image_size
    
    def get_lbp_hist_res(self, image: np.ndarray, r: int = 2) -> np.ndarray:
        features = local_binary_pattern(image, self.p * r, r, method="uniform").ravel()
        bins = self.p * r + 2  # because of uniform method
        counts, _ = np.histogram(features, bins=bins)
        return counts
    
    def transform_data(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        YCbCr = []
        HSV = []
        for i in range(3):
            YCbCr += list(self.get_lbp_hist_res(img_ycbcr[:, :, i]))
            HSV += list(self.get_lbp_hist_res(img_hsv[:, :, i]))

        return np.array(YCbCr + HSV).reshape(1, -1)
    
    def predict(self, image: np.ndarray):
        self.check_model_implemented()
        return self._predict(self.transform_data(image))


class LivenessChecker:
    def __init__(self, path1: str = 'model1.pth', path2: str = 'model2.pkl'):
        self.model_mem = ModelMem().load_est(path1)
        self.model_ads = ModelAds().load_est(path2)
    
    def check_liveness(self, image: np.ndarray) -> bool:
        if self.model_mem.predict(image):
            if self.model_ads.predict(image):
                return True
            return False
        else:
            return False
