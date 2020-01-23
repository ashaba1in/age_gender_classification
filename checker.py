from typing import Any

import cv2.cv2 as cv2
import joblib
import numpy as np
import sklearn.svm as svm
import torch
import torchvision
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
    def __init__(self, device=None):
        super().__init__()
        self.device = device
    
    def _predict(self, data):
        return self.est(data).data.max(1, keepdim=True)[1].cpu().numpy().ravel()
    
    def predict(self, image):
        self.check_model_implemented()
        return self._predict(image)
    
    def load_est(self, path):
        NUM_CLASSES = 2
        self.est = torchvision.models.resnext50_32x4d(pretrained=False, num_classes=NUM_CLASSES)
        self.est.load_state_dict(torch.load(path))
        if self.device is not None:
            self.est.to(self.device)
        self.est.eval()
        return self


class ModelAds(BaseModel):
    def __init__(self):
        super().__init__()
        self.P = 8
        self.R = 2
        self.est = None
    
    def check_model_implemented(self):
        if self.est is None:
            raise NotImplementedError("Model is not defined")
    
    def get_lbp_hist_res(self, image):
        features = local_binary_pattern(image, self.P * self.R, self.R, method="uniform").ravel()
        features_amount = self.P * self.R + 1  # because of uniform method
        bins = features_amount + 1
        counts, _ = np.histogram(features, bins=bins)
        return counts
    
    def transform_image(self, im):
        im = cv2.resize(im, (48, 48), interpolation=cv2.INTER_AREA)
        img_ycbcr = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
        img_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        
        YCbCr = []
        HSV = []
        for i in range(3):
            YCbCr += list(self.get_lbp_hist_res(img_ycbcr[:, :, i]))
            HSV += list(self.get_lbp_hist_res(img_hsv[:, :, i]))
        
        return np.array(YCbCr + HSV).reshape(1, -1)
    
    def transform_images(self, images):
        transformed_x = []
        for img in images:
            transformed_x.append(self.transform_image(img))
        return np.array(transformed_x)
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        transformed_x = self.transform_images(x)
        self.est = svm.SVC(kernel="poly", gamma="scale", degree=7, coef0=0).fit(transformed_x, y)
        return self
    
    def predict(self, x: np.ndarray):
        self.check_model_implemented()
        return self.est.predict(self.transform_image(x))


class LivenessChecker:
    def __init__(self, path1: str = 'model1.pth', path2: str = 'model2.pkl', debug: bool = False, device=None):
        self.model_mem = ModelMem(device).load_est(path1)
        self.model_ads = ModelAds().load_est(path2)
        self.debug = debug
    
    def check_liveness(self, image):
        if self.debug:
            return self.model_mem.predict(image), None
        if self.model_mem.predict(image):
            if self.model_ads.predict(image):
                return True, None
            return False, 'ad'
        else:
            return False, 'mem'
