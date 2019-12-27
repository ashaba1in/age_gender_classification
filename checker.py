from typing import Any

import cv2.cv2 as cv2
import joblib
import numpy as np
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
    def __init__(self, margin: float = 0.5, threshold: float = 0.05):
        super().__init__()
        assert margin > 0, 'Margin should be greater than 0'
        assert threshold > 0, 'Threshold should be greater than 0'
        self.crop_margin = margin
        self.threshold = threshold
    
    def transform_data(self, image: np.ndarray) -> np.ndarray:
        MAX_IMG_SIZE = 200
        h, w = image.shape[0], image.shape[1]
        new_h, new_w = int(np.floor(h * self.crop_margin)), int(np.floor(w * self.crop_margin))
        image = cv2.resize(image, (new_w, new_h))
        k = h / w
        if h > MAX_IMG_SIZE or w > MAX_IMG_SIZE:
            if k >= 1:
                image = cv2.resize(image, (MAX_IMG_SIZE, int(MAX_IMG_SIZE * k)))
            else:
                image = cv2.resize(image, (int(MAX_IMG_SIZE * k), MAX_IMG_SIZE))
        return image.reshape(-1, 3)
    
    def predict(self, image: np.ndarray):
        self.check_model_implemented()
        return self._predict(self.transform_data(image)).mean() >= self.threshold


class ModelAds(BaseModel):
    def __init__(self, p: int = 8, image_size: int = 16):
        super().__init__()
        assert image_size > 0, 'Image size should be greater than 0'
        self.p = p
        self.image_size = image_size
    
    def get_lbp_hist_res(self, image: np.ndarray, r: int = 2) -> np.ndarray:
        features = local_binary_pattern(image, self.p * r, r, method="uniform").ravel()
        bins = int(features.max()) + 1
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
        
        return np.array(YCbCr + HSV)
    
    def predict(self, image: np.ndarray):
        self.check_model_implemented()
        return self._predict(self.transform_data(image))


class LivenessChecker:
    def __init__(self, threshold: float = 0.05, path1: str = 'model1.pkl', path2: str = 'model2.pkl'):
        self.model_mem = ModelMem(threshold).load_est(path1)
        self.model_ads = ModelAds().load_est(path2)
    
    def check_liveness(self, image: np.ndarray) -> bool:
        if self.model_mem.predict(image):
            if self.model_ads.predict(image):
                return True
            return False
        else:
            return False
