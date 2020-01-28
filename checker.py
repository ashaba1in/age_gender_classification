import torch
import torchvision


class ModelMem:
    def __init__(self, device=None):
        self.est = None
        self.device = device
    
    def _predict(self, data):
        return self.est(data).data.max(1, keepdim=True)[1].cpu().numpy().ravel()
    
    def predict(self, image):
        return self._predict(image)
    
    def load_est(self, path):
        NUM_CLASSES = 2
        self.est = torchvision.models.resnext50_32x4d(pretrained=False, num_classes=NUM_CLASSES)
        self.est.load_state_dict(torch.load(path))
        if self.device is not None:
            self.est.to(self.device)
        self.est.eval()
        return self


class LivenessChecker:
    def __init__(self, path: str = 'model.pth', device=None):
        self.model = ModelMem(device).load_est(path)
    
    def check_liveness(self, image):
        return self.model.predict(image)
