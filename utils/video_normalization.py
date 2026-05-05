from pathlib import Path
import glob
import pandas as pd
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
import yaml
import random
import time

BASE_PATH = Path(__file__).resolve().parent.parent

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample
    
class NumpyToTensor(torch.nn.Module):
    def forward(self, sample):
        frames = sample["frames"]
        sample["frames"] = torch.from_numpy(frames / 255.0).float()
        return sample
    
class THWC2TCHW(torch.nn.Module):
    def forward(self, sample):
        frames = sample["frames"]
        sample["frames"] = frames.permute(0, 3, 1, 2)
        return sample

class ImageNetNormalize(torch.nn.Module):
    def forward(self, sample):
        frames = sample["frames"]
        sample["frames"] = F.normalize(frames, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return sample

class Resize(torch.nn.Module):
    def __init__(self, resize_dims):
        super().__init__()
        self.size = resize_dims

    def forward(self, sample):
        frames = sample["frames"]
        sample["frames"] = F.resize(frames, self.size)
        return sample
        
class FPSNormalize(torch.nn.Module):
    def __init__(self, target_fps):
        super().__init__()
        self.target_fps = target_fps

    def forward(self, sample):
        frames = sample["frames"]
        ofps = sample["ofps"]

        T = frames.shape[0]
        duration = T / ofps
        target_T = int(duration * self.target_fps)

        indices = torch.linspace(0, T-1, target_T).long()
        sample["frames"] = frames[indices]
        return sample
    
class SpatialAugmentation(torch.nn.Module):
    def __init__(self, rotate, translate, scale):
        super().__init__()
        self.rotate = rotate
        self.translate = translate
        self.scale = scale

    def forward(self, sample):
        # frames: Tensor (T, C, H, W)
        frames = sample["frames"]
        
        # ---- Random Parameters (video 전체에 동일 적용) ----
        angle = random.uniform(-self.rotate, self.rotate) if random.random() < 0.5 else 0
        
        translate_x = random.uniform(-self.translate, self.translate) if random.random() < 0.5 else 0
        translate_y = random.uniform(-self.translate, self.translate) if random.random() < 0.5 else 0
        
        scale = random.uniform(*self.scale) if random.random() < 0.5 else 1.0
        
        sample["frames"] = F.affine(frames, angle=angle, translate=(int(translate_x), int(translate_y)), scale=scale, shear=0)        
        return sample

class ColorJitterAugmentation(torch.nn.Module):
    def __init__(self, brightness=0.1, contrast=0.1):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast

    def forward(self, sample):
        # frames: (T, C, H, W)
        frames = sample["frames"]

        # ---- Random Parameters (video 전체 동일 적용) ----
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness) if self.brightness > 0 else None
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast) if self.contrast > 0 else None

        # ---- Apply frame-wise ----
        frames = F.adjust_brightness(frames, brightness_factor)
        frames = F.adjust_contrast(frames, contrast_factor)

        sample["frames"] = frames
        return sample