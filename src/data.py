import os
import json
from glob import glob
from PIL import Image, ImageEnhance, ImageFilter
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.utils import resize_and_center_crop, apply_clahe_rgb, sharpen_pil


class TrafficSignDataset(Dataset):
    def __init__(self, root_dir, classes=None, image_size=224, train=True):
        self.samples = []
        self.classes = classes or self._discover_classes(root_dir)
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        self.image_size = image_size
        self.train = train
        for cls in self.classes:
            pattern = os.path.join(root_dir, cls, "*.ppm")
            files = glob(pattern)
            if not files:
                # try common image extensions
                files = glob(os.path.join(root_dir, cls, "*.png")) + glob(os.path.join(root_dir, cls, "*.jpg"))
            for f in files:
                self.samples.append((f, self.class_to_idx[cls]))

        self.transform_train = A.Compose([
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=25, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.2),
            ], p=0.2),
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])

        self.transform_test = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])

    def _discover_classes(self, root_dir):
        entries = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        entries.sort()
        return entries

    def __len__(self):
        return len(self.samples)

    def _preprocess_image(self, path):
        img = Image.open(path).convert('RGB')
        # resize + center crop
        img = resize_and_center_crop(img, self.image_size)
        # handle low-light: apply CLAHE via OpenCV on numpy
        npimg = np.array(img)[:,:,::-1].copy() # RGB->BGR
        npimg = apply_clahe_rgb(npimg)
        img = Image.fromarray(npimg[:,:,::-1])
        # sharpen a bit to help blurred images
        img = sharpen_pil(img)
        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self._preprocess_image(path)
        img = np.array(img)
        if self.train:
            data = self.transform_train(image=img)
        else:
            data = self.transform_test(image=img)
        return data['image'], label


def get_dataloaders(train_dir, test_dir, batch_size=32, image_size=224, num_workers=4):
    train_ds = TrafficSignDataset(train_dir, image_size=image_size, train=True)
    test_ds = TrafficSignDataset(test_dir, classes=train_ds.classes, image_size=image_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, train_ds.classes
