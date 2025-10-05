"""
Data loading and preprocessing utilities for TSR project.

This module provides PyTorch Dataset classes and helper functions to:
- Load images from `dataset/train` and `dataset/test` organized by class folders
- Apply histogram equalization, deblurring (simple), resizing to 224x224
- Augmentations: rotation, brightness/contrast

Comments explain each step so it's easy to follow.
"""
import os
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Ensure image is RGB."""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def equalize_histogram_cv(img: np.ndarray) -> np.ndarray:
    """Apply histogram equalization per channel using OpenCV.

    Expects image in RGB uint8, returns RGB uint8.
    """
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    channels = cv2.split(img)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    return cv2.merge(eq_channels)


def simple_deblur(img: np.ndarray) -> np.ndarray:
    """Apply a mild sharpening kernel to reduce blur effect."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


class TrafficSignDataset(Dataset):
    """PyTorch Dataset for traffic sign images stored in class subfolders.

    Expects structure:
      root/class_x/*.png
      root/class_y/*.png

    Returns images resized to `image_size` and a label index.
    """

    def __init__(self, root, classes=None, image_size=224, augment=False):
        self.root = root
        self.image_size = image_size
        self.augment = augment

        # Build class list from folders if not provided
        if classes is None:
            self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        else:
            self.classes = classes

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for c in self.classes:
            folder = os.path.join(root, c)
            for fname in os.listdir(folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(folder, fname), self.class_to_idx[c]))

        # Albumentations pipeline for robustness and augmentation
        base_transforms = [
            A.Resize(self.image_size, self.image_size),
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                A.RandomBrightnessContrast(0.2, 0.2),
            ], p=0.7),
            A.Rotate(limit=15, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        if self.augment:
            self.transform = A.Compose(base_transforms)
        else:
            # keep fewer stochastic ops when not augmenting
            val_transforms = [A.Resize(self.image_size, self.image_size), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
            self.transform = A.Compose(val_transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Load with OpenCV for histogram equalization control
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Basic enhancements to handle low-light and blur
        img = equalize_histogram_cv(img)
        img = simple_deblur(img)

        augmented = self.transform(image=img)
        tensor = augmented["image"]
        return tensor, label


def build_dataloaders(train_dir, test_dir, image_size=224, batch_size=32, num_workers=4):
    """Create train and test dataloaders from directories.

    Returns (train_loader, val_loader, classes)
    """
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    train_ds = TrafficSignDataset(train_dir, classes=classes, image_size=image_size, augment=True)
    val_ds = TrafficSignDataset(test_dir, classes=classes, image_size=image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, classes
