"""Dataset loading and preprocessing helpers for the Dash app."""
import os
from PIL import Image, ImageFilter
import cv2
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resize_and_equalize(path, out_path, size=224):
    img = Image.open(path).convert('RGB')
    img = img.resize((size,size), Image.BILINEAR)
    # apply histogram equalization using OpenCV on Y channel
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(arr)
    y = cv2.equalizeHist(y)
    result = cv2.cvtColor(cv2.merge([y,cr,cb]), cv2.COLOR_YCrCb2RGB)
    out = Image.fromarray(result)
    out.save(out_path)


def sharpen_image(img):
    return img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
