import os
import cv2
import numpy as np
from PIL import Image, ImageFilter


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def apply_clahe_rgb(img):
    # img: HxWxC in BGR (OpenCV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def sharpen_pil(pil_img):
    return pil_img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))


def resize_and_center_crop(img, size=224):
    # img: PIL Image
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = int(w*scale), int(h*scale)
    img = img.resize((nw, nh), Image.BILINEAR)
    left = (nw - size)//2
    top = (nh - size)//2
    return img.crop((left, top, left+size, top+size))
