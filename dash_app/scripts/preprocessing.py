"""Preprocessing utilities: load images from dataset, resize to 224x224, histogram equalization,
augmentations (rotate, brightness, contrast), and save to processed folders.
"""
import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
from tqdm import tqdm


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def histogram_equalize(img):
    # img: PIL Image RGB
    img_yuv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_eq)


def preprocess_image(img_path, out_path, size=(224,224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size, Image.BILINEAR)
    # apply CLAHE (adaptive histogram equalization) to handle low-light
    img_np = np.array(img)
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img_eq = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    img = Image.fromarray(img_eq)
    # mild deblur/sharpen
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    ensure_dir(os.path.dirname(out_path))
    img.save(out_path)


def augment_image(img_path, out_dir, size=(224,224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size, Image.BILINEAR)
    ensure_dir(out_dir)
    base = os.path.splitext(os.path.basename(img_path))[0]
    # rotations
    for angle in [0, -15, 15, -30, 30]:
        rot = img.rotate(angle)
        rot = histogram_equalize(rot)
        rot.save(os.path.join(out_dir, f'{base}_rot{angle}.jpg'))
    # brightness
    enhancer = ImageEnhance.Brightness(img)
    for i, f in enumerate([0.7, 0.9, 1.1, 1.3]):
        out = enhancer.enhance(f)
        out = histogram_equalize(out)
        out.save(os.path.join(out_dir, f'{base}_b{i}.jpg'))
    # contrast
    enhancer = ImageEnhance.Contrast(img)
    for i, f in enumerate([0.7, 1.0, 1.3]):
        out = enhancer.enhance(f)
        out = histogram_equalize(out)
        out.save(os.path.join(out_dir, f'{base}_c{i}.jpg'))


def process_dataset(src_dir, dst_dir, size=(224,224), augment=True):
    """Process an entire dataset directory organized by class folders.

    - src_dir: path to original dataset (contains class subfolders)
    - dst_dir: path where processed dataset will be written (same class subfolders)
    - size: output image size (width, height)
    - augment: whether to write augmented images into dst_dir/<class>/aug/

    Returns a dict with counts.
    """
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    stats = {'processed':0, 'augmented':0, 'classes':0}
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    stats['classes'] = len(classes)
    for cls in tqdm(classes, desc='Classes'):
        src_cls = os.path.join(src_dir, cls)
        dst_cls = os.path.join(dst_dir, cls)
        os.makedirs(dst_cls, exist_ok=True)
        files = [f for f in os.listdir(src_cls) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        for fname in files:
            src_path = os.path.join(src_cls, fname)
            dst_path = os.path.join(dst_cls, fname)
            try:
                preprocess_image(src_path, dst_path, size=size)
                stats['processed'] += 1
                if augment:
                    aug_dir = os.path.join(dst_cls, 'aug')
                    augment_image(src_path, aug_dir, size=size)
                    # count files in aug_dir
                    stats['augmented'] += len([1 for _ in os.listdir(aug_dir)])
            except Exception:
                # skip problematic files
                continue
    return stats


if __name__ == '__main__':
    # quick CLI
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    preprocess_image(args.input, args.output)
