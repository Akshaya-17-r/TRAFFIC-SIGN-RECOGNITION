"""Create a tiny synthetic dataset for quick testing.
Creates dataset/train/<class>/ and dataset/test/<class>/ with simple colored squares.
"""
import os
from PIL import Image, ImageDraw

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
CLASSES = ['speed_limit_20','stop','yield','no_entry','turn_right']

def make_square(path, color):
    img = Image.new('RGB', (300,300), (255,255,255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([75,75,225,225], fill=color)
    img.save(path)

def generate(num_train=10, num_test=4):
    for split, n in [('train', num_train), ('test', num_test)]:
        for cls in CLASSES:
            d = os.path.join(ROOT, split, cls)
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                color = {
                    'speed_limit_20': (200,0,0),
                    'stop': (255,0,0),
                    'yield': (255,200,0),
                    'no_entry': (0,0,0),
                    'turn_right': (0,0,255),
                }[cls]
                make_square(os.path.join(d, f'{cls}_{i}.png'), color)
    print('Synthetic dataset created at', ROOT)

if __name__ == '__main__':
    generate()
