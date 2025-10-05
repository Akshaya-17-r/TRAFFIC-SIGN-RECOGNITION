from .detect import Detector
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sample = ROOT / 'dataset' / 'test' / 'stop' / 'stop_0.png'

def main():
    d = Detector()
    d.load_vit(num_classes=5)
    img = cv2.imread(str(sample))
    outs = d.detect_frame(img, ['speed_limit_20','stop','yield','no_entry','turn_right'])
    print('Detections:', outs)

if __name__ == '__main__':
    main()
