"""Quick demo: prepare synthetic dataset, download YOLO, run one training epoch and a single detection pass.
This is for validation only, not for production training.
"""
import os
import time
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'

def run(cmd):
    print('RUN:', cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def main():
    # 1) generate synthetic dataset
    run(f'python "{SCRIPTS / "generate_synthetic_dataset.py"}"')
    # 2) download yolov8n weights
    run(f'python "{SCRIPTS / "download_yolo.py"}"')
    # 3) run one-epoch training (use train.py which reads config.json)
    # We'll temporarily modify config to run 1 epoch and small batch size
    cfg = ROOT / 'config.json'
    import json
    with open(cfg,'r') as f:
        c = json.load(f)
    old_epochs = c['training']['epochs']
    old_batch = c['training']['batch_size']
    c['training']['epochs'] = 1
    c['training']['batch_size'] = 8
    with open(cfg,'w') as f:
        json.dump(c, f, indent=2)
    try:
        run(f'python run.py --mode train --config config.json')
    finally:
        # restore config
        c['training']['epochs'] = old_epochs
        c['training']['batch_size'] = old_batch
        with open(cfg,'w') as f:
            json.dump(c, f, indent=2)

    print('Demo training complete — running a single detection on a sample image')
    sample = ROOT / 'dataset' / 'test' / 'stop' / 'stop_0.png'
    run(f'python -c "from src.detect import Detector; d=Detector(); d.load_vit(num_classes=5); import cv2; img=cv2.imread(\"{sample}\"); outs=d.detect_frame(img, [\'speed_limit_20\',\'stop\',\'yield\',\'no_entry\',\'turn_right\']); print(outs)"')

if __name__ == '__main__':
    main()
