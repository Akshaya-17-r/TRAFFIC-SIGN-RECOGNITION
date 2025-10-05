"""
Utility to download YOLOv8n weights to the workspace.

Tries a list of known URLs and writes to the destination path. Returns True on success.
"""
import requests
import os
from pathlib import Path
from tqdm import tqdm

DEFAULT_URLS = [
    "https://ultralytics.com/assets/models/yolov8n.pt",
    "https://github.com/ultralytics/ultralytics/releases/download/v8.0.142/yolov8n.pt",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
]


def download_file(url, dest_path, chunk_size=8192):
    r = requests.get(url, stream=True, timeout=30)
    if r.status_code != 200:
        return False
    total = int(r.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return True


def download_yolo(dest='yolov8n.pt', urls=None):
    urls = urls or DEFAULT_URLS
    dest = Path(dest)
    for url in urls:
        try:
            print(f"Trying to download from {url} -> {dest}")
            ok = download_file(url, dest)
            if ok and dest.exists() and dest.stat().st_size > 1000:
                print(f"Downloaded YOLO weights to {dest}")
                return str(dest)
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue
    raise RuntimeError('Failed to download YOLO weights from known URLs')


if __name__ == '__main__':
    print('Downloading yolov8n.pt ...')
    print(download_yolo())
