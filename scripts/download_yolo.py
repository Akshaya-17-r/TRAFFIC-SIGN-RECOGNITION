"""Download YOLOv8n weights to project root if missing."""
import os
import requests

URL = 'https://ultralytics.com/assets/yolov8n.pt'
DEST = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yolov8n.pt')

def download():
    if os.path.exists(DEST):
        print(f'Found existing {DEST}')
        return DEST
    print('Downloading YOLOv8n weights (about 25MB)...')
    r = requests.get(URL, stream=True)
    r.raise_for_status()
    with open(DEST, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print('Downloaded to', DEST)
    return DEST

if __name__ == '__main__':
    download()
