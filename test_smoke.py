import json
import torch
from src.model import create_vit_model
from src.data import TrafficSignDataset

def smoke():
    print('Loading config...')
    with open('config.json') as f:
        cfg = json.load(f)
    print('Creating model...')
    model = create_vit_model(num_classes=5, image_size=cfg['training']['image_size'], pretrained=False)
    x = torch.randn(2,3,cfg['training']['image_size'], cfg['training']['image_size'])
    out = model(x)
    print('Model forward OK, output shape:', out.shape)
    print('Attempting to create dataset object (no file reads)...')
    try:
        ds = TrafficSignDataset(cfg['dataset']['train_dir'], image_size=cfg['training']['image_size'], train=False)
        print('Dataset classes (discovered):', ds.classes[:10])
    except Exception as e:
        print('Dataset creation warning (this is OK if dataset not present):', e)

if __name__ == '__main__':
    smoke()
