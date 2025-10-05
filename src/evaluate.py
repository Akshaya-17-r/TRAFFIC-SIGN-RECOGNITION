import os
import json
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import TrafficSignDataset
from src.model import create_vit_model, load_checkpoint
from PIL import Image


def evaluate(config_path='config.json', checkpoint=None):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    test_dir = cfg['dataset']['test_dir']
    image_size = cfg['training']['image_size']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = TrafficSignDataset(test_dir, image_size=image_size, train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    model = create_vit_model(len(ds.classes), image_size=image_size, pretrained=False)
    if checkpoint:
        load_checkpoint(checkpoint, model=model, map_location=device)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    misclassified_dir = cfg['paths']['results']
    os.makedirs(misclassified_dir, exist_ok=True)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.numpy().tolist())

    print(classification_report(all_targets, all_preds, target_names=ds.classes))
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False)
    plt.title('Confusion Matrix')
    cm_path = os.path.join(misclassified_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)

    # Save misclassified examples
    for i, (pred, tgt) in enumerate(zip(all_preds, all_targets)):
        if pred != tgt:
            # attempt to copy corresponding image
            try:
                src_path, _ = ds.samples[i]
                img = Image.open(src_path).convert('RGB')
                img.save(os.path.join(misclassified_dir, f'mis_{i}_pred_{ds.classes[pred]}_true_{ds.classes[tgt]}.png'))
            except Exception:
                pass


if __name__ == '__main__':
    evaluate()
