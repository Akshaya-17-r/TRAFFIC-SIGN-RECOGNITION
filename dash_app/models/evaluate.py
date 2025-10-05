"""Evaluation utilities for Dash app."""
import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def evaluate_model(model, loader, classes, device='cpu', out_dir='dash_results'):
    model.eval()
    preds = []
    targets = []
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(y.numpy().tolist())
    report = classification_report(targets, preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8,6))
    import seaborn as sns
    sns.heatmap(cm, annot=False)
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    return report, cm
