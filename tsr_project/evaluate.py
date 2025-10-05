"""
Evaluation utilities for the TSR project.

Computes confusion matrix, classification report and saves misclassified
images to `results/`.
"""
import os
import json
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import torch
import shutil
from tsr_project.data import TrafficSignDataset


def evaluate(model, test_dir, classes, device="cpu", image_size=224, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    ds = TrafficSignDataset(test_dir, classes=classes, image_size=image_size, augment=False)

    model.eval()
    y_true = []
    y_pred = []

    for i in range(len(ds)):
        img_tensor, label = ds[i]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            pred = torch.argmax(outputs, dim=1).item()
        y_true.append(label)
        y_pred.append(pred)
        if pred != label:
            # Save misclassified image
            src_path = ds.samples[i][0]
            dst = os.path.join(results_dir, f"mis_{i}_{classes[label]}_as_{classes[pred]}.jpg")
            shutil.copy(src_path, dst)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    return cm, report
