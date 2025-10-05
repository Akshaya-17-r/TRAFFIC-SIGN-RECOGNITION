"""Evaluation utilities: confusion matrix, classification report, save misclassified images.
"""
import os
import torch
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, test_dir, device='cpu', out_dir='results'):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    ds = datasets.ImageFolder(test_dir, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    y_true = []
    y_pred = []
    y_scores = []
    # If no model is provided, use a simple majority-class baseline
    if model is None:
        # count class frequencies in dataset
        class_counts = {}
        for _, label in ds.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        if not class_counts:
            raise RuntimeError('No samples in test dataset')
        majority_class = max(class_counts.items(), key=lambda x: x[1])[0]
        for _, label in ds.samples:
            y_true.append(label)
            y_pred.append(majority_class)
    else:
        model.to(device)
        model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                outs = model(imgs)
                probs = torch.softmax(outs, dim=1)
                pconf, preds = torch.max(probs, dim=1)
                preds = preds.cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(labels.numpy().tolist())
                # collect confidence for predicted class
                y_scores.extend([float(pc.cpu().item()) for pc in pconf])
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))

    # Save classification report as JSON
    import json as _json
    try:
        with open(os.path.join(out_dir, 'classification_report.json'), 'w', encoding='utf-8') as _f:
            _json.dump(report, _f, indent=2)
    except Exception:
        pass

    # Save confidence histogram if we have scores
    try:
        if y_scores:
            plt.figure(figsize=(6,4))
            plt.hist(y_scores, bins=20, range=(0,1))
            plt.title('Prediction confidence distribution')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            plt.savefig(os.path.join(out_dir, 'confidence_hist.png'))
            plt.close()
    except Exception:
        pass

    # Save misclassified images
    mis_dir = os.path.join(out_dir, 'misclassified')
    os.makedirs(mis_dir, exist_ok=True)
    try:
        # ds.samples contains (path, label)
        for idx, (path, label) in enumerate(ds.samples):
            pred = y_pred[idx]
            if pred != label:
                # copy to mis_dir with annotation
                import shutil
                base = os.path.basename(path)
                dst = os.path.join(mis_dir, f"mis_{idx}_{ds.classes[label]}_as_{ds.classes[pred]}_{base}")
                try:
                    shutil.copy(path, dst)
                except Exception:
                    pass
    except Exception:
        pass

    return cm, report


def most_confused_pairs(cm, classes, top_k=10):
    """Return list of top_k most confused class pairs from a confusion matrix.

    Each item: (true_class, predicted_class, count)
    """
    pairs = []
    import numpy as _np
    cm = _np.array(cm)
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairs.append((i, j, int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    out = []
    for t, p, c in pairs[:top_k]:
        out.append((classes[t], classes[p], c))
    return out


def confidence_distribution(y_scores):
    """Given a list of confidence scores, return histogram bins and counts."""
    import numpy as _np
    if not y_scores:
        return [], []
    counts, bins = _np.histogram(_np.array(y_scores), bins=10, range=(0,1))
    return bins.tolist(), counts.tolist()


if __name__ == '__main__':
    print('Run evaluation from Dash or import functions in your workflow')
