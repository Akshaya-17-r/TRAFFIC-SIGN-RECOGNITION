import os
import json
import time
import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from src.data import get_dataloaders
from src.model import create_vit_model, save_checkpoint
from sklearn.metrics import accuracy_score


def train(config_path='config.json'):
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    train_dir = cfg['dataset']['train_dir']
    test_dir = cfg['dataset']['test_dir']
    bs = cfg['training']['batch_size']
    epochs = cfg['training']['epochs']
    lr = cfg['training']['lr']
    image_size = cfg['training']['image_size']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, classes = get_dataloaders(train_dir, test_dir, batch_size=bs, image_size=image_size)
    num_classes = len(classes)
    model = create_vit_model(num_classes, image_size=image_size, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # Simple CSV logger (avoids importing tensorboard which can pull in TensorFlow)
    os.makedirs('logs', exist_ok=True)
    train_log_path = os.path.join('logs', 'train_log.csv')
    if not os.path.exists(train_log_path):
        with open(train_log_path, 'w', newline='', encoding='utf-8') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['epoch','loss','train_acc','val_acc','timestamp'])
    best_acc = 0.0
    patience = cfg['training'].get('early_stopping_patience', 5)
    stale = 0

    os.makedirs(cfg['paths']['checkpoints'], exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.detach().cpu().numpy().tolist())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_targets, all_preds)

        # validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                val_preds.extend(preds.tolist())
                val_targets.extend(labels.detach().cpu().numpy().tolist())
        val_acc = accuracy_score(val_targets, val_preds)

        print(f"Epoch {epoch} loss={epoch_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        with open(train_log_path, 'a', newline='', encoding='utf-8') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, f'{epoch_loss:.6f}', f'{train_acc:.6f}', f'{val_acc:.6f}', time.time()])

        # checkpoint
        ckpt_path = os.path.join(cfg['paths']['checkpoints'], f"vit_epoch{epoch}.pt")
        save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'val_acc': val_acc}, ckpt_path)

        if val_acc > best_acc:
            best_acc = val_acc
            stale = 0
            best_path = os.path.join(cfg['paths']['checkpoints'], 'best.pt')
            save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'val_acc': val_acc}, best_path)
        else:
            stale += 1

        if stale >= patience:
            print('Early stopping triggered')
            break

    # finished


if __name__ == '__main__':
    train()
