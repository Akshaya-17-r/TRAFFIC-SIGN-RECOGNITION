"""
Training utilities for the tsr_project package.

Provides a train() function that reads `config.json` by default and trains a ViT.
"""
import json
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from .model import create_vit_model
from .model_utils import save_checkpoint


def build_loaders(train_dir, val_dir, image_size=224, batch_size=32, num_workers=4):
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
    val_ds = datasets.ImageFolder(val_dir, transform=transform_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds.classes


def train(config_path='config.json', progress_callback=None):
    with open(config_path) as f:
        cfg = json.load(f)
    train_dir = cfg['dataset']['train_dir']
    val_dir = cfg['dataset']['test_dir']
    image_size = cfg['training'].get('image_size', 224)
    batch_size = cfg['training'].get('batch_size', 32)
    epochs = cfg['training'].get('epochs', 20)
    lr = cfg['training'].get('lr', 1e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, classes = build_loaders(train_dir, val_dir, image_size=image_size, batch_size=batch_size, num_workers=cfg['training'].get('num_workers',4))
    model = create_vit_model(len(classes), pretrained=True, model_name=cfg['training'].get('model_name','vit_base_patch16_224'), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    patience = cfg['training'].get('early_stopping_patience', 5)
    wait = 0

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        start = time.time()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss/total
        train_acc = correct/total

        # validation
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, time: {elapsed:.1f}s")

        # report progress to callback (used by dashboards)
        if progress_callback is not None:
            try:
                progress_callback(epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
            except Exception:
                pass

        ckpt_path = os.path.join(cfg['paths'].get('checkpoints','checkpoints'), f"ckpt_epoch_{epoch}.pth")
        save_checkpoint({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'val_acc':val_acc}, ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping triggered')
                break

    # save final best
    final_path = os.path.join(cfg['paths'].get('checkpoints','checkpoints'), 'final.pth')
    save_checkpoint({'model_state_dict':model.state_dict(),'val_acc':best_val_acc}, final_path)
    return model, {'best_val_acc':best_val_acc}


if __name__ == '__main__':
    train()
"""
Training loop for Vision Transformer on traffic sign dataset.

Configurable via `config.json`. Saves checkpoints, logs training/validation
metrics and supports early stopping.
"""
import os
import json
import time
from collections import deque

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from tsr_project.data import build_dataloaders
from tsr_project.model import create_vit_model, save_checkpoint


def train(config_path="config.json"):
    # Load config
    with open(config_path) as f:
        cfg = json.load(f)

    train_dir = cfg["dataset"]["train_dir"]
    val_dir = cfg["dataset"]["test_dir"]
    image_size = cfg["training"].get("image_size", 224)
    batch_size = cfg["training"].get("batch_size", 32)
    epochs = cfg["training"].get("epochs", 20)
    lr = cfg["training"].get("lr", 1e-4)
    num_workers = cfg["training"].get("num_workers", 4)
    patience = cfg["training"].get("early_stopping_patience", 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, classes = build_dataloaders(train_dir, val_dir, image_size=image_size, batch_size=batch_size, num_workers=num_workers)

    model = create_vit_model(num_classes=len(classes), pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir="runs/tsr")

    best_val_acc = 0.0
    best_epoch = -1
    early_stop_queue = deque(maxlen=patience)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        start_time = time.time()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, time: {elapsed:.1f}s")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        # Checkpointing
        ckpt_path = save_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }, checkpoint_dir=cfg["paths"].get("checkpoints", "checkpoints"), filename=f"ckpt_epoch_{epoch}.pth")

        # Early stopping tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            early_stop_queue.clear()
        else:
            early_stop_queue.append(val_acc)

        if len(early_stop_queue) == patience and max(early_stop_queue) <= best_val_acc:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} with val_acc={best_val_acc:.4f}")
            break

    writer.close()


if __name__ == "__main__":
    train()
