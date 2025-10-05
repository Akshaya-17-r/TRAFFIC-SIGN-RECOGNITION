"""Training script: uses PyTorch for training the selected model.
Configurable via command-line or Dash callbacks.
"""
import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from scripts.model import create_vit_model, save_checkpoint
import pandas as pd


def get_dataloaders(train_dir, val_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, train_ds.classes


def train_loop(model, train_loader, val_loader, epochs, lr, device, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    history = {'train_loss':[], 'val_acc':[], 'val_loss':[]}
    patience = 5
    wait = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        # validation
        model.eval()
        correct = 0
        vloss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outs = model(imgs)
                loss = criterion(outs, labels)
                vloss += loss.item() * imgs.size(0)
                preds = outs.argmax(dim=1)
                correct += (preds==labels).sum().item()
        val_loss = vloss / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f'Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
        # checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({'epoch': epoch+1, 'model_state': model.state_dict(), 'val_acc': val_acc}, os.path.join(checkpoint_dir, 'best.pt'))
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping')
                break
    # save history
    pd.DataFrame(history).to_csv(os.path.join(checkpoint_dir, 'train_log.csv'), index=False)
    return history


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--val_dir', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train_loader, val_loader, classes = get_dataloaders(args.train_dir, args.val_dir, args.batch)
    model = create_vit_model(len(classes))
    train_loop(model, train_loader, val_loader, args.epochs, args.lr, args.device)
