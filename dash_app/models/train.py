"""Model training utilities for Dash app. Uses timm ViT by default."""
import os
import json
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


def create_vit(num_classes, model_name='vit_base_patch16_224', pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    if hasattr(model, 'head'):
        in_f = model.head.in_features
        model.head = nn.Linear(in_f, num_classes)
    else:
        model.reset_classifier(num_classes)
    return model


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    preds = []
    targets = []
    criterion = nn.CrossEntropyLoss()
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
        preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
        targets.extend(y.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(targets,preds)
    return avg_loss, acc
