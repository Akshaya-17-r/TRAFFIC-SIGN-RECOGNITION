"""
Utilities for creating and loading Vision Transformer models using timm.

This module exposes a helper to create a ViT (or other timm models) and
optionally load a checkpoint.
"""
import timm
import torch
import torch.nn as nn
import os


def create_model(num_classes, model_name='vit_base_patch16_224', pretrained=True, device=None):
    model = timm.create_model(model_name, pretrained=pretrained)
    # Replace classifier head in a few common locations
    if hasattr(model, 'head'):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError('Unable to find classifier head to replace for model: ' + model_name)

    if device is not None:
        model.to(device)
    return model


def load_checkpoint(model, path, device=None):
    if device is None:
        device = 'cpu'
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    return model


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
