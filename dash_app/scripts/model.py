"""Model utilities: ViT loader using timm, replace head to match number of classes.
"""
import torch
import torch.nn as nn
import timm


def create_vit_model(num_classes, pretrained=True, model_name='vit_base_patch16_224'):
    model = timm.create_model(model_name, pretrained=pretrained)
    if hasattr(model, 'head'):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError('Unknown ViT model head')
    return model


def save_checkpoint(state, path):
    torch.save(state, path)


def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device)


if __name__ == '__main__':
    m = create_vit_model(43)
    print(m)
