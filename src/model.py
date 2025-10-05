import torch
import torch.nn as nn
import timm


def create_vit_model(num_classes, image_size=224, pretrained=True, model_name='vit_base_patch16_224'):
    # timm provides ViT models pretrained on ImageNet
    model = timm.create_model(model_name, pretrained=pretrained)
    # replace classifier head
    if hasattr(model, 'head'):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        # generic last layer name
        model.reset_classifier(num_classes)
    return model


def save_checkpoint(state, path):
    torch.save(state, path)


def load_checkpoint(path, model=None, optimizer=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    if model is not None and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt
