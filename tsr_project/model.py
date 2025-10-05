"""
ViT model loader and helper for traffic sign classification.

Loads a pretrained ViT from timm and replaces the head with the correct number
of classes. Provides helper to save/load checkpoints.
"""
from .model_utils import create_model, load_checkpoint, save_checkpoint


def create_vit_model(num_classes, pretrained=True, model_name="vit_base_patch16_224", device=None):
    """Wrapper to create a ViT model using model_utils.create_model"""
    return create_model(num_classes=num_classes, model_name=model_name, pretrained=pretrained, device=device)

