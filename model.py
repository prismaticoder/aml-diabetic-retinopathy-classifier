import torch
import torch.nn as nn

def get_model(model_name, weights="DEFAULT", n_classes=5, patch_size=16):
    """
    Returns a model instance based on the given model_name.

    Parameters:
    - model_name (str): Name of the model architecture.
    - weights (str): Pretrained weights setting (only used for torchvision models).
    - n_classes (int): Number of output classes.
    - patch_size (int): Patch size (used for MLP-Mixer-style models).

    Returns:
    - torch.nn.Module: Instantiated model ready for training.
    """
    if model_name == "mlp_mixer_v2_addblock":
        from mlp_mixer_v2_addblock import MLPMixerV2AddBlock
        return MLPMixerV2AddBlock(
            image_size=224,
            patch_size=patch_size,
            in_channels=3,
            num_classes=n_classes,
            dim=512,
            depth=9,
            token_dim=256,
            channel_dim=2048
        )
    
    elif model_name == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(pretrained=(weights == "DEFAULT"))
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model
    
    elif model_name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=(weights == "DEFAULT"))
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
        return model

    else:
        raise ValueError(f"Unsupported model: {model_name}")
