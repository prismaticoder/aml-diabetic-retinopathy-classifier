import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    resnet50, ResNet50_Weights,
)

def get_model(model_name, pretrained=False):
    model_name = model_name.lower().strip()

    # Supported model mappings (add aliases here)
    model_mapping = {
        "efficientnet": "efficientnet_b0",  # alias fix
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_v2_s": "efficientnet_v2_s",
        "resnet50": "resnet50"
    }

    if model_name not in model_mapping:
        raise ValueError(f"❌ Unsupported model '{model_name}'")

    corrected_model_name = model_mapping[model_name]
    model_class = getattr(models, corrected_model_name, None)

    if model_class is None or not callable(model_class):
        raise ValueError(f"❌ '{corrected_model_name}' model can’t be loaded from torchvision!")

    # Load pretrained weights if needed
    model = model_class(pretrained=pretrained)

    # Replace final classification layer
    if hasattr(model, "fc"):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)
    elif hasattr(model, "classifier"):
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 5)
    else:
        raise ValueError(f"❌ Cannot modify output layer for model '{corrected_model_name}'")

    return model

