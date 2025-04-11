import torch
import torch.nn as nn
import timm  # For SwinV2 and other transformer models
from torchvision import models

def get_model(model_name, weights=None, input_size=224):
    model_name = model_name.lower().strip()

    # Supported model mappings (add aliases here)
    model_mapping = {
        "efficientnet": "efficientnet_b0",  # alias fix
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_v2_s": "efficientnet_v2_s",
        "resnet50": "resnet50",
        "swin_v2_b": "swin_v2_b"  # SwinV2 model mapping
    }

    if model_name not in model_mapping:
        raise ValueError(f"❌ Unsupported model '{model_name}'")

    corrected_model_name = model_mapping[model_name]

    # Load the model using timm if it's a SwinV2 model, otherwise use torchvision
    if corrected_model_name == "swin_v2_b":
        model = timm.create_model('swinv2_base_window12_192', pretrained=True)
        input_size = 192  # SwinV2 expects 192x192 input
        model.reset_classifier(num_classes=5)  # Safely update classifier layer
        print("✅ SwinV2 model loaded and classifier reset")
    else:
        model_class = getattr(models, corrected_model_name, None)
        if model_class is None or not callable(model_class):
            raise ValueError(f"❌ '{corrected_model_name}' model can’t be loaded from torchvision!")

        # Load pretrained weights if needed
        if weights is None:
            weights = getattr(models, f"{corrected_model_name.upper()}_Weights").DEFAULT

        model = model_class(weights=weights)

        # Modify the classifier layer for 5 classes
        if hasattr(model, 'fc'):  # For ResNet
            num_features = model.fc.in_features
            print(f"Initial number of features before final linear layer: {num_features}")
            model.fc = nn.Linear(num_features, 5)
        elif hasattr(model, 'classifier'):  # For EfficientNet
            num_features = model.classifier[-1].in_features
            print(f"Initial number of features before final linear layer: {num_features}")
            model.classifier[-1] = nn.Linear(num_features, 5)
        else:
            raise ValueError(f"❌ Cannot modify output layer for model '{corrected_model_name}'")

    return model, input_size
