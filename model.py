import torch
import torch.nn as nn
import torchvision.models as models

# Function to dynamically load models
def get_model(model_name):
    model_name = model_name.lower().strip()  # Ensure case-insensitivity and remove spaces

    # Fix: Map valid model names
    model_mapping = {
        "efficientnet": "efficientnet_b0",  # Fix incorrect base name
        "efficientnet_b0": "efficientnet_b0",
        "resnet50": "resnet50"
    }

    if model_name not in model_mapping:
        raise ValueError(f"❌ '{model_name}' is not a valid callable model in torchvision!")

    corrected_model_name = model_mapping[model_name]

    # Get the model function properly
    model_class = getattr(models, corrected_model_name, None)
    
    if model_class is None or not callable(model_class):
        raise ValueError(f"❌ '{corrected_model_name}' is not a valid callable model in torchvision!")

    # Handle weight loading based on torchvision version
    try:
        weights = None  # Default to None

        if corrected_model_name == "efficientnet_b0":
            from torchvision.models.efficientnet import EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT
        elif corrected_model_name == "resnet50":
            from torchvision.models.resnet import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT

        model = model_class(weights=weights)

    except TypeError:
        model = model_class(pretrained=True)  # Fallback for older torchvision versions


    # Adjust last classification layer for 5 classes
    if hasattr(model, "fc"):  # ResNet, DenseNet
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)
    elif hasattr(model, "classifier"):  # VGG, EfficientNet
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 5)
    else:
        raise ValueError(f"❌ Unable to modify classification layer for {corrected_model_name}")

    return model
