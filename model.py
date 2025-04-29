import torch.nn as nn
import torchvision.models as models
from mlp_mixer import MLPMixer

def get_model(model_name, weights=None):
    model_name = model_name.lower().strip()

    model_mapping = {
        "efficientnet": "efficientnet_b0",
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_v2_s": "efficientnet_v2_s",
        "resnet50": "resnet50",
        "mlp_mixer": "mlp_mixer"
    }

    if model_name not in model_mapping:
        raise ValueError(f"❌ Unsupported model '{model_name}'")

    if model_name == "mlp_mixer":
        return MLPMixer(image_size=224, patch_size=16, in_channels=3, num_classes=5)

    corrected_model_name = model_mapping[model_name]
    model_class = getattr(models, corrected_model_name, None)

    if model_class is None or not callable(model_class):
        raise ValueError(f"❌ '{corrected_model_name}' model can’t be loaded from torchvision!")

    if weights == "DEFAULT":
        weights = getattr(models, f"{corrected_model_name.upper()}_Weights").DEFAULT
    elif isinstance(weights, str):
        weights = eval(weights)

    model = model_class(weights=weights)

    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, 5)
    elif hasattr(model, "classifier"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 5)
    else:
        raise ValueError(f"❌ Cannot modify output layer for model '{corrected_model_name}'")

    return model
