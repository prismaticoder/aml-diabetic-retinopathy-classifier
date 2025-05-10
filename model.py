import torch.nn as nn
import torchvision.models as models
from mlp_mixer import MLPMixer
from mlp_mixer_v2 import MLPMixerV2  # ✅ New import


def get_model(model_name, weights=None, n_classes=5):
    model_name = model_name.lower().strip()

    model_mapping = {
        "efficientnet": "efficientnet_b0",
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_v2_s": "efficientnet_v2_s",
        "resnet50": "resnet50",
        "mlp_mixer": "mlp_mixer",
        "mlp_mixer_v2": "mlp_mixer_v2",  # ✅ Add v2 entry
    }

    if model_name not in model_mapping:
        raise ValueError(f"❌ Unsupported model '{model_name}'")

    if model_name == "mlp_mixer":
        return MLPMixer(image_size=224, patch_size=16, in_channels=3, num_classes=n_classes)

    if model_name == "mlp_mixer_v2":  # ✅ Handle v2 logic
        return MLPMixerV2(image_size=224, patch_size=16, in_channels=3, num_classes=n_classes)

    # Handle torchvision models with weights
    if model_name == "resnet50":
        model = models.resnet50(weights=getattr(models, weights) if weights != "DEFAULT" else None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=getattr(models, weights) if weights != "DEFAULT" else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
        return model

    if model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=getattr(models, weights) if weights != "DEFAULT" else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
        return model

    raise ValueError(f"❌ Unknown model architecture '{model_name}'")
