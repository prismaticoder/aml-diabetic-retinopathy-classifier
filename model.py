import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from mlp_mixer import MLPMixer


def get_model(model_name, weights="DEFAULT", n_classes=5, image_size=224, patch_size=16):
    model_name = model_name.lower()

    if model_name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if weights != "DEFAULT" else None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    elif model_name == "efficientnet_v2_s":
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if weights != "DEFAULT" else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    elif model_name == "mlp_mixer":
        model = MLPMixer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            num_classes=n_classes
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model