import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    resnet50, ResNet50_Weights,
)

def get_rsgnet(n_classes=5, variant="baseline"):
    class RSGNet(nn.Module):
        def __init__(self):
            super(RSGNet, self).__init__()

            if variant == "remove_layer":
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64, 64), nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.2),
                    nn.Linear(64, n_classes)
                )

            elif variant == "added_layer":
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),  # extra
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.2),
                    nn.Linear(64, n_classes)
                )

            elif variant == "avgpool":
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
                    nn.AvgPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                    nn.AvgPool2d(2),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.2),
                    nn.Linear(64, n_classes)
                )

            else:  # baseline
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.2),
                    nn.Linear(64, n_classes)
                )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    return RSGNet()

def get_model(model_name, weights="DEFAULT", n_classes=5, model_variant="baseline"):
    model_name = model_name.lower().strip()

    if model_name == "rsgnet":
        return get_rsgnet(n_classes=n_classes, variant=model_variant)

    model_mapping = {
        "efficientnet": "efficientnet_b0",
        "efficientnet_v2_s": "efficientnet_v2_s",
        "resnet50": "resnet50"
    }

    if model_name not in model_mapping:
        raise ValueError(f"❌ Unsupported model '{model_name}'")

    corrected_model_name = model_mapping[model_name]
    model_class = getattr(models, corrected_model_name, None)

    if model_class is None or not callable(model_class):
        raise ValueError(f"❌ '{corrected_model_name}' model can’t be loaded from torchvision!")

    if weights == "DEFAULT":
        weights = getattr(models, f"{corrected_model_name.upper()}_Weights").DEFAULT

    model = model_class(weights=weights)

    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif hasattr(model, "classifier"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    else:
        raise ValueError(f"❌ Cannot modify classifier for '{corrected_model_name}'")

    return model
