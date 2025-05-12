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
            super().__init__()
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
                    nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 56 * 56, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, n_classes)
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
            else:
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

def get_model(model_name, weights=None, n_classes=5, model_variant="baseline"):
    model_name = model_name.lower()
    if model_name == "efficientnet":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if weights == "DEFAULT" else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    elif model_name == "efficientnet_v2_s":
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if weights == "DEFAULT" else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT if weights == "DEFAULT" else None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif model_name == "rsgnet":
        return get_rsgnet(n_classes=n_classes, variant=model_variant)
    else:
        raise ValueError(f"❌ Unsupported model: {model_name}")
    return model
