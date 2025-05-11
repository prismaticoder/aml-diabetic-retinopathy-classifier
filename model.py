import torch
import torch.nn as nn
import torchvision.models as models
from maxvit import MaxViT

from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    resnet50, ResNet50_Weights,
    maxvit_t, MaxViT_T_Weights,
)

def get_rsgnet(n_classes=5):
    class RSGNet(nn.Module):
        def __init__(self):
            super(RSGNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(32 * 56 * 56, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, n_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return RSGNet()

def get_model(name, weights="DEFAULT", n_classes=5):
    name = name.lower()
    if name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if weights == "DEFAULT" else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    elif name == "efficientnet_v2_s":
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if weights == "DEFAULT" else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    elif name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT if weights == "DEFAULT" else None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif name == "rsgnet":
        model = get_rsgnet(n_classes=n_classes)
    elif name == "maxvit":
        model = MaxViT(n_classes=n_classes, use_mbconv=True, use_block_attn=True, use_grid_attn=True)
    elif name == "maxvit_t":
        model = maxvit_t(weights=MaxViT_T_Weights.DEFAULT, num_classes=n_classes)
    else:
        raise ValueError(f"❌ Unsupported model: {name}")
    return model
