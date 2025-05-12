import torch
import torch.nn as nn
import torch.nn.functional as F

class RSGNet(nn.Module):
    def __init__(self, n_classes, variant="baseline"):
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