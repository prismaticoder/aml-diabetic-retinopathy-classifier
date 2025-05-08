import torch
import torch.nn as nn

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class MixerLayer(nn.Module):
    def __init__(self, num_patches, token_dim, channel_dim):
        super(MixerLayer, self).__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.token_mlp = nn.Sequential(
            Transpose(1, 2),
            MLPBlock(num_patches, token_dim),
            Transpose(1, 2)
        )

        self.norm2 = nn.LayerNorm(token_dim)
        self.channel_mlp = MLPBlock(token_dim, channel_dim)

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.token_mlp(y)
        y = self.norm2(x)
        x = x + self.channel_mlp(y)
        return x

class MLPMixer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=5, dim=512, depth=8, token_dim=256, channel_dim=2048):
        super(MLPMixer, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            Transpose(1, 2)  # (B, dim, N) -> (B, N, dim)
        )

        self.mixer_layers = nn.Sequential(*[
            MixerLayer(num_patches, dim, channel_dim)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)        # Shape: (B, N, dim)
        x = self.mixer_layers(x)           # Shape: (B, N, dim)
        x = self.norm(x)                   # Shape: (B, N, dim)
        x = x.mean(dim=1)                  # Global average pooling
        return self.head(x)                # Shape: (B, num_classes)
