import torch
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class MLPBlock(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class MixerLayer(nn.Module):
    def __init__(self, num_patches, embed_dim, token_mlp_dim, channel_mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.token_mixing = nn.Sequential(
            Transpose(1, 2),
            MLPBlock(num_patches, token_mlp_dim, dropout),
            Transpose(1, 2)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.channel_mixing = MLPBlock(embed_dim, channel_mlp_dim, dropout)

    def forward(self, x):
        x = x + self.token_mixing(self.norm1(x))
        x = x + self.channel_mixing(self.norm2(x))
        return x


class MLPMixerV2(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=5,
                 embed_dim=512, depth=8, token_mlp_dim=256, channel_mlp_dim=2048, dropout=0.):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=patch_size),
            Transpose(1, 2),
            nn.Linear(patch_dim, embed_dim)
        )

        self.mixer_layers = nn.Sequential(
            *[MixerLayer(num_patches, embed_dim, token_mlp_dim, channel_mlp_dim, dropout) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)  # B x num_patches x embed_dim
        x = self.mixer_layers(x)     # B x num_patches x embed_dim
        x = self.norm(x)
        x = x.mean(dim=1)            # Global average pooling
        return self.head(x)
