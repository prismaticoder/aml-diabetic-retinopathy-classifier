import torch
import torch.nn as nn

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class MixerBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim, token_dim, channel_dim):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Transpose(1, 2),
            nn.Linear(num_patches, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_patches),
            Transpose(1, 2)
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, hidden_dim)
        )

    def forward(self, x):
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes, num_blocks=8, hidden_dim=512, token_dim=256, channel_dim=2048):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            Transpose(1, 2)
        )

        self.blocks = nn.Sequential(*[
            MixerBlock(num_patches, hidden_dim, token_dim, channel_dim)
            for _ in range(num_blocks)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        return self.mlp_head(x.mean(dim=1))
