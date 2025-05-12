import torch
import torch.nn as nn
from mlp_mixer_v2_batchnorm import MixerLayerWithBatchNorm

# ✅ Custom Transpose module
class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class MLPMixerV2AddBlock(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=5,
                 dim=512, depth=9, token_dim=256, channel_dim=2048):
        super().__init__()

        assert image_size % patch_size == 0, "Image dimensions must be divisible by patch size."
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # ✅ Patch embedding using unfold
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)  # (B, C*P*P, N)
        self.linear_proj = nn.Linear(patch_size * patch_size * in_channels, dim)  # (B, N, dim)

        # ✅ Mixer layers
        self.mixer_layers = nn.Sequential(
            *[MixerLayerWithBatchNorm(self.num_patches, token_dim, channel_dim, dim) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x shape: [B, C, H, W] e.g. [32, 3, 224, 224]
        print("Input image shape:", x.shape)

        # Patch embedding
        x = self.unfold(x)  # [B, C*P*P, N]
        print("After unfold:", x.shape)

        x = x.transpose(1, 2)  # [B, N, C*P*P]
        print("After transpose:", x.shape)

        x = self.linear_proj(x)  # [B, N, dim]
        print("After linear projection:", x.shape)

        # MLP-Mixer block
        x = self.mixer_layers(x)  # [B, N, dim]
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling over patches
        return self.head(x)  # [B, num_classes]
