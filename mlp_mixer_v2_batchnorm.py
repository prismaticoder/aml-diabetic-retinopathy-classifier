import torch
import torch.nn as nn

class MixerLayerWithBatchNorm(nn.Module):
    def __init__(self, num_patches, token_dim, channel_dim, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)

        self.token_mixing = nn.Sequential(
            nn.Linear(num_patches, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_patches)
        )

        self.channel_mixing = nn.Sequential(
            nn.Linear(dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, dim)
        )

    def forward(self, x):
        # x: [B, N, dim]
        B, N, D = x.shape
        assert D == self.norm1.num_features, f"Expected dim={self.norm1.num_features}, got {D}"

        # Token mixing
        x_trans = x.transpose(1, 2)  # [B, dim, N]
        x_norm = self.norm1(x_trans)
        x_mix = self.token_mixing(x_norm)
        x = x + x_mix.transpose(1, 2)  # Residual + reshape back to [B, N, dim]

        # Channel mixing
        x_norm2 = self.norm2(x.transpose(1, 2))
        x = x + self.channel_mixing(x_norm2.transpose(1, 2))  # [B, N, dim]

        return x
