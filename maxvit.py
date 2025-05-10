import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- helpers ----------------------------------------------------------

def window_partition(x, window_size):
    """Split feature map [B,C,H,W] into non‑overlapping windows → [B*n, C, ws, ws]"""
    B, C, H, W = x.shape
    x = x.view(B, C,
               H // window_size, window_size,
               W // window_size, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()          # [B, H/ws, W/ws, C, ws, ws]
    windows = x.view(-1, C, window_size, window_size)      # merge batch & grid dims
    return windows

def window_reverse(windows, window_size, H, W):
    """Undo window_partition → recover full feature map [B,C,H,W]"""
    Bn, C, _, _ = windows.shape
    B = Bn // (H // window_size * W // window_size)
    x = windows.view(B, H // window_size, W // window_size, C,
                     window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, H, W)
    return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class WindowAttention(nn.Module):
    """Standard MSA performed inside windows (local or grid)."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):                 # x: [B*N, ws*ws, C]
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)               # each: [3, Bn, heads, N, dim_h]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        x = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        return self.proj(x)

# ---------- MaxViT block -----------------------------------------------------

class MaxViTBlock(nn.Module):
    """
    Minimal MaxViT block with:
        • MBConv (depth‑wise conv bottleneck)
        • Block‑wise attention  (local)
        • Grid‑wise  attention  (global‑ish)
        • MLP
    Toggle pieces with flags for ablation.
    """

    def __init__(self, dim, window_size=7,
                 use_mbconv=True, use_block_attn=True, use_grid_attn=True):
        super().__init__()
        self.ws = window_size
        self.use_mbconv   = use_mbconv
        self.use_block    = use_block_attn
        self.use_grid     = use_grid_attn

        if use_mbconv:
            self.mbconv = nn.Sequential(
                nn.Conv2d(dim, dim*4, 1, bias=False),
                nn.BatchNorm2d(dim*4),
                nn.GELU(),
                nn.Conv2d(dim*4, dim*4, 3, padding=1, groups=dim*4, bias=False),
                nn.BatchNorm2d(dim*4),
                nn.GELU(),
                nn.Conv2d(dim*4, dim, 1, bias=False),
                nn.BatchNorm2d(dim)
            )

        if use_block_attn:
            self.norm_b  = nn.LayerNorm(dim)
            self.attn_b  = WindowAttention(dim)

        if use_grid_attn:
            self.norm_g  = nn.LayerNorm(dim)
            self.attn_g  = WindowAttention(dim)

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp      = MLP(dim)

    # ---- forward -----------------------------------------------------------
    def forward(self, x):
        """
        x : Tensor shape [B, C, H, W]
        returns same shape
        """
        B, C, H, W = x.shape
        if self.use_mbconv:
            x = x + self.mbconv(x)                      # MBConv residual

        # ------------ Block attention (local windows) -----------------------
        if self.use_block:
            # partition
            windows = window_partition(x, self.ws)      # [B*n, C, ws, ws]
            windows = windows.flatten(2).transpose(1, 2)  # → [B*n, ws*ws, C]
            attn_out = self.attn_b(self.norm_b(windows))
            attn_out = attn_out.transpose(1, 2).view(-1, C, self.ws, self.ws)
            x = x + window_reverse(attn_out, self.ws, H, W)

        # ------------ Grid attention (interleaved grid) ---------------------
        if self.use_grid:
            # shift feature map so grid windows are interleaved
            x = torch.roll(x, shifts=(-self.ws // 2, -self.ws // 2), dims=(2, 3))
            windows = window_partition(x, self.ws)               # grid windows
            windows = windows.flatten(2).transpose(1, 2)
            attn_out = self.attn_g(self.norm_g(windows))
            attn_out = attn_out.transpose(1, 2).view(-1, C, self.ws, self.ws)
            x = window_reverse(attn_out, self.ws, H, W)
            # roll back
            x = torch.roll(x, shifts=(self.ws // 2, self.ws // 2), dims=(2, 3))

        # ------------ MLP ----------------------------------------------------
        # x_flat = x.flatten(2).transpose(1, 2)           # [B, H*W, C]
        # x = x + self.mlp(self.norm_mlp(x_flat))
        # x = x.transpose(1, 2).view(B, C, H, W)
        
        # Corrected MLP section
        x_flat = x.flatten(2).transpose(1, 2)           # [B, H*W, C]
        mlp_out = self.mlp(self.norm_mlp(x_flat))       # [B, H*W, C]
        mlp_out = mlp_out.transpose(1, 2).reshape(B, C, H, W)
        x = x + mlp_out
        return x

# ---------------------------------------------------------------
# NOTE: MaxViTBlock must already be defined / imported in scope
# ---------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Simple patch‑embedding via strided convolution"""
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_chans)
        self.act  = nn.GELU()

    def forward(self, x):                  # [B,3,H,W]
        x = self.proj(x)
        x = self.act(self.norm(x))
        return x                           # [B,C,H/2,W/2]


class Downsample(nn.Module):
    """2× downsample using depth‑wise conv + pointwise"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv_dw = nn.Conv2d(dim_in, dim_in, 3, stride=2, padding=1, groups=dim_in)
        self.conv_pw = nn.Conv2d(dim_in, dim_out, 1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.act(self.bn(self.conv_pw(x)))
        return x


class MaxViT(nn.Module):
    """Minimal yet configurable MaxViT architecture for easy ablation."""
    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        n_classes: int = 5,
        embed_dims: tuple = (64, 128, 256, 512),
        depths:     tuple = (2, 2, 5, 2),
        window_size: int = 7,
        **block_kwargs
    ):
        super().__init__()
        assert len(embed_dims) == len(depths)
        self.num_stages = len(embed_dims)

        # Stem (2× downsample)
        self.stem = PatchEmbed(in_chans, embed_dims[0])

        # Stages
        stages = []
        dim_prev = embed_dims[0]
        H = W = img_size // 2  # after stem
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            stage_layers = []
            if i > 0:                       # Downsample between stages
                stage_layers.append(Downsample(dim_prev, dim))
                H //= 2; W //= 2
            # Stack MaxViTBlocks
            blocks = [
                MaxViTBlock(dim, window_size=window_size, **block_kwargs)
                for _ in range(depth)
            ]
            stage_layers.extend(blocks)
            stages.append(nn.Sequential(*stage_layers))
            dim_prev = dim
        self.stages = nn.ModuleList(stages)

        # Classification head
        self.head_norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], n_classes)

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # -----------------------------------------------------------
    def forward(self, x):                 # [B,3,H,W]
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)                  # stays in [B,C,H',W'] domain
        # Global pooling & head
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).mean(-1)  # gap
        x = self.head(self.head_norm(x))
        return x


# ------------------------ quick usage --------------------------
# if __name__ == "__main__":
#     model = MaxViT(
#         img_size=224,
#         num_classes=5,
#         depths=(2,2,5,2),
#         embed_dims=(64,128,256,512),
#         window_size=7,
#         use_mbconv=True,
#         use_block_attn=True,
#         use_grid_attn=True
#     )
#     dummy = torch.randn(2, 3, 224, 224)
#     out = model(dummy)
#     print("output shape:", out.shape)  # => [2, 5]