import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    resnet50, ResNet50_Weights,
)



class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=4 * dim)
        self.shift_size = shift_size
        self.window_size = window_size
        self.input_resolution = input_resolution

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=32):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, (H, W)



class SwinIJICTransformer(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = 32
        self.patch_embed = PatchEmbedding(patch_size=4, embed_dim=self.embed_dim)

       
        self.stage1 = nn.Sequential(
            SwinBlock(self.embed_dim, (56, 56), num_heads=1, window_size=7, shift_size=0)
        )
        self.patch_merge1 = PatchMerging((56, 56), self.embed_dim)

       
        self.stage2 = nn.Sequential(
            SwinBlock(2 * self.embed_dim, (28, 28), num_heads=2, window_size=7, shift_size=0)
        )
        self.patch_merge2 = PatchMerging((28, 28), 2 * self.embed_dim)

        
        self.stage3 = nn.Sequential(
            SwinBlock(4 * self.embed_dim, (14, 14), num_heads=4, window_size=7, shift_size=0)
        )

        self.norm = nn.LayerNorm(4 * self.embed_dim)
        self.head = nn.Linear(4 * self.embed_dim, num_classes)

    def forward(self, x):
        x, (H, W) = self.patch_embed(x)

        x = self.stage1(x)
        x = self.patch_merge1(x)

        x = self.stage2(x)
        x = self.patch_merge2(x)

        x = self.stage3(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x



def get_model(model_name, weights=None):
    model_name = model_name.lower().strip()

    model_mapping = {
        "efficientnet": "efficientnet_b0",
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_v2_s": "efficientnet_v2_s",
        "resnet50": "resnet50",
        "swin_custom": "swin_custom"
    }

    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model '{model_name}'")

    corrected_model_name = model_mapping[model_name]

    if corrected_model_name == "swin_custom":
        print("Super Tiny Manual Swin Transformer loaded.")
        return SwinIJICTransformer(num_classes=5)

    model_class = getattr(models, corrected_model_name, None)

    if model_class is None or not callable(model_class):
        raise ValueError(f" '{corrected_model_name}' model can’t be loaded from torchvision!")

    if weights is None:
        weights = getattr(models, f"{corrected_model_name.upper()}_Weights").DEFAULT

    model = model_class(weights=weights)

    if hasattr(model, "fc"):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)
    elif hasattr(model, "classifier"):
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 5)
    else:
        raise ValueError(f" Cannot modify output layer for model '{corrected_model_name}'")

    return model
