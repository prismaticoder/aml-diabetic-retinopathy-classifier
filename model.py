import torch
import torch.nn as nn
import torchvision.models as models
from models.maxvit import MaxViT
from models.rsgnet import RSGNet
from models.swin_transformer import SwinIJICTransformer, SwinIJICTransformer_AddStage, SwinIJICTransformer_AddExtraBlocks
from models.mlp_mixer import MLPMixer, MLPMixerV2_AddLayer, MLPMixerV2BatchNorm

from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    resnet50, ResNet50_Weights
)

def get_model(name, weights="DEFAULT", n_classes=5, model_variant="baseline"):
    name = name.lower().strip()
    
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
        model = RSGNet(n_classes=n_classes, variant=model_variant)
        
    elif name == "swin_custom":
        model = SwinIJICTransformer(n_classes=n_classes)
    
    elif name == "swin_custom_addstage":
        model = SwinIJICTransformer_AddStage(n_classes=n_classes)
    
    elif name == "swin_custom_addextrablocks":
        model = SwinIJICTransformer_AddExtraBlocks(n_classes=n_classes)
        
    elif name == "mlp_mixer":
        model = MLPMixer(num_classes=n_classes)
        
    elif name == "mlp_mixer_v2_addlayer":
        model = MLPMixerV2_AddLayer(num_classes=n_classes)
        
    elif name == "mlp_mixer_v2_batchnorm":
        model = MLPMixerV2BatchNorm(num_classes=n_classes)
        
    elif name == "maxvit":
        use_mbconv, use_block_attn, use_grid_attn = True, True, True
        if model_variant == 'remove_block_attn':
            use_block_attn = False
        elif model_variant == 'remove_grid_attn':
            use_grid_attn = False
        elif model_variant == 'remove_mbconv':
            use_mbconv = False
        elif model_variant == 'remove_attn':
            use_block_attn, use_grid_attn = False, False
            
        model = MaxViT(n_classes=n_classes, use_mbconv=use_mbconv, use_block_attn=use_block_attn, use_grid_attn=use_grid_attn)
        
    elif name == "maxvit_mse":
        model = MaxViT(n_classes=n_classes, use_mbconv=True, use_block_attn=False, use_grid_attn=True)
        if hasattr(model, 'head'):
            model.head = nn.Linear(model.head.in_features, 1)
    else:
        raise ValueError(f"❌ Unsupported model: {name}")
    return model
