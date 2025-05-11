import argparse
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from model import get_model
from utils import set_seed
from rich.console import Console

console = Console()

def preprocess_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    return image_tensor, image

def generate_gradcam(model, input_tensor, target_class, target_layer):
    grad = None
    fmap = None

    def forward_hook(module, input, output):
        nonlocal fmap
        fmap = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal grad
        grad = grad_output[0]

    # Register hooks
    hook_fwd = target_layer.register_forward_hook(forward_hook)
    hook_bwd = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    output = model(input_tensor)
    class_idx = target_class if target_class is not None else torch.argmax(output)
    console.print(f"🔍 Target class index: [bold yellow]{class_idx.item() if isinstance(class_idx, torch.Tensor) else class_idx}[/bold yellow]")

    # Backward
    model.zero_grad()
    output[0, class_idx].backward()

    console.print(f"📐 grad shape: {grad.shape}")

    # Compute weights
    if grad.dim() == 4:
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        gradcam = torch.sum(weights * fmap, dim=1, keepdim=True)  # [B, 1, H, W]
    elif grad.dim() == 3:
        weights = torch.mean(grad, dim=1, keepdim=True)  # [B, 1, D]
        gradcam = torch.sum(weights * fmap, dim=2, keepdim=True)  # [B, D, 1]
        gradcam = gradcam.permute(0, 2, 1).view(1, 1, int(fmap.size(1) ** 0.5), -1)  # to [B, 1, H, W]
    else:
        raise ValueError(f"❌ Unexpected gradient shape: {grad.shape}")

    # Normalize
    gradcam = F.relu(gradcam)
    gradcam = F.interpolate(gradcam, size=(224, 224), mode="bilinear", align_corners=False)
    gradcam = gradcam.squeeze().detach().cpu().numpy()

    # Remove hooks
    hook_fwd.remove()
    hook_bwd.remove()

    return gradcam

def visualize_gradcam(original_image, heatmap, output_path):
    heatmap = np.uint8(255 * heatmap / heatmap.max())
    heatmap = Image.fromarray(heatmap).resize(original_image.size)
    heatmap = np.array(heatmap)
    heatmap_color = plt.get_cmap('jet')(heatmap / 255.0)[:, :, :3]  # RGB only
    heatmap_color = (heatmap_color * 255).astype(np.uint8)
    overlay = Image.blend(original_image, Image.fromarray(heatmap_color), alpha=0.5)

    overlay.save(output_path)
    console.print(f"🖼️ Grad-CAM image saved to: [green]{output_path}[/green]")

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model, weights=args.weights, n_classes=args.n_classes).to(device)
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)

    console.print(model)

    input_tensor, original_image = preprocess_image(args.image_path, args.image_size)
    input_tensor = input_tensor.to(device)

    # Identify last mixer block as target layer
    if hasattr(model, 'mixer_layers'):
        target_layer = model.mixer_layers[-1]
    else:
        raise ValueError("❌ Could not locate a valid target layer for Grad-CAM.")

    gradcam_map = generate_gradcam(model, input_tensor, args.target_class, target_layer)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "gradcam_result.png")
    visualize_gradcam(original_image, gradcam_map, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="Model name (e.g., mlp_mixer_v2_batchnorm)")
    parser.add_argument('--weights', default="DEFAULT")
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--output_dir', default="gradcam_outputs")
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--target_class', type=int, default=None, help="Specify class index (optional)")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
