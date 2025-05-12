
import sys

import argparse, os, json, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
from dataset import get_data_loaders
from model import get_model

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    qwk = cohen_kappa_score(targets, preds, weights="quadratic")
    return total_loss, acc, qwk

def extract_epoch(file_name):
    match = re.search(r'epoch(\d+)', file_name)
    return int(match.group(1)) if match else -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--optim", required=True)
    parser.add_argument("--loss", required=True)
    parser.add_argument("--augmentation", required=True)
    parser.add_argument("--weights", default="DEFAULT")
    parser.add_argument("--csv_root_dir", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--val_datacsv", required=True)
    parser.add_argument("--log_output_path", required=True)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variant_map = {
        "rsgnet_removed": "remove_layer",
        "rsgnet_added": "added_layer",
        "rsgnet_avg_best": "avgpool"
    }
    model_variant = variant_map.get(args.model.lower(), "baseline")

    _, val_loader = get_data_loaders(
        args.csv_root_dir, args.img_dir, batch_size=args.batch_size,
        profile=args.augmentation, only_test=False
    )

    model = get_model(args.model, weights=args.weights, model_variant=model_variant).to(device)
    criterion = nn.CrossEntropyLoss()

    log_data = {
        "user": args.user,
        "student_id": "6891120",
        "model": args.model,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optim,
        "loss_function": args.loss,
        "augmentation_profile": args.augmentation,
        "model_variant": model_variant,
        "start_time_human": datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"),
        "epoch_logs": [],
        "early_stopping_triggered": True,
        "early_stopping_epoch": None
    }

    ckpts = sorted(
        [f for f in os.listdir(args.model_dir) if f.endswith(".pth")],
        key=extract_epoch
    )

    for ckpt_file in ckpts:
        epoch = extract_epoch(ckpt_file)
        path = os.path.join(args.model_dir, ckpt_file)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        val_loss, val_acc, val_qwk = validate(model, val_loader, criterion, device)

        log_data["epoch_logs"].append({
            "epoch": epoch,
            "train_loss": None,
            "train_acc": None,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_qwk": val_qwk,
            "epoch_duration": "N/A",
            "gpu_memory_mb": None
        })

    with open(args.log_output_path, "w") as f:
        json.dump(log_data, f, indent=4)
    print(f"✅ Reconstructed and saved log: {args.log_output_path}")
