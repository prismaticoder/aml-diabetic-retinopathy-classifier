import os, json, argparse, torch
import torch.nn as nn
from dataset import get_data_loaders
from model import get_model
from train import validate, get_loss_function
from datetime import datetime

STUDENT_ID = "6891120"

def format_hms(seconds):
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def reconstruct_log(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variant_map = {
        "rsgnet_removed": "remove_layer",
        "rsgnet_added": "added_layer",
        "rsgnet_avg_best": "avgpool"
    }
    args.model_variant = variant_map.get(args.model.lower(), "baseline")

    model = get_model(args.model, weights=args.weights, n_classes=args.n_classes, model_variant=args.model_variant).to(device)
    _, val_loader = get_data_loaders(args.csv_root_dir, args.img_dir, args.batch_size, profile=args.augmentation_profile or "none")
    criterion = get_loss_function(args.loss)

    checkpoint_dir = f"output/{args.model}_opt{args.optim}_lr{args.learning_rate}_bs{args.batch_size}_loss{args.loss}_aug{args.augmentation_profile or 'none'}"
    new_log = {
        "user": args.user,
        "student_id": STUDENT_ID,
        "model": args.model,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "optimizer": args.optim,
        "loss_function": args.loss,
        "lr_scheduler": args.lr_scheduler,
        "augmentation_profile": args.augmentation_profile,
        "model_variant": args.model_variant,
        "epoch_logs": [],
        "early_stopping_triggered": False,
        "early_stopping_epoch": None,
        "start_time_human": datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    }

    print(f"🔁 Searching for checkpoints in: {checkpoint_dir}")
    for filename in sorted(os.listdir(checkpoint_dir)):
        if filename.endswith(".pth") and "epoch" in filename:
            epoch = int(filename.split("epoch")[-1].split(".")[0])
            checkpoint = torch.load(os.path.join(checkpoint_dir, filename), map_location=device)

            model.load_state_dict(checkpoint["model_state_dict"])
            val_loss, val_acc, val_qwk = validate(model, val_loader, criterion, device)

            new_log["epoch_logs"].append({
                "epoch": epoch,
                "train_loss": 0,
                "train_acc": 0,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_qwk": val_qwk,
                "epoch_duration": "reconstructed",
                "gpu_memory_mb": None
            })
            print(f"✅ Reconstructed epoch {epoch} — Val QWK: {val_qwk:.4f}")

    new_log["end_time_human"] = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    new_log["total_training_time"] = "reconstructed"

    log_path = f"logs/{args.user.replace(' ', '_')}_ID{STUDENT_ID}_{args.model}_train_{args.batch_size}_{args.learning_rate}.json"
    with open(log_path, "w") as f:
        json.dump(new_log, f, indent=4)

    print(f"\n📋 Log reconstruction completed: {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--csv_root_dir", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--loss", type=str, default="crossentropy")
    parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--augmentation_profile", type=str, default="none")
    args = parser.parse_args()
    reconstruct_log(args)
