import os
import json
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from rich.console import Console
from dataset import get_data_loaders
from model import get_model
from utils import set_seed, get_gpu_info, EarlyStopping
from sklearn.metrics import cohen_kappa_score
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

console = Console()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return epoch_loss, epoch_acc, qwk

def main(args):
    set_seed(args.seed)
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.rule(f"[bold cyan]🚀 Training {args.model.upper()}")

    train_loader, val_loader, _ = get_data_loaders(
        csv_root_dir=args.csv_root_dir,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        train_csv=args.train_datacsv,
        val_csv=args.val_datacsv,
        test_csv=args.test_datacsv,
        resized_height=args.resized_img_height,
        resized_width=args.resized_img_weight,
        data_aug=True
    )

    model = get_model(
        args.model.lower(),
        weights=args.weights,
        n_classes=args.n_classes,
        patch_size=args.patch_size  # ✅ Now passed in
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = {
        "adam": torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        "adamw": torch.optim.AdamW(model.parameters(), lr=args.learning_rate),
        "sgd": torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    }[args.optim]

    scheduler = {
        "step": StepLR(optimizer, step_size=5, gamma=0.5),
        "cosine": CosineAnnealingLR(optimizer, T_max=10)
    }[args.lr_scheduler]

    early_stopping = EarlyStopping(patience=5, verbose=True)

    out_dir = f"output/{args.model}_lr{args.learning_rate}_bs{args.batch_size}"
    os.makedirs(out_dir, exist_ok=True)

    log = {
        "user": args.user,
        "model": args.model,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "optimizer": args.optim,
        "lr_scheduler": args.lr_scheduler,
        "patch_size": args.patch_size,
        "start_time_human": time.strftime("%Y-%m-%d %I:%M:%S %p"),
        "gpu_name": get_gpu_info(),
        "epoch_logs": []
    }

    best_val_qwk = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_qwk = validate(model, val_loader, criterion, device)
        scheduler.step()

        log["epoch_logs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_qwk": val_qwk
        })

        console.print(f"[bold yellow]Epoch {epoch}[/] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | QWK: {val_qwk:.4f}")
        
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            torch.save(model.state_dict(), os.path.join(out_dir, f"{args.model}_epoch{epoch}.pth"))

        if early_stopping(val_qwk):
            log["early_stopping_triggered"] = True
            log["early_stopping_epoch"] = epoch
            break

    log["end_time_human"] = time.strftime("%Y-%m-%d %I:%M:%S %p")
    log["total_training_time"] = f"{(time.time() - start_time) / 60:.0f}m"

    log_path = f"logs/{args.user}_{args.model}_train_{args.batch_size}_{args.learning_rate}.json"
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    console.rule("[bold green]✅ Training Complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--user', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--csv_root_dir', required=True)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--train_datacsv', required=True)
    parser.add_argument('--val_datacsv', required=True)
    parser.add_argument('--test_datacsv', required=True)
    parser.add_argument('--weights', default="DEFAULT")
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--resized_img_weight', type=int, default=224)
    parser.add_argument('--resized_img_height', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--lr_scheduler', default='cosine', choices=['step', 'cosine'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', default="logs")
    parser.add_argument('--patch_size', type=int, default=16)  # ✅ Added

    args = parser.parse_args()
    args.lr_scheduler = args.lr_scheduler.lower()
    main(args)
