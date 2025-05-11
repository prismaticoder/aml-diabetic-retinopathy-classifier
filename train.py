import argparse, os, time, datetime, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score
from torch.cuda.amp import autocast, GradScaler
from rich.console import Console
from dataset import get_data_loaders
from model import get_model

console = Console()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gpu_info():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        return torch.cuda.get_device_name(device)
    return "CPU"

def log_training_start(args, device):
    start = datetime.datetime.now()
    args._start_time = time.time()
    args._start_time_human = start.strftime("%Y-%m-%d %I:%M:%S %p")
    args._device = device
    console.rule(f"🚀 Training {args.model.upper()}")
    console.print(f"🕒 Started at: {args._start_time_human}")
    console.print(f"💻 Device: {device}")
    console.print(f"📦 Batch Size: {args.batch_size}, LR: {args.learning_rate}, Epochs: {args.epochs}")

def log_training_end(args):
    end = datetime.datetime.now()
    duration = time.time() - args._start_time
    h, rem = divmod(duration, 3600)
    m, s = divmod(rem, 60)
    console.rule("✅ Training Complete")
    console.print(f"⏱️ Duration: {int(h)}h {int(m)}m {int(s)}s")

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_training_start(args, get_gpu_info())

    train_loader, val_loader, _ = get_data_loaders(
        csv_root_dir=args.csv_root_dir,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        train_csv=args.train_datacsv,
        val_csv=args.val_datacsv,
        test_csv=args.test_datacsv,
        resized_height=args.resized_img_height,
        resized_width=args.resized_img_weight,
        data_aug=args.data_augmentation,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        hue=args.hue,
        num_workers=4
    )

    class_labels = pd.read_csv(os.path.join(args.csv_root_dir, args.train_datacsv)).iloc[:, 1].values
    class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = get_model(
        args.model.lower(),
        weights=args.weights,
        n_classes=args.n_classes,
        image_size=args.resized_img_height,
        patch_size=16
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    log = {
        "user": args.user,
        "student_id": args.student_id,
        "model": args.model,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "optimizer": args.optim,
        "lr_scheduler": args.lr_scheduler,
        "start_time_human": args._start_time_human,
        "gpu_name": get_gpu_info(),
        "epoch_logs": []
    }

    out_dir = f"output/{args.model}_lr{args.learning_rate}_bs{args.batch_size}"
    os.makedirs(out_dir, exist_ok=True)

    best_qwk = -1
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / total

        model.eval()
        val_total, val_correct = 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_acc = val_correct / val_total
        val_loss = nn.CrossEntropyLoss()(outputs, labels).item()
        val_qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        epoch_time = round(time.time() - start_time, 2)

        console.print(f"[bold]Epoch {epoch}[/bold] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | QWK: {val_qwk:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            early_stop_counter = 0
            best_model_path = f"{out_dir}/{args.model}_epoch{epoch}.pth"
            torch.save({"model_state_dict": model.state_dict()}, best_model_path)
        else:
            early_stop_counter += 1

        if args.save_every and epoch % args.save_every == 0:
            torch.save({"model_state_dict": model.state_dict()}, f"{out_dir}/{args.model}_epoch{epoch}.pth")

        log["epoch_logs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": round(train_acc * 100, 4),
            "val_loss": val_loss,
            "val_acc": round(val_acc * 100, 4),
            "val_qwk": round(val_qwk, 6),
            "epoch_duration": f"{epoch_time}s"
        })

        if args.early_stopping and early_stop_counter >= 3:
            console.print("🛑 Early stopping triggered")
            log["early_stopping_triggered"] = True
            log["early_stopping_epoch"] = epoch
            break

        scheduler.step()

    log["end_time_human"] = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    log["total_training_time"] = f"{int((time.time() - args._start_time) // 60)}m"
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{args.student_id}_{args.model}_train_{args.batch_size}_{args.learning_rate}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=4)

    log_training_end(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--student_id", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--csv_root_dir", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--train_datacsv", required=True)
    parser.add_argument("--val_datacsv", required=True)
    parser.add_argument("--test_datacsv", required=True)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--resized_img_weight", type=int, default=224)
    parser.add_argument("--resized_img_height", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--lr_scheduler", default="CosineAnnealingLR")
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--contrast", type=float, default=0.0)
    parser.add_argument("--saturation", type=float, default=0.0)
    parser.add_argument("--hue", type=float, default=0.0)

    args = parser.parse_args()
    main(args)
