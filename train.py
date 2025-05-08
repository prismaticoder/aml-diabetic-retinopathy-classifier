import argparse, os, random, time, torch, numpy as np
import pandas as pd
from torch import nn, optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score
from rich.console import Console
from dataset import get_data_loaders
from model import get_model
from utils import set_seed, get_gpu_info, log_training_start, log_training_end

console = Console()

def train():
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Load Data ───────────────────────────────────────────
    train_loader, val_loader = get_data_loaders(
        csv_root_dir=args.csv_root_dir,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        num_workers=2,
        data_aug=args.data_augmentation,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        hue=args.hue
    )

    # ─── Compute Class Weights ───────────────────────────────
    train_csv = pd.read_csv(os.path.join(args.csv_root_dir, "train.csv"))
    class_labels = train_csv.iloc[:, 1].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(class_labels),
        y=class_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # ─── Model and Optimizer ─────────────────────────────────
    model = get_model(args.model.lower(), weights=args.weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer")

    if args.lr_scheduler == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    # ─── Logging ─────────────────────────────────────────────
    log_training_start(args, device)

    best_val_qwk = -1
    early_stop_counter = 0
    os.makedirs(f"output/{args.model}_lr{args.learning_rate}_bs{args.batch_size}", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total

        # ─── Validation ──────────────────────────────────────
        model.eval()
        correct_val, total_val = 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        val_acc = correct_val / total_val
        val_qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

        # ─── Epoch Summary ───────────────────────────────────
        console.print(f"📅 Epoch {epoch}/{args.epochs} | 🏋️ Train Acc: {train_acc:.4f} | 📈 Val Acc: {val_acc:.4f} | QWK: {val_qwk:.4f}")

        # ─── Checkpointing ──────────────────────────────────
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            early_stop_counter = 0
            best_model_path = f"output/{args.model}_lr{args.learning_rate}_bs{args.batch_size}/{args.model}_epoch{epoch}.pth"
            torch.save({'model_state_dict': model.state_dict()}, best_model_path)
            console.print(f"💾 Saved best model at epoch {epoch}")
        else:
            early_stop_counter += 1

        if args.save_every and epoch % args.save_every == 0:
            intermediate_path = f"output/{args.model}_lr{args.learning_rate}_bs{args.batch_size}/{args.model}_epoch{epoch}.pth"
            torch.save({'model_state_dict': model.state_dict()}, intermediate_path)

        if args.early_stopping and early_stop_counter >= 3:
            console.print("🛑 Early stopping at epoch", epoch)
            break

        if scheduler:
            scheduler.step()

    log_training_end(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--csv_root_dir", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--resized_img_weight", type=int, default=224)
    parser.add_argument("--resized_img_height", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--lr_scheduler", default="CosineAnnealingLR")

    # New arguments for data augmentation
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--contrast", type=float, default=0.0)
    parser.add_argument("--saturation", type=float, default=0.0)
    parser.add_argument("--hue", type=float, default=0.0)
    parser.add_argument("--data_augmentation", action="store_true")

    args = parser.parse_args()
    train()
