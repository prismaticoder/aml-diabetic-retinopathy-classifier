import argparse, os, json, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
from dataset import get_data_loaders
from model import get_model

from loss_kappa import WeightedKappaLoss

console = Console()
STUDENT_ID = "6891120"

def format_hms(seconds):
    h, m, s = int(seconds) // 3600, (int(seconds) % 3600) // 60, int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_gpu_memory():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    return None

def get_loss_function(name):
    name = name.lower()
    if name == "crossentropy":
        return nn.CrossEntropyLoss()
    elif name == "focal":
        from torch.nn import functional as F
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
            def forward(self, inputs, targets):
                ce = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce)
                return (self.alpha * (1 - pt) ** self.gamma * ce).mean()
        return FocalLoss()
    elif name == "labelsmoothing":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    elif name == 'kappa':
        return WeightedKappaLoss(num_classes=5)
    elif name == "mse":
        return nn.MSELoss()
    raise ValueError(f"❌ Unsupported loss: {name}")

def get_optimizer(name, params, lr):
    name = name.lower()
    if name == "adam": return optim.Adam(params, lr=lr)
    elif name == "sgd": return optim.SGD(params, lr=lr, momentum=0.9)
    elif name == "adamw": return optim.AdamW(params, lr=lr, weight_decay=0.03)
    raise ValueError(f"❌ Unsupported optimizer: {name}")

def train_epoch(model, loader, criterion, optimizer, scaler, device, progress, task_id, scheduler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            if (isinstance(criterion, nn.MSELoss)):
                prepared_loss_labels = labels.float().unsqueeze(1)
            else:
                prepared_loss_labels = labels
            loss = criterion(outputs, prepared_loss_labels)
        scaler.scale(loss).backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        # scheduler.step()
        
        total_loss += loss.item()
        
        correct += (outputs.argmax(1) == labels).sum().item()
        if isinstance(criterion, nn.MSELoss):
            preds_batch = outputs.round().long().squeeze(1)   # regression → class id
            preds_batch = preds_batch.clamp(0, 4)
        else:
            preds_batch = outputs.argmax(1)
        correct += (preds_batch == labels).sum().item()
        total += labels.size(0)
        acc = 100 * correct / total
        progress.update(task_id, advance=1, description=f"[green]Loss: {loss.item():.4f}, Acc: {acc:.2f}%")
    scheduler.step()
    return total_loss, acc

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(criterion, nn.MSELoss):
                prepared_loss_labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, prepared_loss_labels)
                pred = outputs.round().long().squeeze(1)
                pred = pred.clamp(0, 4)
            else:
                loss = criterion(outputs, labels)
                pred = outputs.argmax(1)
                
            total_loss += loss.item()
            pred = outputs.argmax(1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    qwk = cohen_kappa_score(targets, preds, weights="quadratic")
    return total_loss, acc, qwk

def plot_metrics(train_loss, val_loss, train_acc, val_acc, plot_dir):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.plot(train_loss); plt.plot(val_loss); plt.title("Loss"); plt.xlabel("Epoch"); plt.legend(["Train", "Val"])
    plt.subplot(1, 2, 2); plt.plot(train_acc); plt.plot(val_acc); plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend(["Train", "Val"])
    os.makedirs(plot_dir, exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "metrics.png")); plt.close()

def save_log(log, student_name, model_name, run_type, batch_size, lr):
    student_name = student_name.replace(" ", "_")
    path = f"logs/{student_name}_ID{STUDENT_ID}_{model_name}_{run_type}_{batch_size}_{lr}.json"
    os.makedirs("logs", exist_ok=True)
    with open(path, "w") as f: json.dump(log, f, indent=4)
    console.print(f"📋 Log saved to: {path}", style="cyan")

def save_checkpoint(model, out_dir, model_name, epoch):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}_epoch{epoch}.pth")
    torch.save({"model_state_dict": model.state_dict()}, path)
    console.print(f"💾 Saved checkpoint: {path}", style="green")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--csv_root_dir", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--loss", type=str, default="crossentropy")
    parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--brightness", type=float, default=0.2)
    parser.add_argument("--contrast", type=float, default=0.2)
    parser.add_argument("--saturation", type=float, default=0.2)
    parser.add_argument("--hue", type=float, default=0.2)
    parser.add_argument("--resized_img_weight", type=int, default=224)
    parser.add_argument("--resized_img_height", type=int, default=224)
    parser.add_argument("--train_datacsv", required=True)
    parser.add_argument("--val_datacsv", required=True)
    parser.add_argument("--test_datacsv", required=True)
    parser.add_argument("--saved_checkpoint_path", required=True)
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--save_model_every_epoch", action="store_true")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--augmentation_profile", type=str, default=None)
    parser.add_argument("--model_variant", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_variant is None:
        variant_map = {
            "rsgnet_removed": "remove_layer",
            "rsgnet_added": "added_layer",
            "rsgnet_avg_best": "avgpool"
        }
        args.model_variant = variant_map.get(args.model.lower(), "baseline")

    model = get_model(args.model.lower(), weights=args.weights, n_classes=args.n_classes).to(device)
    optimizer = get_optimizer(args.optim, model.parameters(), args.learning_rate)
    criterion = get_loss_function(args.loss)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)
    scaler = GradScaler()

    train_loader, val_loader = get_data_loaders(args.csv_root_dir, args.img_dir, args.batch_size,
                                                num_workers=args.num_workers)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=args.learning_rate,
    #     epochs=args.epochs,
    #     steps_per_epoch=len(train_loader),
    #     anneal_strategy='cos',
    #     div_factor=25,
    #     final_div_factor=1e4,
    #     pct_start=0.2,
    # )
    model_dir = f"output/{args.model}_opt{args.optim}_lr{args.learning_rate}_bs{args.batch_size}_loss{args.loss}_aug{args.augmentation_profile or 'none'}"
    plot_dir = os.path.join(args.log_dir, "plots")

    log_data = {
        "user": args.user, "student_id": STUDENT_ID, "model": args.model,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate, "epochs": args.epochs,
        "optimizer": args.optim, "loss_function": args.loss, "lr_scheduler": args.lr_scheduler,
        "augmentation_profile": args.augmentation_profile, "model_variant": args.model_variant,
        "epoch_logs": [], "early_stopping_triggered": False, "early_stopping_epoch": None
    }

    start_time = time.time()
    start_clock = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    console.print(f"🚀 Training started at [green]{start_clock}[/green] on [yellow]{device}[/yellow]")

    best_val_loss, patience_counter, patience = float("inf"), 0, 5
    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    with Progress(SpinnerColumn(), TextColumn("[blue]{task.description}"), BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                  TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            task_id = progress.add_task(f"Epoch {epoch}/{args.epochs}", total=len(train_loader))
            train_epoch_loss, train_epoch_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, progress, task_id,scheduler)
            val_epoch_loss, val_epoch_acc, val_epoch_qwk = validate(model, val_loader, criterion, device)

            train_loss.append(train_epoch_loss)
            val_loss.append(val_epoch_loss)
            train_acc.append(train_epoch_acc)
            val_acc.append(val_epoch_acc)

            duration = time.time() - epoch_start
            gpu_mem = get_gpu_memory()

            log_data["epoch_logs"].append({
                "epoch": epoch, "train_loss": train_epoch_loss, "train_acc": train_epoch_acc,
                "val_loss": val_epoch_loss, "val_acc": val_epoch_acc, "val_qwk": val_epoch_qwk,
                "epoch_duration": format_hms(duration), "gpu_memory_mb": gpu_mem
            })

            console.print(f"\n✅ [green]Epoch {epoch} Summary[/green]: "
                          f"Train Loss = {train_epoch_loss:.4f}, Acc = {train_epoch_acc:.2f}% | "
                          f"Val Loss = {val_epoch_loss:.4f}, Acc = {val_epoch_acc:.2f}%, QWK = {val_epoch_qwk:.4f}")

            if args.save_model_every_epoch and epoch % 2 == 1:
                save_checkpoint(model, model_dir, args.model, epoch)

            if args.early_stopping:
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        console.print(f"⛔ [red]Early stopping at epoch {epoch}[/red]")
                        log_data["early_stopping_triggered"] = True
                        log_data["early_stopping_epoch"] = epoch
                        break

    end_time = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    log_data["start_time_human"] = start_clock
    log_data["end_time_human"] = end_time
    log_data["total_training_time"] = format_hms(time.time() - start_time)

    console.rule(f"[bold green]✅ Training Complete[/bold green] at [blue]{end_time}[/blue]")
    plot_metrics(train_loss, val_loss, train_acc, val_acc, plot_dir)
    save_log(log_data, args.user, args.model, "train", args.batch_size, args.learning_rate)