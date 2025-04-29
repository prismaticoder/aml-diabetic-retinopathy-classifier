import argparse, os, json, time
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
from dataset import get_data_loaders
from model import get_model

console = Console()
STUDENT_ID = "6896375"

def format_hms(seconds):
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_gpu_memory():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    return None

def train_epoch(model, loader, criterion, optimizer, device, progress_bar, task_id):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        acc = 100 * correct / total
        progress_bar.update(task_id, advance=1, description=f"[green]Loss: {loss.item():.4f}, Acc: {acc:.2f}%")

    acc = 100 * correct / total
    return total_loss, acc

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

def save_checkpoint(model, model_dir, model_name, epoch):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{model_name}_epoch{epoch}.pth")
    torch.save({"model_state_dict": model.state_dict()}, path)
    console.print(f"💾 Saved checkpoint: {path}", style="bold green")

def plot_metrics(train_loss, val_loss, train_acc, val_acc, plot_dir):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    os.makedirs(plot_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "metrics.png"))
    plt.close()

def save_log(log, student_name, model_name, run_type, batch_size, lr):
    student_name = student_name.replace(" ", "_")
    os.makedirs("logs", exist_ok=True)
    path = f"logs/{student_name}_ID{STUDENT_ID}_{model_name}_{run_type}_{batch_size}_{lr}.json"
    with open(path, "w") as f:
        json.dump(log, f, indent=4)
    console.print(f"📄 Log saved to: {path}", style="cyan")

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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    start_time = time.time()
    start_clock = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

    console.print(f"🚀 [bold blue]Training Started[/bold blue] at [green]{start_clock}[/green] on [yellow]{gpu_name}[/yellow]")

    model = get_model(args.model.lower(), weights=args.weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = get_data_loaders(args.csv_root_dir, args.img_dir, args.batch_size)

    model_dir = f"output/{args.model}_lr{args.learning_rate}_bs{args.batch_size}"
    plot_dir = os.path.join(args.log_dir, "plots")

    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    log_data = {
        "user": args.user,
        "student_id": STUDENT_ID,
        "model": args.model,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "optimizer": args.optim,
        "lr_scheduler": args.lr_scheduler,
        "start_time_human": start_clock,
        "gpu_name": gpu_name,
        "epoch_logs": [],
        "early_stopping_triggered": False,
        "early_stopping_epoch": None
    }

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress_bar:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            task_id = progress_bar.add_task(f"Epoch {epoch}/{args.epochs}", total=len(train_loader))
            train_epoch_loss, train_epoch_acc = train_epoch(model, train_loader, criterion, optimizer, device, progress_bar, task_id)
            val_epoch_loss, val_epoch_acc, val_epoch_qwk = validate(model, val_loader, criterion, device)
            epoch_duration = time.time() - epoch_start
            torch.cuda.reset_peak_memory_stats()
            gpu_mem = get_gpu_memory()

            log_data["epoch_logs"].append({
                "epoch": epoch,
                "train_loss": train_epoch_loss,
                "train_acc": train_epoch_acc,
                "val_loss": val_epoch_loss,
                "val_acc": val_epoch_acc,
                "val_qwk": val_epoch_qwk,
                "epoch_duration": format_hms(epoch_duration),
                "gpu_memory_mb": gpu_mem
            })

            console.print(f"\n✅ [bold green]Epoch {epoch} Summary[/bold green]: "
                          f"Train Loss = {train_epoch_loss:.4f}, Train Acc = {train_epoch_acc:.2f}% | "
                          f"Val Loss = {val_epoch_loss:.4f}, Val Acc = {val_epoch_acc:.2f}%, QWK = {val_epoch_qwk:.4f} | "
                          f"GPU Mem = {gpu_mem} MB")
            console.print(f"🕒 Time for Epoch {epoch}: {format_hms(epoch_duration)}", style="dim")

            if args.save_model_every_epoch and epoch % 2 == 1:
                save_checkpoint(model, model_dir, args.model, epoch)

            # Early stopping
            if args.early_stopping:
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        console.print(f"\n🛑 [bold red]Early stopping triggered at epoch {epoch}[/bold red]")
                        log_data["early_stopping_triggered"] = True
                        log_data["early_stopping_epoch"] = epoch
                        break

    total_time = time.time() - start_time
    end_clock = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    log_data["end_time_human"] = end_clock
    log_data["total_training_time"] = format_hms(total_time)

    console.rule(f"[bold green]✅ Training Completed[/bold green] — "
                 f"[blue]{end_clock}[/blue] — Took {format_hms(total_time)} on [yellow]{gpu_name}[/yellow]")

    plot_metrics(train_loss, val_loss, train_acc, val_acc, plot_dir)
    save_log(log_data, args.user, args.model, "train", args.batch_size, args.learning_rate)
