import os
import random
import time
import datetime
import json
import numpy as np
import torch
from rich.console import Console

console = Console()

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gpu_info():
    """Return GPU model and memory if available, else CPU."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        return f"{gpu_name} ({total_mem} MB)"
    return "CPU"

def log_training_start(args, device):
    """Print and save training start metadata."""
    start_time = datetime.datetime.now()
    formatted_start = start_time.strftime("%Y-%m-%d %I:%M:%S %p")
    gpu_info = get_gpu_info()

    console.rule(f"🚀 Training Started at {formatted_start} on {gpu_info}")
    console.rule(f"📦 Model: {args.model.upper()}")
    args._start_time = time.time()
    args._start_time_human = formatted_start
    args._gpu_name = gpu_info

def log_training_end(args, epoch_logs=None, early_stopping_triggered=False, early_stopping_epoch=None):
    """Log training summary to console and save JSON to logs/"""
    end_time = datetime.datetime.now()
    total_time = time.time() - args._start_time
    h, rem = divmod(total_time, 3600)
    m, s = divmod(rem, 60)
    formatted_end = end_time.strftime("%Y-%m-%d %I:%M:%S %p")

    console.rule("[bold green]✅ Training Complete[/bold green]")
    console.print(f"🕒 Finished at: [cyan]{formatted_end}[/cyan]")
    console.print(f"⏱️ Total Duration: [magenta]{int(h)}h {int(m)}m {int(s)}s[/magenta]")

    # Save log to JSON
    log = {
        "user": args.user,
        "student_id": getattr(args, "student_id", "6896375"),
        "model": args.model,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "optimizer": args.optim,
        "lr_scheduler": args.lr_scheduler,
        "start_time_human": args._start_time_human,
        "gpu_name": args._gpu_name,
        "epoch_logs": epoch_logs or [],
        "early_stopping_triggered": early_stopping_triggered,
        "early_stopping_epoch": early_stopping_epoch,
        "end_time_human": formatted_end,
        "total_training_time": f"{int(h):02}:{int(m):02}:{int(s):02}"
    }

    # Ensure log dir exists
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{args.student_id}_{args.model}_train_{args.batch_size}_{args.learning_rate}_{timestamp}.json"
    log_path = os.path.join(args.log_dir, log_filename)

    with open(log_path, "w") as f:
        json.dump(log, f, indent=4)

    console.print(f"🧾 Training log saved to: [blue]{log_path}[/blue]")


class EarlyStopping:
    def __init__(self, patience=3, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            return False

        if val_score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = val_score
            self.counter = 0
        return False
