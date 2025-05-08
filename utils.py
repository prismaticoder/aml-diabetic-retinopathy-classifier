import os
import random
import time
import datetime
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

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
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        return f"{gpu_name} ({total_mem} MB)"
    return "CPU"

def log_training_start(args, device):
    start_time = datetime.datetime.now()
    formatted_time = start_time.strftime("%Y-%m-%d %I:%M:%S %p")
    gpu_info = get_gpu_info()

    console.rule(f"🧪 Training Started at {formatted_time} on {gpu_info}")
    console.rule(f"🧠 Training {args.model.upper()}")
    args._start_time = time.time()  # Store for later

def log_training_end(args):
    end_time = datetime.datetime.now()
    duration = time.time() - args._start_time
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)

    formatted_end = end_time.strftime("%Y-%m-%d %I:%M:%S %p")
    console.rule("✅ Training Complete")

    console.print(f"🕒 Finished at: [cyan]{formatted_end}[/cyan]")
    console.print(f"⏱️ Total Duration: [magenta]{int(hours)}h {int(minutes)}m {int(seconds)}s[/magenta]")
