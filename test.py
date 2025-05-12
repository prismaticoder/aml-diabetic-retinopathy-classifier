import argparse, os, json, time, torch
import torch.nn as nn
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score
from dataset import get_data_loaders
from model import get_model

console = Console()
STUDENT_ID = "6904186"

def format_hms(seconds):
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_gpu_info():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    return "CPU"

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, targets, filenames = [], [], []

    with torch.no_grad(), Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Evaluating"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Running...", total=len(loader))

        for batch in loader:
            if len(batch) == 2:
                images, labels = batch
                fnames = ["unknown"] * len(labels)
            elif len(batch) == 3:
                images, labels, fnames = batch
            else:
                raise ValueError("Unexpected number of elements returned from test loader.")

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            filenames.extend(fnames)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            progress.update(task_id, advance=1)

    acc = 100 * correct / total
    qwk = cohen_kappa_score(targets, preds, weights="quadratic")
    return total_loss, acc, qwk, preds, targets, filenames

def save_log(log, student_name, model_name, run_type, batch_size, lr):
    student_name = student_name.replace(" ", "_")
    log_file_name = f"logs/{student_name}_ID{STUDENT_ID}_{model_name}_{run_type}_{batch_size}_{lr}.json"
    os.makedirs("logs", exist_ok=True)
    with open(log_file_name, "w") as f:
        json.dump(log, f, indent=4)
    console.print(f"📄 Log saved to {log_file_name}", style="bold cyan")

def save_outputs(model_name, lr, bs, preds, targets, filenames, cm_title,
                 user, student_id, start_clock, end_clock, gpu_name, total_eval_time):
    save_dir = f"output/Test_Results/{model_name}_lr{lr}_bs{bs}"
    os.makedirs(save_dir, exist_ok=True)

    # Save predictions
    df = pd.DataFrame({"filename": filenames, "actual": targets, "predicted": preds})
    df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    # Confusion matrix
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(cm_title)
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # Test summary
    precision = precision_score(targets, preds, average='weighted', zero_division=0)
    recall = recall_score(targets, preds, average='weighted', zero_division=0)
    f1 = f1_score(targets, preds, average='weighted', zero_division=0)
    try:
        roc = roc_auc_score(targets, preds, multi_class='ovr')
    except:
        roc = 0.0
    qwk = cohen_kappa_score(targets, preds, weights="quadratic")

    summary = f"""
👤 User: {user}
🎓 Student ID: {student_id}
🧠 Model: {model_name}
📦 Batch Size: {bs}
🚀 Learning Rate: {lr}
🕐 Evaluation Start: {start_clock}
🏁 Evaluation End: {end_clock}
💻 Device: {gpu_name}
⏱️ Evaluation Duration: {total_eval_time}

• Precision: {precision:.4f}
• Recall:    {recall:.4f}
• F1 Score:  {f1:.4f}
• ROC AUC:   {roc:.4f}
• QWK:       {qwk:.4f}
""".strip()

    with open(os.path.join(save_dir, "test_summary.txt"), "w") as f:
        f.write(summary)

    console.print(f"📝 Test summary saved to: {save_dir}/test_summary.txt", style="bold cyan")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--csv_root_dir", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resized_img_weight", type=int, default=224)
    parser.add_argument("--resized_img_height", type=int, default=224)
    parser.add_argument("--train_datacsv", required=True)
    parser.add_argument("--val_datacsv", required=True)
    parser.add_argument("--test_datacsv", required=True)
    parser.add_argument("--saved_checkpoint_path", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--confusion_matrix_title", type=str, default="Confusion Matrix")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = get_gpu_info()
    start_clock = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    console.print(f"\n🚀 Evaluation started at [bold yellow]{start_clock}[/bold yellow] on [bold green]{gpu_name}[/bold green]")

    start_time = time.time()
    model = get_model(args.model.lower(), weights=args.weights).to(device)
    checkpoint = torch.load(args.saved_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.CrossEntropyLoss()
    from test_dataset import get_test_loader
    test_loader = get_test_loader(args.test_datacsv, args.img_dir, args.batch_size)

    total_loss, accuracy, qwk, preds, targets, filenames = validate(model, test_loader, criterion, device)

    end_clock = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    total_eval_time = format_hms(time.time() - start_time)

    console.print(f"\n📈 [bold green]Evaluation Results:[/bold green] "
                  f"Loss = {total_loss:.4f}, Accuracy = {accuracy:.2f}%, QWK = {qwk:.4f}")
    console.print(f"🏁 Evaluation ended at [bold yellow]{end_clock}[/bold yellow]")
    console.print(f"⏱️ Total Time: [bold]{total_eval_time}[/bold] | Device: [bold green]{gpu_name}[/bold green]")

    log_data = {
        "user": args.user,
        "student_id": STUDENT_ID,
        "model": args.model,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "run_type": "test",
        "start_time": start_clock,
        "end_time": end_clock,
        "gpu": gpu_name,
        "eval_duration": total_eval_time,
        "loss": total_loss,
        "accuracy": accuracy,
        "qwk": qwk
    }

    save_log(log_data, args.user, args.model, "test", args.batch_size, args.learning_rate)
    save_outputs(args.model, args.learning_rate, args.batch_size, preds, targets, filenames,
                 args.confusion_matrix_title, args.user, STUDENT_ID, start_clock, end_clock, gpu_name, total_eval_time)
