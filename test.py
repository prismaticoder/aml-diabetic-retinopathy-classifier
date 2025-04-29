import argparse, os, json, time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, precision_score, recall_score, f1_score, roc_auc_score
from rich.console import Console
from dataset import get_data_loaders
from model import get_model

console = Console()
SEVERITY_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    console.print(f"📊 Confusion matrix saved to: [green]{save_path}[/green]")

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    return all_preds, all_targets

def format_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    try:
        auc = roc_auc_score(
            pd.get_dummies(y_true, drop_first=False),
            pd.get_dummies(y_pred, drop_first=False),
            average="weighted",
            multi_class="ovr"
        )
    except Exception:
        auc = None

    return {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "qwk": round(qwk, 4),
        "roc_auc": round(auc, 4) if auc else "N/A"
    }

def save_predictions(y_true, y_pred, save_path):
    df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    df.to_csv(save_path, index=False)
    console.print(f"📄 Predictions saved to: [cyan]{save_path}[/cyan]")

def save_summary(metrics, save_path, args):
    with open(save_path, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"User: {args.user}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write("\n=== Evaluation Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k.upper()}: {v}\n")
    console.print(f"📘 Summary saved to: [magenta]{save_path}[/magenta]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--csv_root_dir", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--train_datacsv", required=True)
    parser.add_argument("--val_datacsv", required=True)
    parser.add_argument("--test_datacsv", required=True)
    parser.add_argument("--saved_checkpoint_path", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--confusion_matrix_title", type=str, default="Confusion Matrix")
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--resized_img_weight", type=int, default=224)
    parser.add_argument("--resized_img_height", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate_only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model.lower(), weights=args.weights).to(device)

    checkpoint = torch.load(args.saved_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    console.print(f"✅ Loaded checkpoint from: [green]{args.saved_checkpoint_path}[/green]")

    _, _, test_loader = get_data_loaders(
        args.csv_root_dir, args.img_dir, args.batch_size,
        args.train_datacsv, args.val_datacsv, args.test_datacsv,
        resized_height=args.resized_img_height,
        resized_width=args.resized_img_weight
    )

    console.rule(f"🧪 [bold blue]Evaluating {args.model.upper()} on Test Set[/bold blue]")
    start = time.time()
    y_pred, y_true = evaluate(model, test_loader, device)
    duration = round(time.time() - start, 2)
    console.print(f"⏱️ Evaluation time: {duration}s")

    metrics = format_metrics(y_true, y_pred)

    model_tag = f"{args.model}_lr{args.learning_rate}_bs{args.batch_size}"
    save_dir = os.path.join("output", "Test_Results", model_tag)
    os.makedirs(save_dir, exist_ok=True)

    save_predictions(y_true, y_pred, os.path.join(save_dir, "predictions.csv"))
    plot_confusion_matrix(y_true, y_pred, SEVERITY_LABELS, os.path.join(save_dir, "confusion_matrix.png"), args.confusion_matrix_title)
    save_summary(metrics, os.path.join(save_dir, "test_summary.txt"), args)

    console.rule("[bold green]✅ Testing Complete[/bold green]")
    for k, v in metrics.items():
        console.print(f"[bold]{k.upper()}:[/bold] {v}")
