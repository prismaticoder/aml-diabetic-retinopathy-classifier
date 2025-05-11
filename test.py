import argparse, os, time, json, torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score,
    precision_score, recall_score, f1_score, roc_auc_score
)

from model import get_model
from dataset import get_data_loaders
from utils import set_seed

SEVERITY_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def save_confusion_matrix(y_true, y_pred, class_names, save_path, title):
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

def test(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model(args.model.lower(), weights=args.weights, n_classes=args.n_classes).to(device)
    checkpoint = torch.load(args.saved_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    # Load test data
    _, _, test_loader = get_data_loaders(
        csv_root_dir=args.csv_root_dir,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        train_csv=os.path.basename(args.train_datacsv),
        val_csv=os.path.basename(args.val_datacsv),
        test_csv=os.path.basename(args.test_datacsv),
        resized_height=args.resized_img_height,
        resized_width=args.resized_img_weight
    )

    # Inference
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(test_loader.dataset)
    metrics = format_metrics(all_labels, all_preds)

    # Save everything
    model_tag = f"{args.model}_lr{args.learning_rate}_bs{args.batch_size}"
    save_dir = os.path.join("output", "Test_Results", model_tag)
    os.makedirs(save_dir, exist_ok=True)

    # Save metrics
    with open(os.path.join(save_dir, "test_summary.txt"), "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"User: {args.user}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n\n")
        f.write("=== Evaluation Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k.upper()}: {v}\n")

    # Save predictions
    df_pred = pd.DataFrame({"actual": all_labels, "predicted": all_preds})
    df_pred.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    # Save confusion matrix
    save_confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=SEVERITY_LABELS,
        save_path=os.path.join(save_dir, "confusion_matrix.png"),
        title=args.confusion_matrix_title or f"Confusion Matrix for {args.model}"
    )

    # Print to console
    print("✅ Test Complete!")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v}")

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
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--resized_img_weight", type=int, default=224)
    parser.add_argument("--resized_img_height", type=int, default=224)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--confusion_matrix_title", type=str, default=None)
    parser.add_argument("--evaluate_only", action="store_true")
    args = parser.parse_args()
    test(args)
