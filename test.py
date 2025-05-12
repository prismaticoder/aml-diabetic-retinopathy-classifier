import argparse, os, torch, json
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt
from model import get_model
from dataset import get_data_loaders
from utils import set_seed, get_gpu_info
from rich.console import Console

console = Console()

def save_confusion_matrix(y_true, y_pred, class_names, save_path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate(model, loader, device, criterion):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    return total_loss / len(loader), precision, recall, f1, qwk, y_true, y_pred

def test(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        model_name=args.model,
        weights=args.weights,
        n_classes=args.n_classes,
        patch_size=args.patch_size
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.saved_checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load test data
    _, _, test_loader = get_data_loaders(
        csv_root_dir=args.csv_root_dir,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        train_csv=args.train_datacsv,
        val_csv=args.val_datacsv,
        test_csv=args.test_datacsv,
        resized_height=args.resized_img_height,
        resized_width=args.resized_img_weight,
        data_aug=False
    )

    criterion = nn.CrossEntropyLoss()
    loss, precision, recall, f1, qwk, y_true, y_pred = evaluate(model, test_loader, device, criterion)

    console.rule("[bold green]✅ Test Complete![/bold green]")
    console.print(f"📉 Loss      : {loss:.4f}")
    console.print(f"✅ Precision : {precision * 100:.2f}")
    console.print(f"✅ Recall    : {recall * 100:.2f}")
    console.print(f"✅ F1 Score  : {f1 * 100:.2f}")
    console.print(f"📊 QWK       : {qwk:.4f}")

    # Save classification report and confusion matrix
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    out_dir = os.path.join("output", "Test_Results", f"{args.model}_lr{args.learning_rate}_bs{args.batch_size}")
    os.makedirs(out_dir, exist_ok=True)

    df_report.to_csv(os.path.join(out_dir, "predictions.csv"), index=True)

    with open(os.path.join(out_dir, "test_summary.txt"), "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")
        f.write(f"QWK      : {qwk:.4f}\n")

    save_confusion_matrix(
        y_true, y_pred,
        class_names=[str(i) for i in range(args.n_classes)],
        save_path=os.path.join(out_dir, "confusion_matrix.png"),
        title=args.confusion_matrix_title or f"Confusion Matrix for {args.model}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--user', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--csv_root_dir', required=True)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--train_datacsv', required=True)
    parser.add_argument('--val_datacsv', required=True)
    parser.add_argument('--test_datacsv', required=True)
    parser.add_argument('--saved_checkpoint_path', required=True)
    parser.add_argument('--weights', default="DEFAULT")
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--resized_img_weight', type=int, default=224)
    parser.add_argument('--resized_img_height', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--log_dir', default="logs")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--lr_scheduler', default='cosine')
    parser.add_argument('--confusion_matrix_title', type=str, default=None)
    parser.add_argument('--evaluate_only', action="store_true")
    args = parser.parse_args()

    test(args)
