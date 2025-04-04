# test.py
import argparse, os, torch, json
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, cohen_kappa_score
from dataset import DiabeticRetinopathyDataset, val_transform
from model import get_model
from torch.utils.data import DataLoader
from datetime import datetime

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class="ovr")
    except:
        auc = -1
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return y_true, y_pred, precision, recall, f1, auc, qwk

def save_artifacts(y_true, y_pred, user, model, batch_size, lr, precision, recall, f1, auc, qwk):
    folder = f"output/Test_Results/{model}_lr{lr}_bs{batch_size}"
    os.makedirs(folder, exist_ok=True)

    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "confusion_matrix.png"))
    plt.close()

    # Save Predictions
    df = pd.DataFrame({
        "True Label": y_true,
        "Predicted Label": y_pred
    })
    df.to_csv(os.path.join(folder, "predictions.csv"), index=False)

    # Save Summary
    summary_path = os.path.join(folder, "test_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"User: {user}\nModel: {model}\nBatch Size: {batch_size}\nLearning Rate: {lr}\n\n")
        f.write("📊 Evaluation Metrics:\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {auc:.4f}\n")
        f.write(f"QWK: {qwk:.4f}\n")

def save_log(log_data, user, model, run_type, batch_size, lr):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{user}_{model}_{timestamp}_{run_type}_{batch_size}_{lr}.json"
    os.makedirs("logs", exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)
    print(f"✅ Log saved to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--csv_file", required=True)
    parser.add_argument("--img_dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model.lower(), pretrained=False).to(device)

    # Load best model checkpoint
    model_dir = f"output/{args.model}_lr{args.learning_rate}_bs{args.batch_size}"
    pths = sorted([f for f in os.listdir(model_dir) if f.endswith(".pth")])
    checkpoint = torch.load(os.path.join(model_dir, pths[-1]), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = DiabeticRetinopathyDataset(args.csv_file, args.img_dir, transform=val_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    y_true, y_pred, precision, recall, f1, auc, qwk = evaluate(model, loader, device)

    save_artifacts(y_true, y_pred, args.user, args.model, args.batch_size, args.learning_rate,
                   precision, recall, f1, auc, qwk)

    log_data = {
        "user": args.user,
        "model": args.model,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "qwk": qwk
    }
    save_log(log_data, args.user, args.model, "test", args.batch_size, args.learning_rate)
