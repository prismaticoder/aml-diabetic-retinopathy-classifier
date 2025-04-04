# train.py
import argparse, os, json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
from dataset import get_data_loaders
from model import get_model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Training"):
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
    return total_loss, acc

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
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
    model_path = os.path.join(model_dir, f"{model_name}_epoch{epoch}.pth")
    torch.save({"model_state_dict": model.state_dict()}, model_path)
    print(f"💾 Saved model: {model_path}")

def save_log(log, user, model_name, run_type, batch_size, lr):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/{user}_{model_name}_{timestamp}_{run_type}_{batch_size}_{lr}.json"
    os.makedirs("logs", exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=4)
    print(f"📄 Log saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--csv_root_dir", required=True)
    parser.add_argument("--img_dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model.lower(), pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = get_data_loaders(args.csv_root_dir, args.img_dir, args.batch_size)

    model_dir = f"output/{args.model}_lr{args.learning_rate}_bs{args.batch_size}"
    log_data = {
        "user": args.user,
        "model": args.model,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": []
    }

    for epoch in range(1, 11):
        print(f"\n🔁 Epoch {epoch}/10")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_qwk = validate(model, val_loader, criterion, device)

        print(f"📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"📊 Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.2f}% | QWK: {val_qwk:.4f}")

        log_data["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_qwk": val_qwk
        })

        if epoch % 2 == 1:
            save_checkpoint(model, model_dir, args.model, epoch)

    save_log(log_data, args.user, args.model, "train", args.batch_size, args.learning_rate)
