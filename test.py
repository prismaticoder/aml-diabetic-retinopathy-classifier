import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from dataset import get_data_loaders
import torchvision.models as models
from tqdm import tqdm
import torch.nn as nn

# Testing Parameters
batch_size = 32
learning_rate = 0.0001
csv_path = "dataset/trainLabels.csv"
img_dir = "dataset/train"
output_dir = "output"

def test_model(model_name):
    """Function to test the model"""
    
    model_dir = os.path.join(output_dir, f"{model_name}_lr0.0001_bs{batch_size}")

    # Check if trained model exists
    if not os.path.exists(model_dir) or not any(f.endswith(".pth") for f in os.listdir(model_dir)):
        print(f"❌ No trained model found for '{model_name}'! Please train the model first.")
        return

    # Find the latest trained model
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    model_files.sort()
    model_path = os.path.join(model_dir, model_files[-1])
    print(f"✅ Loaded Model: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model Dynamically with Correct Weights
    model = getattr(models, model_name)(weights="DEFAULT")  # Use correct weights

    # Modify last layer for 5 classes
    if hasattr(model, "fc"):  # ResNet, DenseNet
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)
    elif hasattr(model, "classifier"):  # VGG, EfficientNet
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 5)

    model = model.to(device)  # Move model to device

    # Load trained weights correctly
    checkpoint = torch.load(model_path, map_location=device)
    if "state_dict" in checkpoint:  # If saved as {'state_dict': model.state_dict()}
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    test_results_dir = os.path.join(output_dir, "Test_Results", f"{model_name}_lr{learning_rate}_bs{batch_size}")
    # test_results_dir = os.path.join(output_dir, "Test_Results", model_dir.split("/")[-1])

    os.makedirs(test_results_dir, exist_ok=True)

    _, val_loader = get_data_loaders(csv_path, img_dir, batch_size=batch_size)

    y_true, y_pred = [], []
    total_samples, correct_predictions = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Testing {model_name}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    test_accuracy = accuracy_score(y_true, y_pred) * 100
    print(f"\n🔥 {model_name} - Final Test Accuracy: {test_accuracy:.2f}% 🔥")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")

 # Save test summary
    summary_path = os.path.join(test_results_dir, "test_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Final Test Accuracy: {test_accuracy:.2f}%\n")
    print(f"✅ Test summary saved at: {summary_path}")

 # Save predictions CSV
    predictions_df = pd.DataFrame({"True Label": y_true, "Predicted Label": y_pred})
    predictions_csv_path = os.path.join(test_results_dir, "predictions.csv")
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"✅ Predictions saved at: {predictions_csv_path}")

    # Save confusion matrix image
    cm_path = os.path.join(test_results_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"✅ Confusion Matrix saved at: {cm_path}")

if __name__ == "__main__":
    model_name = input("Enter the model to test (e.g., resnet50, efficientnet_b0, vgg16, densenet121): ").strip().lower()
    test_model(model_name)
