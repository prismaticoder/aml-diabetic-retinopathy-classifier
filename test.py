# 📦 Importing Required Libraries

import torch                      # PyTorch - main library for deep learning
import os                         # Helps with file paths and directories
import matplotlib.pyplot as plt   # For plotting graphs
import seaborn as sns             # For prettier plots
import pandas as pd               # For handling CSV files and dataframes
from sklearn.metrics import confusion_matrix, accuracy_score  # For evaluating model performance
from dataset import get_data_loaders                           # Custom function to load our dataset
import torchvision.models as models                            # Pretrained models like ResNet, EfficientNet
from tqdm import tqdm                                            # Shows loading bars during loops
import torch.nn as nn                                            # Neural network layers and functions

# 🧠 Define testing configuration (you can tweak these)
batch_size = 32
learning_rate = 0.0001
csv_path = "dataset/trainLabels.csv"     # Labels CSV
img_dir = "dataset/train"                # Image folder
output_dir = "output"                    # Where model weights and results are saved

# 🧪 This function runs the actual model testing
def test_model(model_name):

    # Build the path to the model directory based on its name and training settings
    model_dir = os.path.join(output_dir, f"{model_name}_lr0.0001_bs{batch_size}")

    # If no trained model found in that folder, give error and stop
    if not os.path.exists(model_dir) or not any(f.endswith(".pth") for f in os.listdir(model_dir)):
        print(f"❌ No trained model found for '{model_name}'! Please train the model first.")
        return

    # Get the latest checkpoint (last model saved during training)
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    model_files.sort()  # Sort by name so we can pick the latest
    model_path = os.path.join(model_dir, model_files[-1])
    print(f"✅ Loaded Model: {model_path}")

    # Choose device: Use GPU (cuda) if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the base model structure (like ResNet or EfficientNet)
    model = getattr(models, model_name)(weights="DEFAULT")

    # Modify the final layer to match our 5 DR classes
    if hasattr(model, "fc"):  # ResNet-style
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)
    elif hasattr(model, "classifier"):  # EfficientNet-style
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 5)

    model = model.to(device)  # Move model to device (CPU or GPU)

    # 🔁 Load the actual trained weights
    checkpoint = torch.load(model_path, map_location=device)

    # Some models are saved as {'state_dict': model.state_dict()}
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()  # Set the model to evaluation mode (disables training features like dropout)

    # Where to save the test results
    test_results_dir = os.path.join(output_dir, "Test_Results", f"{model_name}_lr{learning_rate}_bs{batch_size}")
    os.makedirs(test_results_dir, exist_ok=True)

    # Load validation set using same transformation and batch size as training
    _, val_loader = get_data_loaders(csv_path, img_dir, batch_size=batch_size)

    # Create empty lists to store results
    y_true, y_pred = [], []
    total_samples = 0
    correct_predictions = 0

    # 🧪 Run the model on the validation set
    with torch.no_grad():  # Disable gradient tracking for efficiency
        for images, labels in tqdm(val_loader, desc=f"Testing {model_name}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Pick class with highest probability

            # Save ground truth and predictions
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Count how many predictions were correct
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # ✅ Calculate overall accuracy
    test_accuracy = accuracy_score(y_true, y_pred) * 100
    print(f"\n🔥 {model_name} - Final Test Accuracy: {test_accuracy:.2f}% 🔥")

    # 🧾 Confusion matrix (table showing where predictions go wrong)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")

    # 📝 Save test summary as text file
    summary_path = os.path.join(test_results_dir, "test_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Final Test Accuracy: {test_accuracy:.2f}%\n")
    print(f"✅ Test summary saved at: {summary_path}")

    # 📄 Save predictions as CSV
    predictions_df = pd.DataFrame({"True Label": y_true, "Predicted Label": y_pred})
    predictions_csv_path = os.path.join(test_results_dir, "predictions.csv")
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"✅ Predictions saved at: {predictions_csv_path}")

    # 💾 Save confusion matrix as image
    cm_path = os.path.join(test_results_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"✅ Confusion Matrix saved at: {cm_path}")

# 🔧 Run testing from command line
if __name__ == "__main__":
    model_name = input("Enter the model to test (e.g., resnet50, efficientnet_b0, vgg16, densenet121): ").strip().lower()
    test_model(model_name)
