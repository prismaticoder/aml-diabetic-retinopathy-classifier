import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import torchvision.models as models
from dataset import get_data_loaders

# Training Parameters
num_epochs = 10
batch_size = 32
learning_rate = 0.0001

def train_model(model_name):
    """Function to train the model"""
    
    # Check if model exists in torchvision
    if not hasattr(models, model_name):
        raise ValueError(f"❌ Model '{model_name}' not found in torchvision!")

    csv_path = "dataset/trainLabels.csv"
    img_dir = "dataset/train"
    train_loader, val_loader = get_data_loaders(csv_path, img_dir, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Training {model_name}...")

    # Load Model Dynamically with Correct Weights
    model = getattr(models, model_name)(weights="DEFAULT")  # Updated to avoid deprecation warning

    # Modify last layer for 5 classes
    if hasattr(model, "fc"):  # ResNet, DenseNet
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)
    elif hasattr(model, "classifier"):  # VGG, EfficientNet
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 5)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    output_dir = f"output/{model_name}_lr{learning_rate}_bs{batch_size}"
    os.makedirs(output_dir, exist_ok=True)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - {model_name}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"✅ {model_name} - Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # Save Model Every 2 Epochs
        if epoch % 2 == 0:
            model_filename = f"{output_dir}/{model_name}_epoch{epoch+1}.pth"
            torch.save({'state_dict': model.state_dict()}, model_filename)
            print(f"✅ Model saved: {model_filename}")

if __name__ == "__main__":
    # Take input only once
    model_name = input("Enter model to train (e.g., resnet50, efficientnet_b0, vgg16, densenet121): ").strip().lower()
    
    # Start training
    train_model(model_name)
