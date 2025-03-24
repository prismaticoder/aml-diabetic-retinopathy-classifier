# Import all the important libraries needed for model training
import torch  # Core PyTorch library
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimizers like Adam, SGD
import os  # File operations (creating directories etc.)
from tqdm import tqdm  # Progress bar for training loop
import torchvision.models as models  # Pre-trained models like ResNet, EfficientNet
from dataset import get_data_loaders  # Function to load and preprocess your data

# Set key training parameters
num_epochs = 10  # How many times the model will see the entire dataset
batch_size = 32  # Number of images processed at once
learning_rate = 0.0001  # How fast the model updates weights (smaller = safer)

# 🧠 Main training function that accepts the model name as a string
def train_model(model_name):
    """Loads the model, trains it, and saves checkpoints every 2 epochs."""

    # 🚨 Safety Check: Make sure the model name exists in torchvision
    if not hasattr(models, model_name):
        raise ValueError(f"❌ Model '{model_name}' is not found in torchvision. Please check spelling.")

    # Paths for loading images and labels (CSV file)
    csv_path = "dataset/trainLabels.csv"
    img_dir = "dataset/train"

    # Get our training and validation data loaders
    train_loader, val_loader = get_data_loaders(csv_path, img_dir, batch_size=batch_size)

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Training {model_name} on device: {device}...")

    # Dynamically get the model class from torchvision
    model = getattr(models, model_name)(weights="DEFAULT")

    # ✨ Modify the last classification layer to predict 5 classes (0 to 4 DR severity levels)
    if hasattr(model, "fc"):  # ResNet, DenseNet have 'fc' (fully connected) layers
        num_ftrs = model.fc.in_features  # Get number of input features for that layer
        model.fc = nn.Linear(num_ftrs, 5)  # Replace it with our custom 5-class layer
    elif hasattr(model, "classifier"):  # EfficientNet, VGG use 'classifier'
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 5)

    # Move model to the right device (GPU/CPU)
    model = model.to(device)

    # 🎯 Define the loss function - CrossEntropyLoss is great for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # 🛠️ Define the optimizer - Adam is adaptive and generally performs well
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 📁 Create output folder to save model checkpoints
    output_dir = f"output/{model_name}_lr{learning_rate}_bs{batch_size}"
    os.makedirs(output_dir, exist_ok=True)

    # 🌀 Start training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode (enables dropout, batchnorm updates)
        running_loss = 0.0  # Track cumulative loss for the epoch
        correct = 0  # Count correct predictions
        total = 0  # Total samples processed

        # Loop over batches
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - {model_name}"):
            # Move data to GPU/CPU
            images, labels = images.to(device), labels.to(device)

            # 🧼 Reset gradients before every step
            optimizer.zero_grad()

            # 🔮 Forward pass - get predictions from model
            outputs = model(images)

            # 🎯 Compute how far predictions are from actual labels
            loss = criterion(outputs, labels)

            # 🔁 Backpropagation - update weights
            loss.backward()
            optimizer.step()

            # 📊 Update metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get class with highest probability
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # 🎉 Calculate accuracy for the epoch
        train_acc = 100 * correct / total
        print(f"✅ {model_name} - Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # 💾 Save model every 2 epochs
        if epoch % 2 == 0:
            model_filename = f"{output_dir}/{model_name}_epoch{epoch+1}.pth"
            torch.save({'state_dict': model.state_dict()}, model_filename)
            print(f"✅ Model saved at: {model_filename}")

# 🔽 Entry point when script is run directly
if __name__ == "__main__":
    # 🧑 Ask the user which model they want to train
    model_name = input("Enter model to train (e.g., resnet50, efficientnet_b0, vgg16, densenet121): ").strip().lower()
    
    # Start the training process
    train_model(model_name)
