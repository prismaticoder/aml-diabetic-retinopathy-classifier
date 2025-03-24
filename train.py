import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import torchvision.models as models
from dataset import get_data_loaders, train_transform, val_transform
from sklearn.metrics import cohen_kappa_score
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, Tuple

# Configuration
class TrainingConfig:
    def __init__(self, batch_size: int = 32, learning_rate: float = 0.0001, user_name: str = ""):
        self.num_epochs = 10
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = 5
        self.save_every = 2
        self.csv_path = "dataset/trainLabels.csv"
        self.img_dir = "dataset/train"
        self.user_name = user_name

def setup_directories(model_name: str, config: TrainingConfig) -> Tuple[str, str]:
    """Create necessary directories and return paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup output directory
    output_dir = f"output/{model_name}_lr{config.learning_rate}_bs{config.batch_size}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup log directory with user name
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{config.user_name.lower()}_{model_name}_{timestamp}.json")
    
    return output_dir, log_file

def initialize_logging(model_name: str, config: TrainingConfig) -> Dict[str, Any]:
    """Initialize the logging dictionary with metadata."""
    return {
        "user": config.user_name,
        "model_name": model_name,
        "hyperparameters": {
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs
        },
        "transforms": {
            "train": str(train_transform),
            "validation": str(val_transform)
        },
        "epochs": []
    }

def setup_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    """Initialize and configure the model."""
    if not hasattr(models, model_name):
        raise ValueError(f"❌ Model '{model_name}' not found in torchvision.")
        
    model = getattr(models, model_name)(weights="DEFAULT")
    
    # Modify final layer for classification
    if hasattr(model, "fc"):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif hasattr(model, "classifier"):
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        
    return model.to(device)

def train_epoch(model: nn.Module, train_loader: torch.utils.data.DataLoader, 
                criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                device: torch.device) -> Tuple[float, float, int]:
    """Run one epoch of training."""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return train_loss, correct, total

def validate(model: nn.Module, val_loader: torch.utils.data.DataLoader, 
            criterion: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    """Run validation and compute metrics."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    qwk = cohen_kappa_score(np.array(all_labels), np.array(all_preds), weights='quadratic')
    
    return val_loss, accuracy, qwk

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, metrics: Dict[str, float], 
                   output_dir: str, model_name: str, log_file: str):
    """Save model checkpoint with metrics."""
    model_filename = f"{output_dir}/{model_name}_epoch{epoch+1}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **metrics,
        'log_file': log_file
    }, model_filename)
    print(f"✅ Model saved at: {model_filename}")

def train_model(model_name: str, config: TrainingConfig):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir, log_file = setup_directories(model_name, config)
    training_log = initialize_logging(model_name, config)
    
    # Setup model and training components
    model = setup_model(model_name, config.num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        config.csv_path, config.img_dir, 
        batch_size=config.batch_size
    )
    
    print(f"🔹 Training {model_name} on device: {device}...")
    
    for epoch in range(config.num_epochs):
        # Training phase
        train_loss, train_correct, train_total = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation phase
        val_loss, val_accuracy, val_qwk = validate(
            model, val_loader, criterion, device
        )
        
        # Calculate training accuracy
        train_accuracy = 100 * train_correct / train_total
        
        # Log epoch results
        epoch_log = {
            "epoch": epoch + 1,
            "training": {"loss": train_loss, "accuracy": train_accuracy},
            "validation": {"loss": val_loss, "accuracy": val_accuracy, "qwk": val_qwk}
        }
        training_log["epochs"].append(epoch_log)
        
        # Save log file
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=4)
            
        # Print progress
        print(f"\n✅ {model_name} - Epoch {epoch+1}")
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, QWK: {val_qwk:.4f}")
        print(f"📝 Log saved to: {log_file}")
        print("--------------------------------")
        
        # Save checkpoint
        if epoch % config.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch,
                {"train_loss": train_loss, "val_loss": val_loss, 
                 "val_accuracy": val_accuracy, "val_qwk": val_qwk},
                output_dir, model_name, log_file
            )

def validate_learning_rate(lr_str: str) -> float:
    """Validate and convert learning rate input."""
    try:
        lr = float(lr_str)
        if 0.0001 <= lr <= 0.005:
            return lr
        raise ValueError
    except ValueError:
        print("❌ Invalid learning rate. Using default value 0.0001")
        print("Learning rate must be a float between 0.0001 and 0.005")
        return 0.0001

def validate_batch_size(batch_str: str) -> int:
    """Validate and convert batch size input."""
    try:
        batch_size = int(batch_str)
        if batch_size > 0:
            return batch_size
        raise ValueError
    except ValueError:
        print("❌ Invalid batch size. Using default value 32")
        print("Batch size must be a positive integer")
        return 32

if __name__ == "__main__":
    # Get user selection
    print("Select your name: (Note: This will be used to identify your training results)")
    print("1. Larry")
    print("2. Meena")
    print("3. Tom")
    print("4. Zohaib")
    print("5. Reviewer")
    
    user_map = {
        "1": "Larry",
        "2": "Meena",
        "3": "Tom",
        "4": "Zohaib",
        "5": "Reviewer"
    }
    
    while True:
        user_choice = input("Enter your number (1-4): ").strip()
        if user_choice in user_map:
            user_name = user_map[user_choice]
            print(f"\nHi {user_name}! Please choose your training parameters.")
            break
        print("Invalid choice. Please enter a number between 1 and 4.")

    # Get model input
    model_name = input("Enter model to train (e.g., resnet50, efficientnet_b0, vgg16, densenet121): ").strip().lower()
    
    # Get optional batch size
    batch_input = input("Enter batch size (default=32, press Enter to skip): ").strip()
    batch_size = validate_batch_size(batch_input) if batch_input else 32
    
    # Get optional learning rate
    lr_input = input("Enter learning rate (default=0.0001, press Enter to skip): ").strip()
    learning_rate = validate_learning_rate(lr_input) if lr_input else 0.0001
    
    print(f"\n🔧 Training Configuration:")
    print(f"Model: {model_name}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print("-------------------")
    
    config = TrainingConfig(
        batch_size=batch_size, 
        learning_rate=learning_rate,
        user_name=user_name
    )
    train_model(model_name, config)
