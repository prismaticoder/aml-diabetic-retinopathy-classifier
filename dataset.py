import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

# 📂 This class loads each image and its label from CSV + folder
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations.iloc[idx, 1])
        
        # Check if image is inverted
        # is_inverted = self.detect_inverted(image)
        
        # # Convert to numpy array for albumentation transforms
        # image_np = np.array(image)
        
        # # If image is inverted, flip it horizontally before other transforms
        # if is_inverted:
        #     image_np = np.fliplr(image_np)

        # Apply data augmentations if provided
        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        return image, label

# 🔧 Training transforms with augmentations
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# 🔧 Validation transforms (only essential preprocessing)
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def get_data_loaders(csv_path, img_dir, batch_size=32, num_workers=1):
    # Create two separate datasets with different transforms
    full_dataset = DiabeticRetinopathyDataset(
        csv_file=csv_path, 
        root_dir=img_dir, 
        transform=None
    )
    
    # Calculate split sizes
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Generate indices for split
    indices = torch.randperm(len(full_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train and val datasets with appropriate transforms
    train_dataset = DiabeticRetinopathyDataset(
        csv_file=csv_path, 
        root_dir=img_dir, 
        transform=train_transform
    )
    val_dataset = DiabeticRetinopathyDataset(
        csv_file=csv_path, 
        root_dir=img_dir, 
        transform=val_transform
    )
    
    # Get labels for computing class weights
    labels = [int(full_dataset.annotations.iloc[i, 1]) for i in train_indices]
    
    # Compute class weights
    class_counts = torch.bincount(torch.tensor(labels))
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    
    # Assign weight to each sample
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    # Create sampler for training data
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_indices),
        replacement=True
    )
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    # Use the sampler for training loader
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
