import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# 📂 This class loads each image and its label from CSV + folder
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 🖼️ Get image path like: train/10_left.jpeg
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".jpeg")
        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations.iloc[idx, 1])

        # 🔁 Apply data augmentations if provided
        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        return image, label

# 🔧 Here's the preprocessing + augmentation pipeline used on all images
train_transform = A.Compose([
    A.Resize(224, 224),  # All images resized to 224x224 for model input
    A.HorizontalFlip(p=0.5),  # Flip 50% of images for variety
    A.RandomBrightnessContrast(p=0.2),  # Small brightness/contrast changes
    A.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize like ImageNet
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()  # Convert the image to a PyTorch Tensor
])

# 🚚 This function returns train and validation DataLoaders
def get_data_loaders(csv_path, img_dir, batch_size=32, num_workers=1):
    dataset = DiabeticRetinopathyDataset(csv_file=csv_path, root_dir=img_dir, transform=train_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
