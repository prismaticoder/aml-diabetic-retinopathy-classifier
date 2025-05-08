import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        base_filename = self.annotations.iloc[idx, 0]
        label = int(self.annotations.iloc[idx, 1])

        for ext in [".jpeg", ".jpg", ".png"]:
            img_path = os.path.join(self.root_dir, base_filename + ext)
            if os.path.exists(img_path):
                break
        else:
            raise FileNotFoundError(f"Image not found for base filename: {base_filename}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        return image, label

def get_train_transform(data_aug, brightness, contrast, saturation, hue):
    transforms = [
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ]

    if data_aug:
        transforms.insert(1, A.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=1.0
        ))
        transforms.insert(2, A.HorizontalFlip(p=0.5))

    transforms.append(ToTensorV2())
    return A.Compose(transforms)

def get_data_loaders(csv_root_dir, img_dir, batch_size=32, num_workers=1,
                     only_test=False, data_aug=False, brightness=0.0,
                     contrast=0.0, saturation=0.0, hue=0.0):

    if only_test:
        test_csv_path = os.path.join(csv_root_dir, "test.csv")
        test_dataset = DiabeticRetinopathyDataset(test_csv_path, img_dir, get_train_transform(False, 0, 0, 0, 0))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return test_loader

    train_csv_path = os.path.join(csv_root_dir, "train.csv")
    val_csv_path = os.path.join(csv_root_dir, "val.csv")

    train_dataset = DiabeticRetinopathyDataset(train_csv_path, img_dir, get_train_transform(data_aug, brightness, contrast, saturation, hue))

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_dataset = DiabeticRetinopathyDataset(val_csv_path, img_dir, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
