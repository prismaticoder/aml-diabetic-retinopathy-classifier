import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".jpeg")
        image = Image.open(img_path).convert("RGB")
        image = preprocess_image_cv(image)
        label = int(self.annotations.iloc[idx, 1])

        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        return image, label

def preprocess_image_cv(image):
    image = np.array(image)
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    image = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

def get_transforms(profile="none"):
    if profile == "none":
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif profile == "basic":
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif profile == "advanced":
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.2),
                A.GaussianBlur(p=0.3)
            ], p=0.3),
            A.CLAHE(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        raise ValueError(f"❌ Unknown augmentation profile: {profile}")

def get_data_loaders(csv_root_dir, img_dir, batch_size=32, num_workers=1, profile="basic", only_test=False):
    train_csv_path = os.path.join(csv_root_dir, "train.csv")
    val_csv_path = os.path.join(csv_root_dir, "val.csv")
    test_csv_path = os.path.join(csv_root_dir, "test.csv")

    val_transform = get_transforms("none")
    train_transform = get_transforms(profile)

    if only_test:
        test_dataset = DiabeticRetinopathyDataset(test_csv_path, img_dir, val_transform)
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    train_dataset = DiabeticRetinopathyDataset(train_csv_path, img_dir, train_transform)
    val_dataset = DiabeticRetinopathyDataset(val_csv_path, img_dir, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
