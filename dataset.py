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

    def detect_notch(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check the main contour (should be the retina circle)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            # Approximate the contour to simplify shape
            epsilon = 0.02 * cv2.arcLength(main_contour, True)
            approx = cv2.approxPolyDP(main_contour, epsilon, True)
            
            # If the approximated contour has more than 8 points but less than expected for a circle,
            # it likely has a notch
            return len(approx) > 8 and len(approx) < 50
        return False

    def detect_macula_position(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to identify dark regions (macula)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find the center of the image
        height, width = thresh.shape
        center_y = height // 2
        
        # Split image into left and right halves
        left_half = thresh[:, :width//2]
        right_half = thresh[:, width//2:]
        
        # Find the darkest region (macula) in each half
        left_dark = np.sum(left_half > 0)
        right_dark = np.sum(right_half > 0)
        
        # If more dark pixels in left half, macula is likely on left
        macula_on_left = left_dark > right_dark
        
        # Find average y-position of dark pixels
        y_coords = np.where(thresh > 0)[0]
        if len(y_coords) > 0:
            avg_y = np.mean(y_coords)
            # If average y position is above center, macula is higher than midline
            macula_higher = avg_y < center_y
            return macula_higher
        return False

    def detect_inverted(self, image):
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Check both criteria
        has_notch = self.detect_notch(image)
        macula_higher = self.detect_macula_position(image)
        
        # Logic based on the criteria:
        # 1. If there's a notch, image is NOT inverted
        # 2. If macula is higher than midline, image IS inverted
        # 3. If criteria conflict, prioritize notch detection
        if has_notch:
            return False
        return macula_higher

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
