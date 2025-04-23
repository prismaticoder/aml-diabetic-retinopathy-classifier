import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class TestRetinopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 0]
        label = int(self.annotations.iloc[idx, 1])
        img_path = os.path.join(self.root_dir, img_name + ".jpeg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        return image, label, img_name  # include filename

test_transform = A.Compose([
    A.Resize(192, 192),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def get_test_loader(test_csv_path, img_dir, batch_size=32):
    dataset = TestRetinopathyDataset(test_csv_path, img_dir, transform=test_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return loader
