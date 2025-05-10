import os
import pandas as pd
import numpy as np
from PIL import Image
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
        img_name = self.annotations.iloc[idx, 0]
        label = int(self.annotations.iloc[idx, 1])
        img_path = os.path.join(self.root_dir, img_name + ".jpeg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        return image, label, img_name  # Include filename for misclassification

def get_data_loaders(
    csv_root_dir,
    img_dir,
    batch_size=32,
    num_workers=2,
    train_csv="train.csv",
    val_csv="val.csv",
    test_csv="test.csv",
    resized_height=224,
    resized_width=224
):
    base_transform = lambda: A.Compose([
        A.Resize(resized_height, resized_width),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    def make_loader(csv_file):
        path = os.path.join(csv_root_dir, csv_file)
        dataset = DiabeticRetinopathyDataset(
            csv_file=path,
            root_dir=img_dir,
            transform=base_transform()
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    train_loader = make_loader(train_csv) if train_csv else None
    val_loader = make_loader(val_csv) if val_csv else None
    test_loader = make_loader(test_csv) if test_csv else None

    return train_loader, val_loader, test_loader
