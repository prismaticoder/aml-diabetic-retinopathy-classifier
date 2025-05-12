import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.image_names = self.data.iloc[:, 0].values
        self.labels = self.data.iloc[:, 1].values

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_id = str(self.image_names[idx])
        if not image_id.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_id += '.jpeg'
        img_path = os.path.join(self.img_dir, image_id)

        image = Image.open(img_path).convert("RGB")
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label

# ✅ Add this to allow train.py to import get_data_loaders
def get_data_loaders(csv_root_dir, img_dir, batch_size,
                     train_csv, val_csv, test_csv,
                     resized_height=224, resized_width=224, data_aug=True):

    train_transform = transforms.Compose([
        transforms.Resize((resized_height, resized_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ]) if data_aug else transforms.Compose([
        transforms.Resize((resized_height, resized_width)),
        transforms.ToTensor(),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((resized_height, resized_width)),
        transforms.ToTensor(),
    ])

    train_dataset = DRDataset(os.path.join(csv_root_dir, train_csv), img_dir, transform=train_transform)
    val_dataset = DRDataset(os.path.join(csv_root_dir, val_csv), img_dir, transform=val_test_transform)
    test_dataset = DRDataset(os.path.join(csv_root_dir, test_csv), img_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
