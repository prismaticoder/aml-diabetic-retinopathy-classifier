from dataset import get_data_loaders

# Define file paths
csv_path = "dataset/trainLabels.csv"
img_dir = "dataset/train"

# Load dataset
train_loader, val_loader = get_data_loaders(csv_path, img_dir, batch_size=4)

# Check one batch
data_iter = iter(train_loader)
images, labels = next(data_iter)

print(f"Loaded batch size: {len(images)}")
print(f"Labels: {labels}")
 