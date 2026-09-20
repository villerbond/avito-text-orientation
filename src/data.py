import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd

IMAGE_SIZE = (64, 256)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def get_train_transform(is_aug, image_size=IMAGE_SIZE):
    if (is_aug):
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN,std=NORM_STD),
])


def get_val_transform(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN,std=NORM_STD),
])


class OrientationDataset(Dataset):

    def __init__(self, image_dir, labels_df, transform=None):
        self.image_dir = image_dir
        self.labels = labels_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        image = Image.open(self.image_dir / row["file_name"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["label"],dtype=torch.float32)

        return image, label

def build_dataloaders(
    train_dir,
    val_dir,
    batch_size = 64,
    num_workers = 2,
    is_aug = False
):
    train_labels = pd.read_csv(train_dir / "labels.csv")
    val_labels = pd.read_csv(val_dir / "labels.csv")

    train_dataset = OrientationDataset(train_dir / "crops", train_labels, get_train_transform(is_aug))
    val_dataset = OrientationDataset(val_dir / "crops", val_labels, get_val_transform())

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    return train_loader, val_loader