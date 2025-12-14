from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ImagePathDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label


def list_images(class_dir):
    for path in class_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
            yield path


def has_any_images(class_dir):
    return next(list_images(class_dir), None) is not None


def collect_samples(raw_dir):
    class_dirs = [
        p
        for p in raw_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and has_any_images(p)
    ]
    class_dirs = sorted(class_dirs, key=lambda p: p.name)
    class_names = [p.name for p in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    samples = []
    for class_dir in class_dirs:
        label = class_to_idx[class_dir.name]
        for image_path in sorted(list_images(class_dir)):
            samples.append((image_path, label))

    if not samples:
        raise RuntimeError(f"No images found in '{raw_dir}'.")
    return samples, class_names


def compute_class_weights(train_labels, num_classes):
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv / inv.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


class SharkDataModule(pl.LightningDataModule):
    def __init__(
        self,
        raw_dir,
        batch_size,
        num_workers,
        image_size,
        mean,
        std,
        test_ratio,
        val_ratio_within_train,
        random_state,
    ):
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.test_ratio = test_ratio
        self.val_ratio_within_train = val_ratio_within_train
        self.random_state = random_state

        self.class_names = []
        self.class_weights = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        samples, class_names = collect_samples(self.raw_dir)
        self.class_names = class_names

        labels = np.array([label for _, label in samples], dtype=np.int64)
        indices = np.arange(len(samples))

        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_ratio,
            random_state=self.random_state,
            stratify=labels,
        )

        train_labels = labels[train_idx]
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=self.val_ratio_within_train,
            random_state=self.random_state,
            stratify=train_labels,
        )

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        test_samples = [samples[i] for i in test_idx]

        self.class_weights = compute_class_weights(
            train_labels=np.array([label for _, label in train_samples], dtype=np.int64),
            num_classes=len(self.class_names),
        )

        train_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        self.train_dataset = ImagePathDataset(train_samples, transform=train_transform)
        self.val_dataset = ImagePathDataset(val_samples, transform=eval_transform)
        self.test_dataset = ImagePathDataset(test_samples, transform=eval_transform)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
