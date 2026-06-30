import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    """Image + binary-label dataset loaded from a CSV manifest.

    Expected CSV columns: ``image_name``, ``class``.
    All images must live under ``root``.
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        df = pd.read_csv(os.path.join(root, "dataset.csv"))
        self.img_paths = df["image_name"].tolist()
        self.labels = df["class"].tolist()

        print(f"CrackDataset: {len(self.img_paths)} images in {root}")
        for p in self.img_paths:
            if not os.path.exists(os.path.join(root, p)):
                raise FileNotFoundError(f"Image not found: {os.path.join(root, p)}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.img_paths[idx])).convert("RGB")
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label
