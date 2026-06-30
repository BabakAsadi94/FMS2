import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    dataset_csv,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    transform=None,
    only_class=False,
):
    """Infinite generator of (mask_tensor, class_label) batches.

    Args:
        dataset_csv: Path to CSV with columns ``image_name`` and ``class``.
        data_dir: Root directory; image paths in the CSV are relative to this.
        batch_size: Batch size.
        image_size: Spatial resolution to resize masks to.
        deterministic: If True, disable shuffling.
        transform: Optional torchvision transform; default applies Resize +
            RandomHorizontalFlip + ToTensor + Normalize(0.5, 0.5).
        only_class: If True, skip image loading and return (path, class_label).
            Useful for inference when only the class conditioning is needed.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    df = pd.read_csv(dataset_csv)
    img_paths = df["image_name"].tolist()
    classes = df["class"].tolist()

    print(f"Found {len(img_paths)} images | {len(np.unique(classes))} classes")

    dataset = MaskDataset(
        data_dir=data_dir,
        resolution=image_size,
        image_paths=img_paths,
        classes=classes,
        transform=transform,
        only_class=only_class,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )

    while True:
        yield from loader


class MaskDataset(Dataset):
    def __init__(self, data_dir, resolution, image_paths, classes, transform=None, only_class=False):
        super().__init__()
        self.data_dir = data_dir
        self.resolution = resolution
        self.image_paths = image_paths
        self.classes = classes
        self.only_class = only_class

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.image_paths[idx])

        class_label = self.classes[idx]
        if not isinstance(class_label, torch.Tensor):
            class_label = torch.tensor(class_label, dtype=torch.long)

        if self.only_class:
            return path, class_label

        with open(path, "rb") as f:
            pil_image = Image.open(f).convert("L")

        image_tensor = self.transform(pil_image)
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0)

        return image_tensor, class_label
