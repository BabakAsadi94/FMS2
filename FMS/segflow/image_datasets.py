import math
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    class_dir,
    batch_size,
    image_size,
    deterministic=False,
    random_crop=True,
    random_flip=True,
    drop_last=True,
    normalsize_0_1=False,
):
    """Return ``(infinite_generator, dataset_length)`` for paired image/mask data.

    Args:
        data_dir: List of directories containing RGB images.
        class_dir: List of directories containing annotation masks (same length as data_dir).
        batch_size: Batch size.
        image_size: Spatial resolution.
        deterministic: Disable shuffling when True.
        random_crop: Apply random crop augmentation.
        random_flip: Apply random horizontal flip.
        drop_last: Drop the last incomplete batch.
        normalsize_0_1: Normalize mask to [0,1] instead of [-1,1].
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    if len(data_dir) != len(class_dir):
        raise ValueError("data_dir and class_dir must have the same length")

    all_files = []
    all_classes = []
    for d_dir, c_dir in zip(data_dir, class_dir):
        all_files.extend(_list_images(d_dir))
        all_classes.extend(_list_images(c_dir))

    print(f"Found {len(all_classes)} annotation maps")

    img_basenames = set(os.path.basename(f) for f in all_files)
    cls_basenames = set(os.path.basename(f) for f in all_classes)
    assert img_basenames == cls_basenames, (
        f"Image/annotation mismatch — "
        f"only-in-images: {img_basenames - cls_basenames}, "
        f"only-in-annotations: {cls_basenames - img_basenames}"
    )

    print(f"Dataset size: {len(all_files)}")

    dataset = ImageDataset(
        resolution=image_size,
        image_paths=all_files,
        classes=all_classes,
        random_crop=random_crop,
        random_flip=random_flip,
        normalsize_0_1=normalsize_0_1,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=drop_last,
    )

    print(f"Loader batches: {len(loader)}")

    def _generator():
        while True:
            yield from loader

    return _generator(), len(dataset)


def _list_images(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, entry)
        if os.path.isfile(full) and entry.rsplit(".", 1)[-1].lower() in {"jpg", "jpeg", "png", "gif"}:
            results.append(full)
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None,
                 random_crop=False, random_flip=True, normalsize_0_1=False):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths
        self.local_classes = classes
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.normalsize_0_1 = normalsize_0_1

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        class_path = self.local_classes[idx]
        assert os.path.basename(path) == os.path.basename(class_path), (
            f"Filename mismatch: {path} vs {class_path}"
        )

        pil_image = Image.open(path).convert("RGB")
        pil_class = Image.open(class_path).convert("RGB")

        if self.random_crop:
            arr_image, arr_class = random_crop_arr([pil_image, pil_class], self.resolution)
        else:
            arr_image, arr_class = center_crop_arr([pil_image, pil_class], self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr_image = arr_image[:, ::-1].copy()
            arr_class = arr_class[:, ::-1].copy()

        arr_image = arr_image.astype(np.float32) / 127.5 - 1
        if self.normalsize_0_1:
            arr_class = arr_class.astype(np.float32) / 255.0
        else:
            arr_class = arr_class.astype(np.float32) / 127.5 - 1

        return (
            np.transpose(arr_image, [2, 0, 1]),
            np.transpose(arr_class, [2, 0, 1]),
            os.path.basename(path),
        )


def center_crop_arr(pil_list, image_size):
    pil_image, pil_class = pil_list
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    cy = (arr_image.shape[0] - image_size) // 2
    cx = (arr_image.shape[1] - image_size) // 2
    s = np.s_[cy: cy + image_size, cx: cx + image_size]
    return arr_image[s], arr_class[s]


def random_crop_arr(pil_list, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_dim = math.ceil(image_size / max_crop_frac)
    max_dim = math.ceil(image_size / min_crop_frac)
    dim = random.randrange(min_dim, max_dim + 1)

    pil_image, pil_class = pil_list
    while min(*pil_image.size) >= 2 * dim:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = dim / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    cy = random.randrange(arr_image.shape[0] - image_size + 1)
    cx = random.randrange(arr_image.shape[1] - image_size + 1)
    s = np.s_[cy: cy + image_size, cx: cx + image_size]
    return arr_image[s], arr_class[s]
