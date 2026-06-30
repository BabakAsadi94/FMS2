import math
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except (ImportError, OSError, RuntimeError):
    MPI_AVAILABLE = False

    class _MockComm:
        @staticmethod
        def Get_rank():
            return 0

        @staticmethod
        def Get_size():
            return 1

    class _MockMPI:
        COMM_WORLD = _MockComm()

    MPI = _MockMPI()


def load_data(
    *,
    dataset_mode,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=True,
    random_flip=True,
    is_train=True,
):
    """Infinite generator of (image_tensor, label_dict) batches.

    Supported dataset modes: ``fms``.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if dataset_mode == "fms":
        all_files = _list_images(os.path.join(data_dir, "training", "images"))
        classes   = _list_images(os.path.join(data_dir, "training", "annotations"))
    else:
        raise NotImplementedError(f"dataset_mode '{dataset_mode}' not implemented")

    _check_alignment(all_files, classes, dataset_mode)
    print(f"Dataset size: {len(all_files)} images")

    dataset = ImageDataset(
        dataset_mode,
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        is_train=is_train,
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


def load_data_sample(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    is_train=False,
):
    """Like load_data but only returns annotation masks (no images).
    Useful for sampling / evaluation loops.
    """
    classes = _list_images(data_dir)
    if not classes:
        raise ValueError(f"No image files found in {data_dir}")
    print(f"Annotation-only dataset: {len(classes)} files")

    dataset = _AnnotationOnlyDataset(
        image_size,
        classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _list_images(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, entry)
        if os.path.isfile(full) and entry.rsplit(".", 1)[-1].lower() in {"jpg", "jpeg", "png", "gif"}:
            results.append(full)
        elif os.path.isdir(full):
            results.extend(_list_images(full))
    return results


def _check_alignment(all_files, classes, mode):
    img_names = {os.path.basename(p) for p in all_files}
    cls_names = {os.path.basename(p) for p in classes}
    only_img = img_names - cls_names
    only_cls = cls_names - img_names
    if only_img:
        print(f"[{mode}] images without annotation: {only_img}")
    if only_cls:
        print(f"[{mode}] annotations without image: {only_cls}")
    assert img_names == cls_names, f"[{mode}] image/annotation mismatch"


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_mode,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        is_train=True,
    ):
        super().__init__()
        self.dataset_mode = dataset_mode
        self.resolution = resolution
        self.is_train = is_train
        self.random_crop = random_crop
        self.random_flip = random_flip

        self.local_images  = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        pil_image = Image.open(self.local_images[idx]).convert("RGB")
        pil_class = Image.open(self.local_classes[idx]).convert("L")

        if self.is_train:
            fn = _random_crop if self.random_crop else _center_crop
            arr_img, arr_cls = fn([pil_image, pil_class], self.resolution)
        else:
            arr_img, arr_cls = _resize([pil_image, pil_class], self.resolution, keep_aspect=False)

        if self.random_flip and random.random() < 0.5:
            arr_img = arr_img[:, ::-1].copy()
            arr_cls = arr_cls[:, ::-1].copy()

        arr_img = arr_img.astype(np.float32) / 127.5 - 1.0

        out = {"path": self.local_images[idx], "label_ori": arr_cls.copy()}

        if self.dataset_mode in ("fms",):
            arr_cls = (arr_cls > 0).astype(arr_cls.dtype)

        out["label"] = arr_cls[None]

        return np.transpose(arr_img, [2, 0, 1]), out


class _AnnotationOnlyDataset(Dataset):
    def __init__(self, resolution, classes, shard=0, num_shards=1,
                 random_crop=False, random_flip=False):
        super().__init__()
        self.resolution = resolution
        self.random_flip = random_flip
        self.local_classes = classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_classes)

    def __getitem__(self, idx):
        path = self.local_classes[idx]
        pil_cls = Image.open(path).convert("L").resize(
            (self.resolution, self.resolution), resample=Image.NEAREST
        )
        arr_cls = np.array(pil_cls)
        out = {"path": path, "label_ori": arr_cls.copy()}
        arr_cls = (arr_cls > 0).astype(arr_cls.dtype)
        out["label"] = arr_cls[None]
        placeholder = np.zeros((3, self.resolution, self.resolution), dtype=np.float32)
        return placeholder, out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _resize(pil_list, size, keep_aspect=True):
    pil_image, pil_class = pil_list
    while min(*pil_image.size) >= 2 * size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    if keep_aspect:
        scale = size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    else:
        pil_image = pil_image.resize((size, size), resample=Image.BICUBIC)
    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    return np.array(pil_image), np.array(pil_class)


def _center_crop(pil_list, size):
    pil_image, pil_class = pil_list
    while min(*pil_image.size) >= 2 * size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    arr = np.array(pil_image)
    cy = (arr.shape[0] - size) // 2
    cx = (arr.shape[1] - size) // 2
    s = np.s_[cy: cy + size, cx: cx + size]
    return arr[s], np.array(pil_class)[s]


def _random_crop(pil_list, size, min_frac=0.8, max_frac=1.0):
    min_dim = math.ceil(size / max_frac)
    max_dim = math.ceil(size / min_frac)
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

    arr = np.array(pil_image)
    cy = random.randrange(arr.shape[0] - size + 1)
    cx = random.randrange(arr.shape[1] - size + 1)
    s = np.s_[cy: cy + size, cx: cx + size]
    return arr[s], np.array(pil_class)[s]
