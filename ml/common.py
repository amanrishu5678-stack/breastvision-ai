"""Shared constants and helpers for the ML pipeline.

Kept free of heavy imports (torch is imported lazily) so that simple scripts
such as prepare_dataset.py start fast.
"""
import json
import os
import random
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parent
DATASET_DIR = ML_DIR / "dataset"
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", ML_DIR / "artifacts"))

SEED = 42
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Folder names inside the BUSI dataset (alphabetical order = class index order)
CLASS_FOLDERS = ["benign", "malignant", "normal"]
CLASS_NAMES = ["Benign", "Malignant", "Normal"]

DATASET_NAME = "Breast Ultrasound Images Dataset (BUSI)"
DATASET_SOURCE = (
    "Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound "
    "images. Data in Brief, 2020. https://doi.org/10.1016/j.dib.2019.104863 "
    "(Kaggle mirror: aryashah2k/breast-ultrasound-images-dataset)"
)

MODEL_NAME = "EfficientNet-B0"
MODEL_VERSION = "1.0"


def set_seed(seed: int = SEED) -> None:
    """Make runs as reproducible as PyTorch allows."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size, mean, std, train=False):
    """Single source of truth for preprocessing.

    The SAME eval transform is used for validation, testing and the live API,
    so training and serving cannot drift apart.
    """
    from torchvision import transforms as T

    resize = T.Resize((image_size, image_size))
    normalize = T.Normalize(mean=mean, std=std)
    if train:
        return T.Compose(
            [
                resize,
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                normalize,
            ]
        )
    return T.Compose([resize, T.ToTensor(), normalize])


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
