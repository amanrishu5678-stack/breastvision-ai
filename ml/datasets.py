"""PyTorch dataset that reads the split files written by prepare_dataset.py."""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from common import ARTIFACTS_DIR, DATASET_DIR, read_json


def load_splits(artifacts_dir=ARTIFACTS_DIR):
    path = Path(artifacts_dir) / "splits.json"
    if not path.exists():
        raise FileNotFoundError(
            "splits.json not found. Run `python prepare_dataset.py` first."
        )
    return read_json(path)


class ImageListDataset(Dataset):
    def __init__(self, items, transform, root=DATASET_DIR):
        self.items = items
        self.transform = transform
        self.root = Path(root)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        with Image.open(self.root / item["path"]) as img:
            img = img.convert("RGB")
        return self.transform(img), item["label"]
