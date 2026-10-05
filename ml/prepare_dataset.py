"""Step 1: scan the BUSI dataset, clean it, and create a stratified split.

Usage (from the ml/ folder):
    python prepare_dataset.py

Expected folder layout after unzipping the Kaggle download into ml/dataset/:
    ml/dataset/Dataset_BUSI_with_GT/benign/benign (1).png ...
    ml/dataset/Dataset_BUSI_with_GT/malignant/...
    ml/dataset/Dataset_BUSI_with_GT/normal/...
(any nesting depth is fine as long as the three class folders exist)

What it does:
  * ignores the *_mask*.png segmentation masks (they are not classification images)
  * skips unreadable/corrupt images
  * removes byte-identical duplicate files
  * makes a stratified 70/15/15 train/val/test split with a fixed seed
  * writes artifacts/splits.json and artifacts/dataset_stats.json
"""
import hashlib
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

from common import (
    ARTIFACTS_DIR,
    CLASS_FOLDERS,
    CLASS_NAMES,
    DATASET_DIR,
    DATASET_NAME,
    DATASET_SOURCE,
    SEED,
    write_json,
)

IMG_EXT = {".png", ".jpg", ".jpeg"}
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15


def find_dataset_root():
    """Find the folder that directly contains benign/malignant/normal."""
    if not DATASET_DIR.exists():
        return None
    candidates = [DATASET_DIR] + [p for p in DATASET_DIR.rglob("*") if p.is_dir()]
    for cand in candidates:
        names = {p.name.lower() for p in cand.iterdir() if p.is_dir()}
        if all(f in names for f in CLASS_FOLDERS):
            return cand
    return None


def class_dir(root: Path, folder: str) -> Path:
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() == folder:
            return p
    raise FileNotFoundError(folder)


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = find_dataset_root()
    if root is None:
        raise SystemExit(
            "\nDataset not found.\n"
            "Download BUSI from Kaggle (aryashah2k/breast-ultrasound-images-dataset),\n"
            "unzip it, and place the folder inside ml/dataset/ so that these exist:\n"
            "  ml/dataset/<any folder>/benign\n"
            "  ml/dataset/<any folder>/malignant\n"
            "  ml/dataset/<any folder>/normal\n"
        )
    print(f"Dataset root: {root}")

    records, corrupt, masks_skipped, duplicates = [], [], 0, 0
    seen_hashes = {}
    sizes = []

    for label, folder in enumerate(CLASS_FOLDERS):
        for path in sorted(class_dir(root, folder).iterdir()):
            if path.suffix.lower() not in IMG_EXT:
                continue
            if "_mask" in path.stem.lower():
                masks_skipped += 1
                continue
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    width, height = im.size
                    im.load()
            except Exception:
                corrupt.append(path.name)
                continue
            digest = file_md5(path)
            if digest in seen_hashes:
                duplicates += 1
                continue
            seen_hashes[digest] = path.name
            sizes.append((width, height))
            records.append(
                {"path": path.relative_to(DATASET_DIR).as_posix(), "label": label}
            )

    if len(records) < 30:
        raise SystemExit(f"Only {len(records)} usable images found - check the dataset folder.")

    labels = [r["label"] for r in records]
    train_val, test = train_test_split(
        records, test_size=TEST_FRAC, stratify=labels, random_state=SEED
    )
    val_size = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=[r["label"] for r in train_val],
        random_state=SEED,
    )
    splits = {"train": train, "val": val, "test": test}

    def dist(items):
        c = Counter(r["label"] for r in items)
        return {CLASS_NAMES[i]: c.get(i, 0) for i in range(len(CLASS_NAMES))}

    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    stats = {
        "dataset_name": DATASET_NAME,
        "source": DATASET_SOURCE,
        "license_note": (
            "BUSI is publicly available for research use. Check the Kaggle page and the "
            "original paper for the current license/citation terms before redistributing."
        ),
        "total_images": len(records),
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "class_distribution": dist(records),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_distribution": {k: dist(v) for k, v in splits.items()},
        "split_strategy": "Stratified 70/15/15 (train/val/test), seed %d" % SEED,
        "image_size_original": {
            "width_min": min(widths),
            "width_max": max(widths),
            "width_median": statistics.median(widths),
            "height_min": min(heights),
            "height_max": max(heights),
            "height_median": statistics.median(heights),
        },
        "cleaning": {
            "mask_files_ignored": masks_skipped,
            "corrupt_files_skipped": len(corrupt),
            "exact_duplicates_removed": duplicates,
        },
    }

    write_json(ARTIFACTS_DIR / "splits.json", {"seed": SEED, "classes": CLASS_NAMES, **splits})
    write_json(ARTIFACTS_DIR / "dataset_stats.json", stats)

    print(f"\nUsable images: {len(records)}  (masks ignored: {masks_skipped}, "
          f"corrupt: {len(corrupt)}, duplicates removed: {duplicates})")
    print("Class distribution:", stats["class_distribution"])
    for k, v in splits.items():
        print(f"  {k:5s}: {len(v):4d}  {dist(v)}")
    print(f"\nSaved to {ARTIFACTS_DIR}. Next: python train.py")


if __name__ == "__main__":
    main()
