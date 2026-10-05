"""Step 2: train EfficientNet-B0 with transfer learning.

Usage (from the ml/ folder):
    python train.py
    python train.py --epochs-head 5 --epochs-finetune 15 --batch-size 16

Phase 1 (head): backbone frozen, only the new classifier layer is trained.
Phase 2 (finetune): the last EfficientNet blocks are unfrozen and trained with
                    a lower learning rate.
The checkpoint with the lowest VALIDATION loss is saved. The test set is not
touched here - it is only used by evaluate.py.
"""
import argparse
import copy
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    ARTIFACTS_DIR,
    CLASS_NAMES,
    DATASET_NAME,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_NAME,
    MODEL_VERSION,
    SEED,
    build_transforms,
    set_seed,
    write_json,
)
from datasets import ImageListDataset, load_splits
from model_builder import build_model


def run_epoch(model, loader, criterion, optimizer, device, train, head_only):
    model.train(train)
    if train and head_only:
        model.features.eval()  # keep frozen BatchNorm statistics fixed
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-finetune", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5, help="early stopping (fine-tune phase)")
    ap.add_argument("--num-workers", type=int, default=0, help="keep 0 on Windows")
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits = load_splits()
    train_ds = ImageListDataset(splits["train"], build_transforms(IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, train=True))
    val_ds = ImageListDataset(splits["val"], build_transforms(IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, train=False))
    gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # Class imbalance: weight each class inversely to its frequency in TRAIN only
    counts = torch.zeros(len(CLASS_NAMES))
    for item in splits["train"]:
        counts[item["label"]] += 1
    class_weights = counts.sum() / (len(CLASS_NAMES) * counts)
    print("Train class counts:", counts.tolist(), "-> loss weights:", [round(w, 3) for w in class_weights.tolist()])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model = build_model(len(CLASS_NAMES), pretrained=True).to(device)
    history, best = [], {"val_loss": float("inf"), "state": None, "epoch": 0}
    epoch_counter, start = 0, time.time()

    def run_phase(name, epochs, lr, head_only):
        nonlocal epoch_counter
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        if not head_only:
            for block in list(model.features)[6:]:
                for p in block.parameters():
                    p.requires_grad = True
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        stale = 0
        for _ in range(epochs):
            epoch_counter += 1
            tl, ta = run_epoch(model, train_loader, criterion, optimizer, device, True, head_only)
            vl, va = run_epoch(model, val_loader, criterion, None, device, False, head_only)
            history.append({"epoch": epoch_counter, "phase": name,
                            "train_loss": round(tl, 5), "train_acc": round(ta, 5),
                            "val_loss": round(vl, 5), "val_acc": round(va, 5)})
            improved = vl < best["val_loss"]
            if improved:
                best.update(val_loss=vl, epoch=epoch_counter, state=copy.deepcopy(model.state_dict()))
                stale = 0
            else:
                stale += 1
            print(f"[{name:8s}] epoch {epoch_counter:2d} | train loss {tl:.4f} acc {ta:.3f} | "
                  f"val loss {vl:.4f} acc {va:.3f}{'  *saved' if improved else ''}")
            if not head_only and stale >= args.patience:
                print("Early stopping.")
                break

    run_phase("head", args.epochs_head, args.lr_head, head_only=True)
    run_phase("finetune", args.epochs_finetune, args.lr_finetune, head_only=False)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], ARTIFACTS_DIR / "model.pt")
    write_json(ARTIFACTS_DIR / "class_names.json", CLASS_NAMES)
    write_json(ARTIFACTS_DIR / "history.json", history)
    write_json(ARTIFACTS_DIR / "config.json", {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "framework": f"PyTorch {torch.__version__}",
        "pretrained_weights": "ImageNet (torchvision EfficientNet_B0_Weights.DEFAULT)",
        "dataset_name": DATASET_NAME,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "seed": SEED,
        "batch_size": args.batch_size,
        "epochs_head": args.epochs_head,
        "epochs_finetune": args.epochs_finetune,
        "lr_head": args.lr_head,
        "lr_finetune": args.lr_finetune,
        "optimizer": "AdamW (weight decay 1e-4)",
        "loss": "Cross-entropy with inverse-frequency class weights",
        "class_weights": [round(w, 4) for w in class_weights.tolist()],
        "augmentation": "Random horizontal flip, rotation up to 10 degrees, brightness/contrast jitter 0.2 (training only)",
        "preprocessing": f"Resize to {IMAGE_SIZE}x{IMAGE_SIZE}, scale to [0,1], ImageNet mean/std normalization",
        "best_epoch": best["epoch"],
        "best_val_loss": round(best["val_loss"], 5),
        "epochs_run": epoch_counter,
        "device": str(device),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": round(time.time() - start, 1),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"\nBest epoch: {best['epoch']} (val loss {best['val_loss']:.4f}). Saved to {ARTIFACTS_DIR}.")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
