"""Step 3: evaluate the saved model on the HELD-OUT TEST SET.

Usage (from the ml/ folder):
    python evaluate.py

Run this after you have finished training. Do not use test results to tune the
model - that would make the numbers optimistic.

Writes: artifacts/metrics.json, confusion_matrix.png, roc_curve.png, training_history.png
"""
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import ARTIFACTS_DIR, build_transforms, read_json, write_json
from datasets import ImageListDataset, load_splits
from metrics_utils import compute_metrics
from model_builder import build_model


def main():
    cfg = read_json(ARTIFACTS_DIR / "config.json")
    class_names = read_json(ARTIFACTS_DIR / "class_names.json")
    splits = load_splits()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(class_names), pretrained=False)
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "model.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()

    tf = build_transforms(cfg["image_size"], cfg["mean"], cfg["std"], train=False)
    loader = DataLoader(ImageListDataset(splits["test"], tf), batch_size=32, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            all_probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
            all_labels.append(y.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    metrics = compute_metrics(labels, probs, class_names)
    metrics["split"] = "test"
    metrics["split_description"] = "Held-out test set (never used for training or model selection)"
    metrics["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(ARTIFACTS_DIR / "metrics.json", metrics)

    # ---- console summary
    print(f"\nTEST SET results (n={metrics['n_samples']})")
    print(f"  accuracy         {metrics['accuracy']:.4f}")
    for k, v in metrics["macro"].items():
        print(f"  macro {k:11s}{v}")
    for name, m in metrics["per_class"].items():
        print(f"  {name:10s} P={m['precision']:.3f} R={m['recall']:.3f} Spec={m['specificity']:.3f} F1={m['f1']:.3f}")

    # ---- plots
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, c in metrics["roc_curves"].items():
        ax.plot(c["fpr"], c["tpr"], label=f"{name} (AUC {c['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves, one-vs-rest (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    hist = read_json(ARTIFACTS_DIR / "history.json")
    ep = [h["epoch"] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(ep, [h["train_loss"] for h in hist], label="train")
    a1.plot(ep, [h["val_loss"] for h in hist], label="validation")
    a1.set_title("Loss"); a1.set_xlabel("Epoch"); a1.legend()
    a2.plot(ep, [h["train_acc"] for h in hist], label="train")
    a2.plot(ep, [h["val_acc"] for h in hist], label="validation")
    a2.set_title("Accuracy"); a2.set_xlabel("Epoch"); a2.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "training_history.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved metrics.json and plots to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
