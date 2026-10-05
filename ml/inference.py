"""Inference + Grad-CAM. Used by the Flask backend (and usable from the command line).

    python inference.py path/to/image.png
"""
import base64
import io
import sys
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from common import ARTIFACTS_DIR, build_transforms, read_json
from model_builder import build_model

OVERLAY_MAX_SIDE = 512


def _jet(v):
    """Simple jet colormap: v in [0,1] -> RGB uint8 (avoids a matplotlib dependency)."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


class Predictor:
    def __init__(self, artifacts_dir=ARTIFACTS_DIR):
        self.config = read_json(artifacts_dir / "config.json")
        self.class_names = read_json(artifacts_dir / "class_names.json")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(len(self.class_names), pretrained=False)
        state = torch.load(artifacts_dir / "model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.transform = build_transforms(
            self.config["image_size"], self.config["mean"], self.config["std"], train=False
        )
        self._lock = threading.Lock()

    def predict(self, image: Image.Image, explain: bool = True) -> dict:
        image = ImageOps.exif_transpose(image).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)

        with self._lock:
            if explain:
                with torch.enable_grad():
                    logits, cam = self._forward_with_cam(x)
            else:
                with torch.no_grad():
                    logits = self.model(x)
                cam = None
            probs = torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()

        idx = int(probs.argmax())
        result = {
            "class": self.class_names[idx],
            "class_index": idx,
            "probability": float(probs[idx]),
            "probabilities": {n: float(p) for n, p in zip(self.class_names, probs)},
        }
        if cam is not None:
            result["explainability"] = {
                "method": "Grad-CAM",
                "target_class": self.class_names[idx],
                "overlay": self._overlay(image, cam),
            }
        return result

    def _forward_with_cam(self, x):
        """Grad-CAM on the last convolutional feature map of EfficientNet-B0.

        Same forward path as torchvision's EfficientNet (features -> avgpool ->
        flatten -> classifier), so the probabilities match the normal forward pass.
        """
        feats = self.model.features(x)  # (1, 1280, 7, 7)
        logits = self.model.classifier(torch.flatten(self.model.avgpool(feats), 1))
        idx = int(logits.argmax(dim=1))
        grads = torch.autograd.grad(logits[0, idx], feats)[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feats).sum(dim=1))[0]  # (7, 7)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return logits, cam.detach().cpu().numpy()

    @staticmethod
    def _overlay(image: Image.Image, cam: np.ndarray) -> str:
        w, h = image.size
        scale = min(1.0, OVERLAY_MAX_SIDE / max(w, h))
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        base = image.resize(size, Image.BILINEAR)
        heat = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BICUBIC)
        heat_rgb = _jet(np.asarray(heat).astype(np.float32) / 255.0)
        blended = (0.6 * np.asarray(base, dtype=np.float32) + 0.4 * heat_rgb).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python inference.py path/to/image.png")
    out = Predictor().predict(Image.open(sys.argv[1]), explain=False)
    print(out["class"], f"{out['probability']:.3f}")
    for k, v in out["probabilities"].items():
        print(f"  {k}: {v:.3f}")
