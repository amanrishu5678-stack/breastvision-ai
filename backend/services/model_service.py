"""Loads the trained model once and exposes saved artifacts (config, metrics, history)."""
import json
import logging
import sys
from datetime import datetime, timezone

import config

log = logging.getLogger(__name__)

REQUIRED_FILES = ("model.pt", "config.json", "class_names.json")
TRAIN_HINT = (
    "The model has not been trained yet. From the ml/ folder run: "
    "python prepare_dataset.py, python train.py, python evaluate.py."
)


class ModelService:
    def __init__(self, artifacts_dir=config.ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.predictor = None
        self.load_error = None

    # ---- model loading (called once at startup)
    def load(self):
        missing = [f for f in REQUIRED_FILES if not (self.artifacts_dir / f).exists()]
        if missing:
            self.load_error = TRAIN_HINT
            log.warning("Model artifacts missing: %s", missing)
            return
        try:
            if str(config.ML_DIR) not in sys.path:
                sys.path.append(str(config.ML_DIR))  # append: never shadow backend modules
            from inference import Predictor  # imports torch

            self.predictor = Predictor(self.artifacts_dir)
            self.load_error = None
            log.info("Model loaded: %s", self.predictor.config.get("model_name"))
        except ImportError as exc:
            self.load_error = (
                "PyTorch is not installed in this Python environment. "
                "Run: pip install -r backend/requirements.txt"
            )
            log.error("Import failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            self.load_error = "The model files could not be loaded. Re-run training."
            log.exception("Model load failed: %s", exc)

    @property
    def loaded(self):
        return self.predictor is not None

    # ---- artifact readers
    def _read(self, name):
        path = self.artifacts_dir / name
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def model_info(self):
        cfg = self._read("config.json")
        if cfg is None:
            return None
        keys = [
            "model_name", "model_version", "framework", "pretrained_weights", "num_classes",
            "class_names", "image_size", "preprocessing", "augmentation", "optimizer", "loss",
            "batch_size", "epochs_head", "epochs_finetune", "lr_head", "lr_finetune", "seed",
            "best_epoch", "epochs_run", "num_parameters", "trained_at",
        ]
        return {
            "model": {k: cfg[k] for k in keys if k in cfg},
            "dataset": self._read("dataset_stats.json"),
            "loaded": self.loaded,
        }

    def model_metrics(self):
        metrics = self._read("metrics.json")
        if metrics is None:
            return None
        return {
            "metrics": metrics,
            "history": self._read("history.json") or [],
            "dataset": self._read("dataset_stats.json"),
        }

    def analyze(self, image, explain=True):
        cfg = self.predictor.config
        prediction = self.predictor.predict(image, explain=explain)
        return {
            "success": True,
            "prediction": prediction,
            "model": {"name": cfg.get("model_name"), "version": cfg.get("model_version")},
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }
