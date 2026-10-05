import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
ML_DIR = BACKEND_DIR.parent / "ml"
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", ML_DIR / "artifacts"))

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]
PORT = int(os.environ.get("PORT", "5000"))
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/pjpeg"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}
MIN_IMAGE_SIDE = 64
MAX_IMAGE_PIXELS = 25_000_000
