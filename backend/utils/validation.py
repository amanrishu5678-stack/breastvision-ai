"""Upload validation. Images are handled in memory only - nothing is written to disk."""
import io

from PIL import Image, UnidentifiedImageError

import config
from utils.errors import ApiError

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS

FRIENDLY_TYPES = "JPG, JPEG, or PNG"


def validate_and_open_image(file_storage) -> Image.Image:
    if file_storage is None or not file_storage.filename:
        raise ApiError("NO_FILE", "No image was uploaded. Send the file in the 'image' field.", 400)

    filename = file_storage.filename
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in config.ALLOWED_EXTENSIONS:
        raise ApiError("INVALID_TYPE", f"Please upload a valid {FRIENDLY_TYPES} image.", 415)

    mime = (file_storage.mimetype or "").lower()
    if mime not in config.ALLOWED_MIME_TYPES:
        raise ApiError("INVALID_TYPE", f"Please upload a valid {FRIENDLY_TYPES} image.", 415)

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    data = file_storage.stream.read(max_bytes + 1)
    if len(data) == 0:
        raise ApiError("EMPTY_FILE", "The uploaded file is empty.", 400)
    if len(data) > max_bytes:
        raise ApiError("FILE_TOO_LARGE", f"The image is larger than {config.MAX_UPLOAD_MB} MB.", 413)

    try:
        probe = Image.open(io.BytesIO(data))
        if probe.format not in config.ALLOWED_PIL_FORMATS:
            raise ApiError("INVALID_TYPE", f"Please upload a valid {FRIENDLY_TYPES} image.", 415)
        probe.verify()  # detects truncated / corrupt files
        image = Image.open(io.BytesIO(data))
        image.load()
    except ApiError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError):
        raise ApiError("UNREADABLE_IMAGE", "The file could not be read as an image. It may be corrupted.", 422)

    if min(image.size) < config.MIN_IMAGE_SIDE:
        raise ApiError(
            "IMAGE_TOO_SMALL",
            f"The image is too small. Each side must be at least {config.MIN_IMAGE_SIDE} pixels.",
            422,
        )
    return image.convert("RGB")
