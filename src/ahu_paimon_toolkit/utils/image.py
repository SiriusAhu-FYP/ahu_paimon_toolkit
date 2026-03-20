"""Image encoding, MIME detection, and batch-loading utilities."""

from __future__ import annotations

import base64
from pathlib import Path

from loguru import logger as lg

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def encode_image(path: Path) -> str:
    """Encode an image file to a Base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(path: Path) -> str:
    """Return MIME type based on file extension."""
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
    }
    return mime_map.get(path.suffix.lower(), "image/png")


def load_images(
    assets_dir: Path,
) -> list[tuple[str, str, str, Path]]:
    """Load all supported images from a directory.

    Supports both flat layout (``assets_dir/*.png``) and per-asset
    sub-folder layout (``assets_dir/01_Name/01_Name.png``).

    Returns: [(filename, base64_data, mime_type, file_path), ...]
    """
    images: list[tuple[str, str, str, Path]] = []

    candidates: list[Path] = []
    for entry in sorted(assets_dir.iterdir()):
        if entry.is_dir():
            candidates.extend(sorted(entry.iterdir()))
        elif entry.is_file():
            candidates.append(entry)

    for p in candidates:
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            b64 = encode_image(p)
            mime = get_image_mime(p)
            images.append((p.name, b64, mime, p))
            lg.debug("Loaded image: {} ({:.1f} KB)", p.name, len(b64) * 3 / 4 / 1024)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {assets_dir}")
    lg.info("Loaded {} images", len(images))
    return images
