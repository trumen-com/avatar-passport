"""Image I/O. Contract: every image in the codebase is np.uint8, shape HxWx3, BGR, display orientation.

- load_image: Pillow (+ pillow-heif for HEIC/HEIF), EXIF orientation applied, converted to BGR uint8.
- save_jpeg: OpenCV encoder -> bytes -> file. The JPEG carries no EXIF/XMP/ICC metadata at all.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from functools import cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


@cache
def _register_heif() -> bool:
    """Register the HEIC/HEIF opener with Pillow once per process. False if pillow-heif is missing."""
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return False
    register_heif_opener()
    return True


def check_bgr8(img: np.ndarray, what: str = "image") -> None:
    """Raise ValueError unless ``img`` is a non-empty np.uint8 HxWx3 array."""
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        desc = f"{img.dtype} {img.shape}" if isinstance(img, np.ndarray) else type(img).__name__
        raise ValueError(f"{what} must be np.uint8 HxWx3 BGR, got {desc}")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError(f"{what} is empty: {img.shape}")


def load_image(path: str | Path, *, max_side: int | None = None) -> np.ndarray:
    """Load JPEG/PNG/HEIC (via pillow-heif) as uint8 BGR, EXIF-transposed, optionally downscaled
    so the long side is <= max_side (INTER_AREA). Raises FileNotFoundError / ValueError."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"image not found: {p}")
    _register_heif()
    try:
        with Image.open(p) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as exc:
        raise ValueError(f"cannot decode image {p}: {exc}") from exc
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return resize_max_side(bgr, max_side)


def resize_max_side(img: np.ndarray, max_side: int | None) -> np.ndarray:
    """Downscale (never upscale) so max(h, w) <= max_side. None -> unchanged copy-free return."""
    if max_side is None:
        return img
    if max_side < 1:
        raise ValueError(f"max_side must be >= 1, got {max_side}")
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    s = max_side / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def save_jpeg(img: np.ndarray, path: str | Path, *, max_side: int | None = 1280, quality: int = 88) -> Path:
    """Write a JPEG with NO metadata (no EXIF/XMP/ICC), creating parent dirs. Returns the path.
    Callers must only pass images that already went through privacy anonymisation."""
    check_bgr8(img)
    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be 1..100, got {quality}")
    ok, buf = cv2.imencode(".jpg", resize_max_side(img, max_side), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError(f"JPEG encoding failed for {path}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # write to a temp file in the same directory, then rename: no half-written JPEGs on a crash
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(buf.tobytes())
        os.chmod(tmp, 0o644)  # mkstemp creates 0600
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return p


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of a bytes object."""
    return hashlib.sha256(data).hexdigest()
