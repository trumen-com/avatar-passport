"""io: JPEG round trip with no metadata, EXIF orientation, HEIC, max_side, hashing. Synthetic images only."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from avtopassport.io import load_image, resize_max_side, save_jpeg, sha256_bytes, sha256_file


def synthetic(h: int = 120, w: int = 200) -> np.ndarray:
    """Smooth BGR gradients plus a red block near the top-left (BGR (0, 0, 255))."""
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), np.full_like(x, 128)], axis=2)
    img = img.astype(np.uint8)
    img[10:40, 10:50] = (0, 0, 255)
    return img


def jpeg_markers(data: bytes) -> list[int]:
    """Marker bytes of the JPEG header segments up to start-of-scan."""
    assert data[:2] == b"\xff\xd8"
    i, markers = 2, []
    while data[i] == 0xFF and data[i + 1] != 0xDA:
        markers.append(data[i + 1])
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    return markers


def test_round_trip_keeps_pixels_and_bgr_order(tmp_path: Path) -> None:
    img = synthetic()
    out = save_jpeg(img, tmp_path / "a" / "b" / "x.jpg", max_side=None)
    assert out == tmp_path / "a" / "b" / "x.jpg" and out.is_file()
    back = load_image(out)
    assert back.dtype == np.uint8 and back.shape == img.shape and back.flags["C_CONTIGUOUS"]
    assert np.abs(back.astype(int) - img.astype(int)).mean() < 4
    b, g, r = back[25, 30]
    assert r > 200 and b < 60 and g < 60
    assert [p.name for p in out.parent.iterdir()] == ["x.jpg"]  # no temp files left behind


def test_save_jpeg_strips_all_metadata(tmp_path: Path) -> None:
    src = tmp_path / "camera.jpg"
    exif = Image.Exif()
    exif[0x010F] = "SyntheticCam"  # Make
    exif[0x0132] = "2026:10:10 12:00:00"  # DateTime
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(synthetic()[:, :, ::-1]).save(src, exif=exif, icc_profile=icc)
    with Image.open(src) as im:
        assert len(im.getexif()) > 0 and "icc_profile" in im.info  # the source really has metadata

    out = save_jpeg(load_image(src), tmp_path / "clean.jpg")
    with Image.open(out) as im:
        assert len(im.getexif()) == 0
        assert not {"exif", "icc_profile", "xmp", "comment"} & set(im.info)
    markers = jpeg_markers(out.read_bytes())
    assert not [m for m in markers if 0xE1 <= m <= 0xEF or m == 0xFE]  # no APP1..APP15, no COM


def test_load_image_applies_exif_orientation(tmp_path: Path) -> None:
    stored = np.full((20, 40, 3), 40, np.uint8)  # landscape as stored, RGB
    stored[:8, :8] = (255, 0, 0)                   # red in the stored top-left
    exif = Image.Exif()
    exif[0x0112] = 6                               # display = stored rotated 90 deg clockwise
    p = tmp_path / "rotated.jpg"
    Image.fromarray(stored).save(p, exif=exif, quality=95)
    img = load_image(p)
    assert img.shape == (40, 20, 3)
    b, g, r = img[2, -3]                           # stored top-left -> display top-right
    assert r > 200 and b < 60 and g < 60


def test_load_heic(tmp_path: Path) -> None:
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()  # for the synthetic encode below; load_image registers it itself
    p = tmp_path / "photo.heic"
    try:
        Image.fromarray(np.full((64, 96, 3), (200, 30, 30), np.uint8)).save(p, quality=90)
    except (KeyError, OSError, ValueError) as exc:
        pytest.skip(f"no HEIC encoder in this pillow-heif build: {exc}")
    img = load_image(p)
    assert img.shape == (64, 96, 3) and img.dtype == np.uint8
    b, g, r = img[10, 10]
    assert r > 170 and b < 70


def test_load_png_with_alpha_gives_three_channels(tmp_path: Path) -> None:
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., 0], rgba[..., 3] = 255, 128
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a.png")
    img = load_image(tmp_path / "a.png")
    assert img.shape == (10, 12, 3) and tuple(img[0, 0]) == (0, 0, 255)


def test_max_side_downscales_never_upscales(tmp_path: Path) -> None:
    p = save_jpeg(synthetic(100, 200), tmp_path / "x.jpg", max_side=None)
    assert load_image(p, max_side=64).shape == (32, 64, 3)
    assert load_image(p, max_side=1000).shape == (100, 200, 3)

    big = save_jpeg(synthetic(1000, 2000), tmp_path / "big.jpg")  # default max_side=1280
    with Image.open(big) as im:
        assert im.size == (1280, 640)


def test_resize_max_side() -> None:
    img = synthetic()
    assert resize_max_side(img, None) is img
    assert resize_max_side(img, 500) is img
    assert resize_max_side(img, 100).shape == (60, 100, 3)
    with pytest.raises(ValueError):
        resize_max_side(img, 0)


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_image(bad)
    out = tmp_path / "out"
    for not_bgr8 in (np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.float32), np.zeros((0, 4, 3), np.uint8)):
        with pytest.raises(ValueError):
            save_jpeg(not_bgr8, out / "x.jpg")
    assert not out.exists()


def test_sha256_helpers(tmp_path: Path) -> None:
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    data = bytes(range(256)) * 5000  # > 1 MiB: several read chunks
    p = tmp_path / "blob.bin"
    p.write_bytes(data)
    assert sha256_file(p) == sha256_bytes(data) == hashlib.sha256(data).hexdigest()
