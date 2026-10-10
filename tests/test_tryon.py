"""tryon.recolor: LAB recolouring, outside-mask identity, body-panel mask. Synthetic images only, CPU, offline."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from avtopassport.paths import REFERENCE
from avtopassport.schema import Segment
from avtopassport.tryon.recolor import (
    PANEL_EXCLUDE_KEYWORDS,
    body_mask,
    hex_to_lab,
    load_palette,
    recolor,
    recolor_all,
)

RED = "#C03028"
BLUE = "#2A3678"
GREEN = "#028377"


def lab_of(img: np.ndarray) -> np.ndarray:
    """Float LAB (L 0..100) of a uint8 BGR image."""
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def grey_car(h: int = 120, w: int = 160, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Noisy textured background with a shaded grey 'panel' (vertical gradient) in a central rectangle."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[30:90, 40:120] = True
    shade = np.linspace(80, 170, 60, dtype=np.float64)[:, None] + rng.normal(0, 3, (60, 80))
    grey = np.clip(np.rint(shade), 0, 255).astype(np.uint8)
    img[30:90, 40:120] = grey[..., None]
    return img, mask


def seg(cls: str, mask: np.ndarray | None, box: tuple[float, float, float, float] = (0, 0, 0, 0)) -> Segment:
    area = int(mask.sum()) if mask is not None else int((box[2] - box[0]) * (box[3] - box[1]))
    return Segment(cls=cls, score=0.9, box=box, mask=mask, area_px=area)


# ------------------------------------------------------------------------------------------------ recolor

@pytest.mark.parametrize("kw", [
    {},
    {"feather_px": 0},
    {"feather_px": 8},
    {"chroma_gain": 1.6, "luminance_mix": 0.7},
])
@pytest.mark.parametrize("target", [RED, BLUE, "#FFFFFF", "#1A232C"])
def test_outside_mask_bit_identical(kw: dict, target: str) -> None:
    img, mask = grey_car()
    before = img.copy()
    out = recolor(img, mask, target, **kw)
    assert out.dtype == np.uint8 and out.shape == img.shape
    np.testing.assert_array_equal(out[~mask], img[~mask])
    np.testing.assert_array_equal(img, before)  # input not modified
    if target != "#FFFFFF" or kw.get("luminance_mix"):  # white on a neutral grey with L kept is a no-op
        assert np.any(out[mask] != img[mask])


@pytest.mark.parametrize("target", [RED, BLUE, GREEN, "#FFFFFF", "#1A232C"])
@pytest.mark.parametrize("feather_px", [0, 3])
def test_l_channel_preserved_inside_mask(target: str, feather_px: int) -> None:
    img, mask = grey_car()
    out = recolor(img, mask, target, feather_px=feather_px)
    dl = np.abs(lab_of(out)[..., 0] - lab_of(img)[..., 0])[mask]
    assert dl.max() < 1.0, dl.max()  # uint8 rounding only


@pytest.mark.parametrize("target,a_sign,b_sign", [
    (RED, +1, +1),     # red: +a, +b
    (BLUE, +1, -1),    # midnight blue: slightly +a, strongly -b
    (GREEN, -1, 0),    # emerald: -a
])
def test_grey_patch_hue_moves_toward_target(target: str, a_sign: int, b_sign: int) -> None:
    img, mask = grey_car()
    lab_in, lab_out = lab_of(img)[mask], lab_of(recolor(img, mask, target, feather_px=0))[mask]
    assert np.abs(lab_in[:, 1:]).max() < 1.0  # the patch starts neutral
    a, b = float(np.median(lab_out[:, 1])), float(np.median(lab_out[:, 2]))
    _, ta, tb = hex_to_lab(target)
    if a_sign:
        assert np.sign(a) == a_sign and abs(a) > 0.5 * abs(ta)
    if b_sign:
        assert np.sign(b) == b_sign and abs(b) > 0.5 * abs(tb)


def test_chroma_gain_scales_saturation() -> None:
    img, mask = grey_car()
    chroma = []
    for g in (0.0, 0.5, 1.0):
        lab = lab_of(recolor(img, mask, RED, chroma_gain=g, feather_px=0))[mask]
        chroma.append(float(np.median(np.hypot(lab[:, 1], lab[:, 2]))))
    assert chroma[0] < 1.0 < chroma[1] < chroma[2]


def test_white_car_default_is_pastel_and_luminance_mix_darkens() -> None:
    img = np.full((40, 40, 3), 235, dtype=np.uint8)
    mask = np.ones((40, 40), dtype=bool)
    t_l = hex_to_lab(BLUE)[0]
    keep = lab_of(recolor(img, mask, BLUE, feather_px=0))[mask]
    assert keep[:, 0].min() > 90 and np.median(keep[:, 2]) < -2  # light, blue-ish tint
    dark = lab_of(recolor(img, mask, BLUE, luminance_mix=1.0, feather_px=0))[mask]
    assert abs(float(np.median(dark[:, 0])) - t_l) < 2.0
    assert np.median(dark[:, 2]) < 0.5 * hex_to_lab(BLUE)[2]


def test_specular_white_stays_white() -> None:
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    out = recolor(img, np.ones((10, 10), dtype=bool), RED, feather_px=0)
    assert out.min() >= 250  # chroma is gamut-limited at L=100 instead of clipping to pink


def test_feather_ramps_inside_and_never_touches_far_outside() -> None:
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True
    out = recolor(img, mask, RED, feather_px=6)
    np.testing.assert_array_equal(out[~mask], img[~mask])
    np.testing.assert_array_equal(out[:25], img[:25])  # far outside
    chroma = np.hypot(*np.moveaxis(lab_of(out)[..., 1:], -1, 0))
    edge, inner, centre = chroma[50, 30], chroma[50, 33], chroma[50, 50]
    assert 0 < edge < inner < centre
    hard = np.hypot(*np.moveaxis(lab_of(recolor(img, mask, RED, feather_px=0))[..., 1:], -1, 0))
    assert abs(hard[50, 30] - hard[50, 50]) < 1.0


def test_empty_mask_and_bad_inputs() -> None:
    img, _ = grey_car()
    empty = np.zeros(img.shape[:2], dtype=bool)
    np.testing.assert_array_equal(recolor(img, empty, RED), img)
    with pytest.raises(ValueError):
        recolor(img, empty, "red")
    with pytest.raises(ValueError):
        recolor(img, np.ones((5, 5), dtype=bool), RED)
    with pytest.raises(ValueError):
        recolor(img.astype(np.float32), empty, RED)
    _, mask = grey_car()
    with pytest.raises(ValueError):
        recolor(img, mask, RED, luminance_mix=1.5)


# ------------------------------------------------------------------------------------------------ palette

def test_load_palette_matches_json() -> None:
    raw = json.loads((REFERENCE / "colours.json").read_text(encoding="utf-8"))["colours"]
    pal = load_palette()
    assert pal == raw and len(pal) >= 6
    for c in pal:
        assert {"name", "model", "hex", "source_url"} <= set(c)


def test_recolor_all_one_image_per_colour_and_matches_recolor() -> None:
    img, mask = grey_car()
    pal = load_palette()
    res = recolor_all(img, mask, pal, feather_px=2, chroma_gain=1.2)
    assert [r["name"] for r in res] == [c["name"] for c in pal]
    assert [r["hex"] for r in res] == [c["hex"] for c in pal]
    for r, c in zip(res, pal):
        assert r["image"].shape == img.shape and r["image"].dtype == np.uint8
        np.testing.assert_array_equal(r["image"], recolor(img, mask, c["hex"], feather_px=2, chroma_gain=1.2))
        np.testing.assert_array_equal(r["image"][~mask], img[~mask])
    with pytest.raises(TypeError):
        recolor_all(img, mask, pal, bogus=1)


# ------------------------------------------------------------------------------------------------ body mask

def test_body_mask_excludes_glass_wheels_lights_plate_grille() -> None:
    h, w = 100, 200

    def rect(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        m = np.zeros((h, w), dtype=bool)
        m[y0:y1, x0:x1] = True
        return m

    door = rect(20, 80, 50, 150)
    window = rect(20, 45, 60, 140)  # overlaps the door mask: must be cut out
    parts = [
        seg("left_front-door", door),
        seg("right_back-window", window),
        seg("windshield", rect(0, 10, 0, 40)),
        seg("back-windshield", rect(0, 10, 160, 200)),
        seg("left_front-wheel", rect(70, 100, 0, 30)),
        seg("left_headlight", rect(50, 60, 0, 20)),
        seg("right_tail-light", rect(50, 60, 180, 200)),
        seg("license-plate", rect(85, 95, 90, 110)),
        seg("grille", rect(60, 70, 20, 45)),
        seg("left_mirror", rect(10, 18, 150, 160)),
        seg("Hood", rect(0, 15, 50, 150)),
        seg("left_fender", None, box=(160.2, 20.0, 180.0, 40.9)),  # detection-only: box footprint
    ]
    m = body_mask(parts, (h, w))
    assert m.dtype == bool and m.shape == (h, w)
    assert m[60:80, 50:150].all()                 # door below the window
    assert not m[20:45, 60:140].any()             # window cut out of the door
    assert m[10:18, 150:160].all()                # mirror kept (body colour)
    assert m[0:15, 50:150][~window[0:15, 50:150]].all()
    assert m[20:41, 160:180].all() and not m[41:, 160:180].any()
    for excluded in (rect(0, 10, 0, 40), rect(0, 10, 160, 200), rect(70, 100, 0, 30), rect(50, 60, 0, 20),
                     rect(50, 60, 180, 200), rect(85, 95, 90, 110), rect(60, 70, 20, 45)):
        assert not m[excluded].any()


def test_body_mask_keywords_and_resize() -> None:
    for cls in ("front_glass", "back_light", "front-wheel", "fog-lamp", "license_plate", "Grille", "rear-tyre"):
        assert any(k in cls.lower() for k in PANEL_EXCLUDE_KEYWORDS), cls
    for cls in ("hood", "front-bumper", "left_quarter-panel", "roof", "trunk", "left_mirror", "rocker-panel"):
        assert not any(k in cls.lower() for k in PANEL_EXCLUDE_KEYWORDS), cls
    small = np.zeros((50, 100), dtype=bool)
    small[10:40, 20:80] = True
    m = body_mask([seg("hood", small)], (100, 200))
    assert m.shape == (100, 200) and m[20:80, 40:160].all() and not m[:20].any()
    assert not body_mask([], (10, 10)).any()
