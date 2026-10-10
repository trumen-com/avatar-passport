"""Paint try-on: recolour the car's body panels in CIE LAB, keeping each pixel's lightness (CLAUDE.md section 4).

Flow: ``body_mask`` (part segments minus glass / wheels / lights / plate / grille) -> ``recolor`` (replace a/b
with the target colour's a/b, keep L, feather the edge inwards) -> ``recolor_all`` over ``load_palette()``.

Images are uint8 (H, W, 3) BGR. Masks are bool (H, W). Pixels outside the mask are never modified.
Colours come from ``data/reference/colours.json``: website swatches, not factory paint codes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from avtopassport.paths import REFERENCE
from avtopassport.schema import Segment

# Substrings (case-insensitive) of part classes that are NOT painted body panels. Covers the mitbersh parts
# model ("windshield", "back-window", "front-wheel", "headlight", "tail-light", "license-plate", "grille") and
# Ultralytics carparts-seg style names ("front_glass", "back_light"). Mirrors are deliberately kept: on the BYD /
# Toyota cars we target, mirror caps are usually body colour. Rocker panels and bumpers are kept as well.
PANEL_EXCLUDE_KEYWORDS: tuple[str, ...] = (
    "window", "windshield", "glass",
    "wheel", "tire", "tyre",
    "light", "lamp",
    "plate",
    "grille",
)

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")
_REQUIRED_KEYS = ("name", "model", "hex", "source_url")

# OpenCV's float Lab uses D65 with these white-point values (cv2 colour-conversion docs).
_XN, _ZN = 0.950456, 1.088754
_XYZ2RGB = np.array([[3.240479, -1.53715, -0.498535],
                     [-0.969256, 1.875991, 0.041556],
                     [0.055648, -0.204043, 1.057311]], dtype=np.float64)
_L_GRID = np.linspace(0.0, 100.0, 1001)


# ---------------------------------------------------------------------------------------------------- palette

def load_palette(path: Path = REFERENCE / "colours.json") -> list[dict[str, Any]]:
    """Return the ``colours`` entries of colours.json as stored (name, model, hex, source_url, ...)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    colours = data["colours"] if isinstance(data, dict) else data
    out: list[dict[str, Any]] = []
    for i, c in enumerate(colours):
        missing = [k for k in _REQUIRED_KEYS if k not in c]
        if missing:
            raise ValueError(f"{path}: colour #{i} is missing {missing}")
        hex_to_bgr(c["hex"])  # validates the hex string
        out.append(dict(c))
    return out


def hex_to_bgr(target_hex: str) -> tuple[int, int, int]:
    """Parse '#RRGGBB' (or 'RRGGBB') into a uint8 (B, G, R) tuple."""
    m = _HEX_RE.match(target_hex.strip())
    if not m:
        raise ValueError(f"not a #RRGGBB colour: {target_hex!r}")
    v = m.group(1)
    r, g, b = (int(v[k:k + 2], 16) for k in (0, 2, 4))
    return b, g, r


def hex_to_lab(target_hex: str) -> tuple[float, float, float]:
    """CIE LAB (L in 0..100, a/b roughly -128..127) of a hex colour, as computed by OpenCV's float path."""
    bgr = np.array([[hex_to_bgr(target_hex)]], dtype=np.float32) / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)[0, 0]
    return float(lab[0]), float(lab[1]), float(lab[2])


# ---------------------------------------------------------------------------------------------------- body mask

def _is_excluded(cls: str) -> bool:
    name = cls.lower()
    return any(k in name for k in PANEL_EXCLUDE_KEYWORDS)


def _footprint(seg: Segment, shape: tuple[int, int]) -> np.ndarray:
    """Segment mask in the (H, W) grid (nearest-neighbour resize if needed), or its box when mask is None."""
    h, w = shape
    if seg.mask is not None:
        m = np.asarray(seg.mask).astype(bool)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        return m
    out = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = seg.box
    xa, ya = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    xb, yb = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if xb > xa and yb > ya:
        out[ya:yb, xa:xb] = True
    return out


def body_mask(parts: Sequence[Segment], shape: tuple[int, int]) -> np.ndarray:
    """Union of painted-panel footprints minus the footprints of glass / wheels / lights / plate / grille.

    A part is excluded when its class contains any of ``PANEL_EXCLUDE_KEYWORDS`` (case-insensitive). Excluded
    parts are also subtracted, so an overlapping door/window mask pair leaves the glass untouched.
    """
    h, w = int(shape[0]), int(shape[1])
    body = np.zeros((h, w), dtype=bool)
    excluded = np.zeros((h, w), dtype=bool)
    for seg in parts:
        if _is_excluded(seg.cls):
            excluded |= _footprint(seg, (h, w))
        else:
            body |= _footprint(seg, (h, w))
    return body & ~excluded


# ---------------------------------------------------------------------------------------------------- recolour

def _lab_to_linear_rgb(lab: np.ndarray) -> np.ndarray:
    """(N, 3) Lab -> (N, 3) linear RGB, unclipped (cv2 clips its float output, so it cannot test gamut)."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    d = 6.0 / 29.0

    def finv(t: np.ndarray) -> np.ndarray:
        return np.where(t > d, t ** 3, 3.0 * d * d * (t - 4.0 / 29.0))

    xyz = np.stack([_XN * finv(fx), finv(fy), _ZN * finv(fz)], axis=1)
    return xyz @ _XYZ2RGB.T


def _chroma_limit_lut(a_t: float, b_t: float, steps: int = 14) -> np.ndarray:
    """For each L on ``_L_GRID``: largest s in [0, 1] with Lab(L, s*a_t, s*b_t) inside sRGB (binary search).

    Scaling chroma down instead of clipping RGB keeps both the pixel's L and the target hue.
    """
    lo = np.zeros_like(_L_GRID)
    hi = np.ones_like(_L_GRID)
    eps = 1e-6

    def inside(s: np.ndarray) -> np.ndarray:
        rgb = _lab_to_linear_rgb(np.stack([_L_GRID, s * a_t, s * b_t], axis=1))
        return np.all((rgb >= -eps) & (rgb <= 1.0 + eps), axis=1)

    ok_full = inside(hi)
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        ok = inside(mid)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return np.where(ok_full, 1.0, lo * 0.995)


@dataclass(frozen=True)
class _Prepared:
    """Per-image work shared by every target colour (crop, LAB of masked pixels, feather alpha)."""

    y0: int
    x0: int
    sel: np.ndarray        # bool crop mask
    lab: np.ndarray        # float32 (N, 3) LAB of the masked pixels
    alpha: np.ndarray      # float32 (N, 1) blend weight, 1 in the interior, ramps down to the mask edge
    median_l: float


def _check_inputs(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"img must be uint8 (H, W, 3) BGR, got {img.dtype} {img.shape}")
    m = np.asarray(mask)
    if m.shape != img.shape[:2]:
        raise ValueError(f"mask shape {m.shape} != image shape {img.shape[:2]}")
    return m.astype(bool, copy=False)


def _prepare(img: np.ndarray, mask: np.ndarray, feather_px: int) -> _Prepared | None:
    m = _check_inputs(img, mask)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return None
    h, w = m.shape
    # one pixel of margin so mask edges on the bbox border see a non-mask neighbour; the image border does not
    y0, y1 = max(0, int(ys.min()) - 1), min(h, int(ys.max()) + 2)
    x0, x1 = max(0, int(xs.min()) - 1), min(w, int(xs.max()) + 2)
    sel = m[y0:y1, x0:x1]
    px = img[y0:y1, x0:x1][sel].reshape(-1, 1, 3).astype(np.float32) / 255.0
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    if feather_px > 0:
        dist = cv2.distanceTransform(sel.astype(np.uint8), cv2.DIST_L2, 3)[sel]
        alpha = np.clip(dist / float(feather_px + 1), 0.0, 1.0).astype(np.float32)
    else:
        alpha = np.ones(lab.shape[0], dtype=np.float32)
    return _Prepared(y0=y0, x0=x0, sel=sel, lab=lab, alpha=alpha.reshape(-1, 1),
                     median_l=float(np.median(lab[:, 0])))


def _apply(img: np.ndarray, prep: _Prepared | None, target_hex: str, chroma_gain: float,
           luminance_mix: float) -> np.ndarray:
    out = img.copy()
    if prep is None:
        hex_to_bgr(target_hex)  # still validate the colour
        return out
    if not 0.0 <= luminance_mix <= 1.0:
        raise ValueError(f"luminance_mix must be in [0, 1], got {luminance_mix}")
    if chroma_gain < 0.0:
        raise ValueError(f"chroma_gain must be >= 0, got {chroma_gain}")
    t_l, t_a, t_b = hex_to_lab(target_hex)
    a_t, b_t = t_a * chroma_gain, t_b * chroma_gain

    L = prep.lab[:, 0].astype(np.float64)
    if luminance_mix > 0.0:
        shaded = np.clip(L * t_l / max(prep.median_l, 1.0), 0.0, 100.0)
        L = L * (1.0 - luminance_mix) + shaded * luminance_mix
    s = np.interp(L, _L_GRID, _chroma_limit_lut(a_t, b_t))
    new = np.stack([L, s * a_t, s * b_t], axis=1).astype(np.float32)
    lab = prep.lab * (1.0 - prep.alpha) + new * prep.alpha

    bgr = cv2.cvtColor(lab.reshape(-1, 1, 3), cv2.COLOR_Lab2BGR).reshape(-1, 3)
    vals = np.clip(np.rint(bgr * 255.0), 0, 255).astype(np.uint8)
    sh = prep.sel.shape
    region = out[prep.y0:prep.y0 + sh[0], prep.x0:prep.x0 + sh[1]]
    region[prep.sel] = vals
    return out


def recolor(img: np.ndarray, mask: np.ndarray, target_hex: str, *, chroma_gain: float = 1.0,
            luminance_mix: float = 0.0, feather_px: int = 3) -> np.ndarray:
    """Recolour the masked pixels of a uint8 BGR image toward ``target_hex``; returns a new image.

    Method: convert to CIE LAB (OpenCV float path), set each masked pixel's a/b to the target's a/b times
    ``chroma_gain`` and keep its L, so shading, reflections and damage stay visible. Where that a/b would fall
    outside sRGB at the pixel's L, chroma is scaled down (hue and L kept) instead of clipping RGB. The edge is
    feathered inwards over ``feather_px`` pixels (blend in LAB), so pixels outside the mask are bit-identical.

    CLAUDE.md asks to keep the L channel, so with the default ``luminance_mix=0`` a white car stays light and
    comes out as a pastel tint of a dark target (Midnight Blue -> pale blue); a black car stays near-black.
    ``luminance_mix`` in (0, 1] is EXPERIMENTAL: it blends L toward ``L * target_L / median_L_in_mask``
    (clipped to 0..100), which keeps relative shading but moves overall lightness to the target's - useful for
    dark targets on light cars. Highlights may wash out; show it labelled as experimental.
    """
    prep = _prepare(img, mask, int(feather_px))
    return _apply(img, prep, target_hex, chroma_gain, luminance_mix)


def recolor_all(img: np.ndarray, mask: np.ndarray, palette: Sequence[dict[str, Any]],
                **kw: Any) -> list[dict[str, Any]]:
    """Recolour for every palette entry: [{"name", "hex", "image", "model", "source_url"}].

    Keyword args are those of ``recolor``. The LAB conversion and feather mask are computed once and reused.
    """
    feather_px = int(kw.pop("feather_px", 3))
    chroma_gain = float(kw.pop("chroma_gain", 1.0))
    luminance_mix = float(kw.pop("luminance_mix", 0.0))
    if kw:
        raise TypeError(f"unexpected keyword arguments: {sorted(kw)}")
    prep = _prepare(img, mask, feather_px)
    results: list[dict[str, Any]] = []
    for c in palette:
        results.append({
            "name": c["name"],
            "hex": c["hex"],
            "image": _apply(img, prep, c["hex"], chroma_gain, luminance_mix),
            "model": c.get("model"),
            "source_url": c.get("source_url"),
        })
    return results
