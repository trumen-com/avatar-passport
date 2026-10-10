"""Frame quality filter: Laplacian-variance sharpness + mean brightness.

Thresholds are UNCALIBRATED defaults (to be calibrated on the local test set in Phase 2).
Both measurements run on a copy downscaled (never upscaled) to a long side of ``analysis_side``,
so the numbers do not depend on the source resolution of large frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..io import resize_max_side
from .frames import Frame


@dataclass(frozen=True)
class QualityThresholds:
    min_sharpness: float = 60.0      # variance of Laplacian on the analysis-size grayscale image
    min_brightness: float = 35.0     # mean V (HSV) 0..255
    max_brightness: float = 235.0
    analysis_side: int = 640         # long side used for the measurements


DEFAULT_THRESHOLDS = QualityThresholds()  # all values: uncalibrated defaults


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    sharpness: float
    brightness: float
    reason: str | None               # "blurry" | "too_dark" | "too_bright" | None ("invalid": not an image)


def _analysis_copy(img: np.ndarray, analysis_side: int) -> np.ndarray:
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim not in (2, 3) or img.size == 0:
        raise ValueError(f"expected a non-empty uint8 image, got {getattr(img, 'dtype', type(img))}")
    if img.ndim == 3 and img.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR, got shape {img.shape}")
    return resize_max_side(img, analysis_side)


def sharpness(img: np.ndarray, analysis_side: int = 640) -> float:
    """Variance of the Laplacian of the grayscale image (long side <= analysis_side). Higher = sharper."""
    small = _analysis_copy(img, analysis_side)
    gray = small if small.ndim == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(img: np.ndarray, analysis_side: int = 640) -> float:
    """Mean of the HSV V channel (= max of B, G, R) in 0..255, on the analysis-size copy."""
    small = _analysis_copy(img, analysis_side)
    if small.ndim == 2:
        return float(small.mean())
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2].mean())


def assess(img: np.ndarray, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> QualityResult:
    """Measure one image against the (uncalibrated default) thresholds.
    Exposure is checked first: a dark frame also has a low Laplacian variance, and "too_dark" is the
    more useful reason to show the user."""
    s = sharpness(img, thresholds.analysis_side)
    b = brightness(img, thresholds.analysis_side)
    if b < thresholds.min_brightness:
        reason: str | None = "too_dark"
    elif b > thresholds.max_brightness:
        reason = "too_bright"
    elif s < thresholds.min_sharpness:
        reason = "blurry"
    else:
        reason = None
    return QualityResult(ok=reason is None, sharpness=s, brightness=b, reason=reason)


def filter_frames(
    frames: list[Frame], thresholds: QualityThresholds = DEFAULT_THRESHOLDS
) -> tuple[list[Frame], list[tuple[Frame, QualityResult]], list[QualityResult]]:
    """Returns (kept, rejected_with_result, all_results_in_input_order).
    A frame whose image cannot be measured is rejected with reason "invalid" instead of raising."""
    kept: list[Frame] = []
    rejected: list[tuple[Frame, QualityResult]] = []
    results: list[QualityResult] = []
    for f in frames:
        try:
            r = assess(f.image, thresholds)
        except (ValueError, cv2.error):
            r = QualityResult(ok=False, sharpness=math.nan, brightness=math.nan, reason="invalid")
        results.append(r)
        if r.ok:
            kept.append(f)
        else:
            rejected.append((f, r))
    return kept, rejected, results
