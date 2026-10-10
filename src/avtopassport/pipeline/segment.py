"""Damage and car-part segmentation: ultralytics YOLO-seg wrappers that return ``schema.Segment``.

- Masks are rasterised from ``r.masks.xy`` (polygons already scaled to the input image) with
  cv2.fillPoly, so every mask is a bool (H, W) array in the INPUT image's pixel grid. Predictions use
  ``retina_masks=False`` (faster; the polygons are scaled back to the input size anyway).
- An instance whose polygon is empty (< 3 points, or rasterises to 0 px) is dropped: a zero-area mask
  cannot be mapped to a part or rated.
- ``classes`` is passed to ``predict`` AND re-checked on the output (a model that ignores it cannot
  leak other classes through).
- Every predict call passes imgsz / conf / iou / classes / retina_masks explicitly: ultralytics keeps the
  previous call's predictor arguments on a model instance, so a model shared with another caller must
  not inherit their settings.
- MODEL_SPECS mirrors the Phase 1 smoke-test notebook (scripts/build_notebook_01.py). conf / iou come
  from the model's own inference config where it ships one (mitbersh); otherwise ultralytics' defaults.
  All thresholds are uncalibrated until Phase 2.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..paths import MODELS, REPO_ROOT
from ..schema import Segment

log = logging.getLogger("avtopassport.segment")

DEFAULT_CONF, DEFAULT_IOU = 0.25, 0.7   # ultralytics' own defaults (uncalibrated)


@dataclass(frozen=True)
class ModelSpec:
    role: str            # "damage" | "parts"
    file: str            # path relative to the repo root
    imgsz: int
    config: str | None = None   # model's own inference config JSON (relative to the repo root)


MODEL_SPECS: dict[str, ModelSpec] = {
    "damage_mitbersh": ModelSpec("damage", "models/damage/mitbersh_damage_segmentation.pt", 896,
                                 "models/damage/mitbersh_damage_inference_config.json"),
    "damage_danishali": ModelSpec("damage", "models/damage/danishali_best.pt", 640),
    "damage_harpreet": ModelSpec("damage", "models/damage/harpreet_best.pt", 1024),
    "parts_mitbersh": ModelSpec("parts", "models/parts/mitbersh_parts_segmentation.pt", 768,
                                "models/parts/mitbersh_parts_inference_config.json"),
}


def _np(x: Any) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("the model has no class names")
    return {int(k): str(v) for k, v in names.items()}


def load_yolo(path: Path, *, task: str | None = None) -> Any:
    """Load an ultralytics YOLO model with auto-install disabled; keeps the caller's OpenCV thread count."""
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    threads = cv2.getNumThreads()
    from ultralytics import YOLO

    cv2.setNumThreads(threads)  # importing ultralytics sets cv2.setNumThreads(0)
    return YOLO(str(path), task=task) if task else YOLO(str(path))


def polygon_mask(poly: Any, h: int, w: int) -> np.ndarray:
    """bool (h, w) mask of one polygon given in image pixels ((N, 2) x, y); empty if < 3 points."""
    mask = np.zeros((h, w), np.uint8)
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2) if poly is not None else np.zeros((0, 2))
    if len(pts) >= 3:
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


class YoloSegmenter:
    """One loaded YOLO-seg (or detect) model + its inference settings. ``segment(img)`` -> list[Segment]."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cpu",
        imgsz: int,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        classes: list[int] | None = None,
        model: Any | None = None,      # an already-loaded ultralytics-like model (tests, sharing)
    ) -> None:
        if not 0 <= conf < 1 or not 0 < iou <= 1:
            raise ValueError(f"conf must be in [0, 1) and iou in (0, 1], got {conf}, {iou}")
        if imgsz < 32:
            raise ValueError(f"imgsz must be >= 32, got {imgsz}")
        self.model_path = Path(model_path)
        self.device, self.imgsz, self.conf, self.iou = str(device), int(imgsz), float(conf), float(iou)
        if model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"model file missing: {self.model_path} (run scripts/download_models.py)")
            model = load_yolo(self.model_path)
        self.model = model
        self.names: dict[int, str] = _names(model)
        self.classes: list[int] | None = None if classes is None else [int(c) for c in classes]

    def predict_kwargs(self) -> dict[str, Any]:
        return dict(imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device, classes=self.classes,
                    retina_masks=False, save=False, verbose=False)

    def segment(self, img: np.ndarray) -> list[Segment]:
        """Instances on ``img`` (uint8 BGR); masks are bool arrays at img's resolution (None for a detect model)."""
        h, w = img.shape[:2]
        results = list(self.model.predict(img, **self.predict_kwargs()))
        if not results:
            raise RuntimeError("the model returned no result for the image")
        allowed = None if self.classes is None else set(self.classes)
        out: list[Segment] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                raise RuntimeError("the model returned no boxes; a segment or detect model is required")
            if len(boxes) == 0:
                continue
            xyxy, scores, cls = _np(boxes.xyxy), _np(boxes.conf), _np(boxes.cls).astype(int)
            masks = getattr(r, "masks", None)
            polys = None if masks is None else masks.xy
            for j, c in enumerate(cls):
                c = int(c)
                if allowed is not None and c not in allowed:
                    continue
                x1, y1, x2, y2 = (float(v) for v in xyxy[j][:4])
                box = (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2))
                if polys is None:
                    mask = None
                    area = int(round(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])))
                else:
                    mask = polygon_mask(polys[j] if j < len(polys) else None, h, w)
                    area = int(mask.sum())
                    if area == 0:
                        log.debug("dropped %s instance with an empty mask", self.names.get(c, c))
                        continue
                out.append(Segment(cls=self.names.get(c, str(c)), score=float(scores[j]), box=box, mask=mask,
                                   area_px=area))
        return out


def inference_config(rel: str | None, root: Path = REPO_ROOT) -> dict[str, Any]:
    """The 'inference' block of a model's config JSON ({} if absent or unreadable)."""
    if not rel:
        return {}
    try:
        return json.loads((root / rel).read_text(encoding="utf-8")).get("inference") or {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("cannot read %s: %s", rel, exc)
        return {}


def resolve_spec(key: str, *, root: Path = REPO_ROOT) -> dict[str, Any]:
    """Path, imgsz, conf, iou for a MODEL_SPECS key (conf/iou from the model's config when present)."""
    if key not in MODEL_SPECS:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[key]
    cfg = inference_config(spec.config, root)
    return {"path": root / spec.file, "imgsz": spec.imgsz, "conf": float(cfg.get("conf", DEFAULT_CONF)),
            "iou": float(cfg.get("iou", DEFAULT_IOU)), "role": spec.role}


def load_segmenter(key: str, *, device: str, classes: Sequence[int] | None = None) -> YoloSegmenter:
    """YoloSegmenter for a MODEL_SPECS key, mirroring the smoke-test notebook settings."""
    s = resolve_spec(key)
    return YoloSegmenter(s["path"], device=device, imgsz=s["imgsz"], conf=s["conf"], iou=s["iou"],
                         classes=None if classes is None else list(classes))


def manifest_sha256(keys: Sequence[str], manifest: Path = MODELS / "manifest.json") -> dict[str, str | None]:
    """sha256 recorded in models/manifest.json for each key (None if absent)."""
    try:
        files = json.loads(manifest.read_text(encoding="utf-8")).get("files", {})
    except (OSError, ValueError):
        files = {}
    return {k: (files.get(k) or {}).get("sha256") for k in keys}
