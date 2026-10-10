"""Privacy: detect and pixelate licence plates and faces. FAIL CLOSED.

Detectors (files from models/manifest.json, see scripts/download_models.py):
- plates: morsetechlab YOLO11n (models/plate/license-plate-finetune-v1n.pt on CUDA, .onnx on CPU)
  pass A imgsz 640 conf 0.25, pass B imgsz 1536 conf 0.10 (run on the FULL-RES image); the
  two-pass settings follow a reported test on Uzbek plates (docs/research/phase0_findings_models.md,
  section 3) and are uncalibrated on our data
- faces: YuNet via cv2.FaceDetectorYN, file chosen by OpenCV major version
  (>=5 -> face_detection_yunet_2026may.onnx, 4.x -> face_detection_yunet_2023mar.onnx),
  multi-scale (working image, long side ~400, and 1280-px tiles for images > 2500 px), score >= 0.5
- plates on photos with a long side > 2500 px: also 1280-px tiles at 640 (pass "plate@tiles")
- secondary-pass gate: a "plate@1536" or "plate@tiles" box that is low-confidence, not overlapped by a
  primary plate box (640 pass, parts model) and large or taller than wide is dropped (PLATE_* constants;
  counted in passes as "plate:secondary_dropped")
- body-panel check: with the parts-model detector, any plate box (the parts model's own included) that
  lies mostly on doors/fenders/windows/roof/hood/mirrors/wheels is dropped - plates are not mounted there
- person heads: make_person_head_detector(coco_model) adds the head band of every COCO person box as a
  face region (backstop for faces YuNet misses)
- extra_detectors: optional callables (e.g. the parts model's 'license-plate' class, see
  make_parts_plate_detector) -> list[Box], called with the full image
Overlapping boxes of the same kind (across passes) are merged into their union for counting only;
the detected boxes themselves are padded by `pad` (fraction of box size) and pixelated: mosaic cells of `block` px, but never more
than MOSAIC_MAX_CELLS cells along the long side of a box, so large plates in full-resolution photos
are not left legible.
FAIL CLOSED: Anonymizer() raises PrivacyUnavailable if a detector cannot be loaded; anonymize()
returns (None, report with ok=False) if any detector raises. Callers MUST drop such images:
never display, save or pass them on.
Uses ultralytics; YOLO_AUTOINSTALL must be "false" in the environment (set as a default here before
ultralytics is first imported). All thresholds and scales below are uncalibrated defaults.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import cv2
import numpy as np

from ..io import check_bgr8
from ..paths import MODELS

log = logging.getLogger("avtopassport.privacy")


class PrivacyUnavailable(RuntimeError):
    """A required privacy detector could not be loaded."""


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    kind: Literal["plate", "face"]
    source: str                       # e.g. "plate@640", "plate@1536", "face@full", "parts:license-plate"


@dataclass
class PrivacyReport:
    ok: bool
    n_plates: int = 0
    n_faces: int = 0
    boxes: list[Box] = field(default_factory=list)          # merged boxes that were pixelated
    passes: dict[str, int] = field(default_factory=dict)   # detections per pass
    errors: list[str] = field(default_factory=list)


Detector = Callable[[np.ndarray], list[Box]]
_Pass = Callable[[np.ndarray], tuple[list[Box], dict[str, int]]]
_KINDS = ("plate", "face")

DEFAULT_PLATE_PT = MODELS / "plate" / "license-plate-finetune-v1n.pt"
DEFAULT_PLATE_ONNX = MODELS / "plate" / "license-plate-finetune-v1n.onnx"
YUNET_V5 = MODELS / "face" / "face_detection_yunet_2026may.onnx"
YUNET_V4 = MODELS / "face" / "face_detection_yunet_2023mar.onnx"

# Face multi-scale schedule (uncalibrated defaults)
FACE_FULL_MAX_SIDE = 2000      # run on the image itself up to this long side ...
FACE_WORKING_SIDE = 1600       # ... otherwise on a copy with this long side
FACE_LARGE_SIDE = 400          # extra copy for large, close-up faces
FACE_TILE = 1280               # full-resolution tiles for small faces ...
FACE_TILE_OVERLAP = 0.15
FACE_TILE_MIN_SIDE = 2500      # ... on images whose long side exceeds this
FACE_NMS_IOU = 0.3
FACE_TOP_K = 5000
# Mosaic: at most this many cells along the long side of a padded box (uncalibrated default).
MOSAIC_MAX_CELLS = 10
# Same-object test used to merge boxes across passes (uncalibrated defaults).
MERGE_IOU = 0.3
MERGE_CONTAIN = 0.6
# Plate tiles for large photos: small background plates that the 640 and 1536 passes miss
# (uncalibrated defaults; tiles are run at the 640 pass's size and confidence).
PLATE_TILE = 1280
PLATE_TILE_OVERLAP = 0.15
PLATE_TILE_MIN_SIDE = 2500
# Secondary plate passes (high-resolution 1536 pass and 1280-px tiles) exist for SMALL, distant plates;
# the primary passes (plate model at 640 on the full image, parts-model plate class) find near plates.
# Seen in the smoke test: secondary passes fire on plain door/fender areas (scores 0.12-0.84, tile boxes
# as wide as the tile), and those boxes blurred whole side panels. Gate (uncalibrated defaults): a
# secondary box that no PRIMARY box overlaps is dropped if it is large (long side > PLATE_HI_MAX_FRAC of
# the image's long side; tiles: > PLATE_TILE_MAX_FRAC of the tile), whatever its score, or if it is
# weak (< PLATE_HI_KEEP_SCORE) and taller than wide (a plate seen that edge-on is not legible anyway).
PLATE_HI_KEEP_SCORE = 0.5
PLATE_HI_MAX_FRAC = 0.15
PLATE_TILE_MAX_FRAC = 0.25
PLATE_MAX_TALL = 1.3
# Person-head backstop from a COCO detector: the top of each person box is treated as a face region,
# so heads YuNet misses (small, turned away, partly hidden) are still pixelated (uncalibrated defaults).
HEAD_FRAC_FULL_BODY = 0.3      # person box taller than 2x its width: top 30 % is head and neck
HEAD_FRAC_PARTIAL = 0.5        # otherwise (upper body only): top half
HEAD_WIDEN = 0.1               # widen the head band by 10 % of the box width on each side


# --------------------------------------------------------------------------- pixelation


def _check_pad(pad: float) -> None:
    """A negative pad would shrink every box and leave its edges (or all of it) unpixelated."""
    if not (math.isfinite(pad) and pad >= 0):
        raise ValueError(f"pad must be a finite number >= 0, got {pad}")


def _padded_rect(b: Box, pad: float, w: int, h: int) -> tuple[int, int, int, int]:
    """Integer (x1, y1, x2, y2) of the box grown by pad * max(box w, box h) per side, clipped to the image."""
    (x1, x2), (y1, y2) = sorted((b.x1, b.x2)), sorted((b.y1, b.y2))
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2, pad)):
        raise ValueError(f"non-finite box or pad: {b}, pad={pad}")
    p = pad * max(x2 - x1, y2 - y1)
    return (max(0, math.floor(x1 - p)), max(0, math.floor(y1 - p)),
            min(w, math.ceil(x2 + p)), min(h, math.ceil(y2 + p)))


def pixelate(
    img: np.ndarray, boxes: Sequence[Box], *, pad: float = 0.18, block: int = 12,
    max_cells: int | None = MOSAIC_MAX_CELLS,
) -> np.ndarray:
    """Return a copy with each padded box pixelated (mosaic of `block`-px cells, minimum 3x3 cells).

    Each box is padded by pad * max(box w, box h) on every side and clipped to the image. Cells grow
    beyond `block` px when the padded box is longer than max_cells * block. Reason (arithmetic, not a
    measurement): a 700-px plate in a full-resolution photo pads to ~950 px, i.e. ~79 cells of 12 px,
    ~9 cells per character, which can stay legible. ``max_cells=None`` gives plain `block`-px cells.
    """
    check_bgr8(img)
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    _check_pad(pad)
    out = img.copy()
    h_img, w_img = img.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = _padded_rect(b, pad, w_img, h_img)
        if x2 <= x1 or y2 <= y1:
            continue  # box entirely outside the image
        roi = out[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        cell = max(float(block), max(w, h) / max_cells) if max_cells else float(block)
        small = cv2.resize(roi, (max(3, int(w // cell)), max(3, int(h // cell))), interpolation=cv2.INTER_LINEAR)
        out[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


# Smooth blur used by Anonymizer (uncalibrated defaults). Sigma scales with the box's short side, so a
# near plate and a distant one are blurred equally hard relative to their character height; the box is
# blurred at a reduced size (sigma ~ BLUR_WORK_SIGMA px there) and scaled back, which is fast and smooth.
BLUR_SIGMA_FRAC = 0.35
BLUR_WORK_SIGMA = 4.0
DEFAULT_PAD = 0.08


def blur_boxes(img: np.ndarray, boxes: Sequence[Box], *, pad: float = DEFAULT_PAD,
               sigma_frac: float = BLUR_SIGMA_FRAC) -> np.ndarray:
    """Return a copy with each padded box replaced by a strong Gaussian blur of itself.

    Sigma = sigma_frac * the padded box's short side (>= 2 px): ~0.45x the character height of a plate
    and ~a third of a face width, which leaves neither legible nor recognisable. Pixels outside the
    padded boxes are unchanged."""
    check_bgr8(img)
    _check_pad(pad)
    if not (math.isfinite(sigma_frac) and sigma_frac > 0):
        raise ValueError(f"sigma_frac must be > 0, got {sigma_frac}")
    out = img.copy()
    h_img, w_img = img.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = _padded_rect(b, pad, w_img, h_img)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = out[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        sigma = max(2.0, sigma_frac * min(h, w))
        f = max(1.0, sigma / BLUR_WORK_SIGMA)
        small = cv2.resize(roi, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma / f, sigmaY=sigma / f, borderType=cv2.BORDER_REPLICATE)
        out[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


# --------------------------------------------------------------------------- box utilities


def _overlap(a: Box, b: Box) -> tuple[float, float]:
    """(IoU, intersection / area of the smaller box)."""
    iw = min(a.x2, b.x2) - max(a.x1, b.x1)
    ih = min(a.y2, b.y2) - max(a.y1, b.y1)
    if iw <= 0 or ih <= 0:
        return 0.0, 0.0
    inter = iw * ih
    area_a = max(a.x2 - a.x1, 0.0) * max(a.y2 - a.y1, 0.0)
    area_b = max(b.x2 - b.x1, 0.0) * max(b.y2 - b.y1, 0.0)
    union = area_a + area_b - inter
    return (inter / union if union > 0 else 0.0), (inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0)


def _union(a: Box, b: Box) -> Box:
    sources = sorted(set(a.source.split("+")) | set(b.source.split("+")))
    return Box(min(a.x1, b.x1), min(a.y1, b.y1), max(a.x2, b.x2), max(a.y2, b.y2),
               max(a.score, b.score), a.kind, "+".join(sources))


def _merge_boxes(boxes: Sequence[Box], iou: float = MERGE_IOU, contain: float = MERGE_CONTAIN) -> list[Box]:
    """Replace same-kind boxes that overlap (IoU >= iou, or intersection >= contain of the smaller one)
    by their union, until no pair overlaps. The union covers every input box, so nothing is lost.
    Each sweep folds every box into the first kept box it overlaps; sweeps repeat while anything merged
    (grown boxes can overlap new ones), so a sweep is O(n * kept) instead of restarting after each merge."""
    out = list(boxes)
    merged = True
    while merged:
        merged = False
        kept: list[Box] = []
        for b in out:
            for k, a in enumerate(kept):
                if a.kind != b.kind:
                    continue
                o_iou, o_contain = _overlap(a, b)
                if o_iou >= iou or o_contain >= contain:
                    kept[k] = _union(a, b)
                    merged = True
                    break
            else:
                kept.append(b)
        out = kept
    return out


def _np(x: Any) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _yolo_rows(
    model: Any, img: np.ndarray, *, imgsz: int, conf: float, device: str, classes: Sequence[int] | None = None
) -> list[tuple[float, float, float, float, float, int]]:
    """(x1, y1, x2, y2, score, class) in input-image pixels from an ultralytics model.predict().
    Raises if the model returns no result or a result without boxes (a classify or OBB model): that is a
    wrong model, and reading it as "no detections" would let every image through."""
    kw: dict[str, Any] = {"classes": list(classes)} if classes is not None else {}
    rows: list[tuple[float, float, float, float, float, int]] = []
    results = list(model.predict(img, imgsz=imgsz, conf=conf, device=device, verbose=False, **kw))
    if not results:
        raise RuntimeError("the model returned no result for the image")
    for r in results:
        b = getattr(r, "boxes", None)
        if b is None:
            raise RuntimeError(f"the model returned no boxes (task {getattr(model, 'task', '?')!r}); "
                               "a detect or segment model is required")
        if len(b) == 0:
            continue
        for xyxy, s, c in zip(_np(b.xyxy), _np(b.conf), _np(b.cls)):
            rows.append((float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(s), int(c)))
    return rows


def _tiled_rows(
    run: Callable[[np.ndarray], list[tuple[float, float, float, float, float, int]]],
    img: np.ndarray,
    *,
    tile: int = PLATE_TILE,
    overlap: float = PLATE_TILE_OVERLAP,
) -> list[tuple[float, float, float, float, float, int]]:
    """Run a row detector on full-resolution tiles and return rows in full-image pixels."""
    h, w = img.shape[:2]
    rows: list[tuple[float, float, float, float, float, int]] = []
    for y0 in _tile_starts(h, tile, overlap):
        for x0 in _tile_starts(w, tile, overlap):
            view = np.ascontiguousarray(img[y0:y0 + tile, x0:x0 + tile])
            rows += [(x1 + x0, y1 + y0, x2 + x0, y2 + y0, s, c) for x1, y1, x2, y2, s, c in run(view)]
    return rows


def _is_primary_plate(b: Box, primary_sources: Sequence[str]) -> bool:
    return b.kind == "plate" and (b.source in primary_sources or b.source.startswith("parts:"))


def _gate_secondary_plate_boxes(
    boxes: Sequence[Box],
    image_long_side: int,
    *,
    primary_sources: Sequence[str],
    hi_source: str,
    tile_source: str = "plate@tiles",
    tile: int = PLATE_TILE,
    keep_score: float = PLATE_HI_KEEP_SCORE,
    max_frac: float = PLATE_HI_MAX_FRAC,
    tile_max_frac: float = PLATE_TILE_MAX_FRAC,
    max_tall: float = PLATE_MAX_TALL,
) -> tuple[list[Box], int]:
    """Drop secondary-pass plate boxes (hi_source, tile_source) that no PRIMARY plate box overlaps and that
    are either large (any score: near plates are the primary passes' job) or low-confidence and taller
    than wide. Small secondary boxes (distant plates), primary boxes and faces are always kept.
    Returns (kept boxes, number dropped)."""
    primary = [b for b in boxes if _is_primary_plate(b, primary_sources)]
    kept: list[Box] = []
    dropped = 0
    for b in boxes:
        if b.kind == "plate" and b.source in (hi_source, tile_source):
            w, h = b.x2 - b.x1, b.y2 - b.y1
            limit = tile_max_frac * tile if b.source == tile_source else max_frac * image_long_side
            corroborated = any(iou >= 0.1 or contain >= 0.5 for iou, contain in (_overlap(b, o) for o in primary))
            large = max(w, h) > limit                       # not a distant plate: the primary passes own these
            tall_and_weak = b.score < keep_score and h > max_tall * w
            if not corroborated and (large or tall_and_weak):
                dropped += 1
                continue
        kept.append(b)
    return kept, dropped


@dataclass
class _PersonHeadDetector:
    model: Any
    device: str
    imgsz: int
    conf: float
    person_ids: tuple[int, ...]

    @property
    def name(self) -> str:
        return "coco:person-head"

    def __call__(self, img: np.ndarray) -> list[Box]:
        rows = _yolo_rows(self.model, img, imgsz=self.imgsz, conf=self.conf, device=self.device,
                          classes=list(self.person_ids))
        h_img, w_img = img.shape[:2]
        out: list[Box] = []
        for x1, y1, x2, y2, s, c in rows:
            if c not in self.person_ids:
                continue
            w, h = x2 - x1, y2 - y1
            band = h * (HEAD_FRAC_FULL_BODY if h >= 2 * w else HEAD_FRAC_PARTIAL)
            out.append(Box(max(0.0, x1 - HEAD_WIDEN * w), max(0.0, y1), min(float(w_img), x2 + HEAD_WIDEN * w),
                           min(float(h_img), y1 + band), s, "face", "coco:person-head"))
        return out


def make_person_head_detector(coco_model: Any, *, device: str, imgsz: int = 640, conf: float = 0.25) -> Detector:
    """Backstop for faces: the head band of every COCO 'person' box, as Box(kind="face",
    source="coco:person-head"). Raises PrivacyUnavailable if the model has no 'person' class."""
    names = getattr(coco_model, "names", None)
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise PrivacyUnavailable("COCO model has no class names")
    ids = tuple(int(k) for k, v in names.items() if str(v).lower() == "person")
    if not ids:
        raise PrivacyUnavailable("no 'person' class in the COCO model")
    return _PersonHeadDetector(coco_model, device, imgsz, conf, ids)


# --------------------------------------------------------------------------- faces


def _default_face_model() -> Path:
    return YUNET_V5 if int(cv2.__version__.split(".")[0]) >= 5 else YUNET_V4


def _tile_starts(n: int, tile: int = FACE_TILE, overlap: float = FACE_TILE_OVERLAP) -> list[int]:
    if n <= tile:
        return [0]
    return [*range(0, n - tile, max(1, int(tile * (1 - overlap)))), n - tile]


def _face_views(h: int, w: int) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """(pass name, scale, (x0, y0, x1, y1) crop in full-res pixels) for each face-detector run."""
    long_side = max(h, w)
    full = (0, 0, w, h)
    if long_side <= FACE_FULL_MAX_SIDE:
        views = [("face@full", 1.0, full)]
    else:
        views = [(f"face@{FACE_WORKING_SIDE}", FACE_WORKING_SIDE / long_side, full)]
    if long_side > FACE_LARGE_SIDE:
        views.append((f"face@{FACE_LARGE_SIDE}", FACE_LARGE_SIDE / long_side, full))
    if long_side > FACE_TILE_MIN_SIDE:
        views += [("face@tiles", 1.0, (x0, y0, min(x0 + FACE_TILE, w), min(y0 + FACE_TILE, h)))
                  for y0 in _tile_starts(h) for x0 in _tile_starts(w)]
    return views


def _detect_faces_multiscale(
    img: np.ndarray, run: Callable[[np.ndarray], np.ndarray | None]
) -> tuple[list[Box], dict[str, int]]:
    """Run a YuNet-style detector (rows of x, y, w, h, ..., score) on every view; boxes in full-res pixels."""
    boxes: list[Box] = []
    passes: dict[str, int] = {}
    for name, scale, (x0, y0, x1, y1) in _face_views(*img.shape[:2]):
        view = img[y0:y1, x0:x1]
        if scale != 1.0:
            size = (max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale)))
            view = cv2.resize(view, size, interpolation=cv2.INTER_AREA)
        sx, sy = (x1 - x0) / view.shape[1], (y1 - y0) / view.shape[0]
        rows = run(np.ascontiguousarray(view))
        found = [
            Box(x0 + float(r[0]) * sx, y0 + float(r[1]) * sy,
                x0 + (float(r[0]) + float(r[2])) * sx, y0 + (float(r[1]) + float(r[3])) * sy,
                float(r[-1]), "face", name)
            for r in (() if rows is None else rows)
        ]
        passes[name] = passes.get(name, 0) + len(found)
        boxes += found
    return boxes, passes


# --------------------------------------------------------------------------- detectors


def _extra_pass(det: Detector, i: int) -> _Pass:
    name = str(getattr(det, "name", None) or getattr(det, "__name__", None) or f"extra{i}")

    def run(img: np.ndarray) -> tuple[list[Box], dict[str, int]]:
        found = list(det(img))
        for b in found:
            if not isinstance(b, Box):
                raise TypeError(f"detector {name} returned {type(b).__name__}, expected Box")
            if b.kind not in _KINDS:  # would be pixelated but counted as neither plate nor face
                raise ValueError(f"detector {name} returned kind {b.kind!r}, expected one of {_KINDS}")
        return found, {name: len(found)}

    return run


# Body panels on which a licence plate cannot be mounted (substring match on the parts-model class
# names). A plate box from the plate model that lies mostly on these panels is a false positive.
PANEL_KEYWORDS = ("door", "fender", "quarter-panel", "rocker-panel", "window", "windshield", "roof", "hood",
                  "mirror", "wheel")
PLATE_ON_PANEL_MAX = 0.5   # share of a plate box covered by body panels above which it is dropped (uncalibrated)
PANEL_MASK_SCALE = 0.25    # body-panel mask resolution relative to the image


@dataclass
class _PartsPlateDetector:
    model: Any
    device: str
    imgsz: int
    conf: float
    class_ids: dict[int, str]
    panel_ids: tuple[int, ...] = ()
    _cache: tuple[int, tuple[int, ...], np.ndarray] | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "parts:" + "+".join(sorted(set(self.class_ids.values())))

    def __call__(self, img: np.ndarray) -> list[Box]:
        classes = sorted(set(self.class_ids) | set(self.panel_ids))
        results = list(self.model.predict(img, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                          verbose=False, classes=classes))
        if not results:
            raise RuntimeError("the parts model returned no result for the image")
        h, w = img.shape[:2]
        mask = np.zeros((max(1, round(h * PANEL_MASK_SCALE)), max(1, round(w * PANEL_MASK_SCALE))), np.uint8)
        boxes: list[Box] = []
        for r in results:
            b = getattr(r, "boxes", None)
            if b is None:
                raise RuntimeError("the parts model returned no boxes; a detect or segment model is required")
            if len(b) == 0:
                continue
            cls = [int(c) for c in _np(b.cls)]
            for xyxy, sc, c in zip(_np(b.xyxy), _np(b.conf), cls):
                if c in self.class_ids:
                    boxes.append(Box(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(sc),
                                     "plate", f"parts:{self.class_ids[c]}"))
            polys = getattr(getattr(r, "masks", None), "xy", None)
            if polys is not None:
                for poly, c in zip(polys, cls):
                    if c in self.panel_ids and len(poly) >= 3:
                        cv2.fillPoly(mask, [np.round(np.asarray(poly) * PANEL_MASK_SCALE).astype(np.int32)], 1)
        self._cache = (id(img), img.shape, mask)
        return boxes

    def panel_mask(self, img: np.ndarray) -> np.ndarray:
        """Body-panel mask (PANEL_MASK_SCALE resolution) for `img`; reuses the last prediction on it."""
        if self._cache is None or self._cache[0] != id(img) or self._cache[1] != img.shape:
            self(img)
        assert self._cache is not None
        return self._cache[2]


def _drop_plates_on_panels(boxes: Sequence[Box], mask: np.ndarray, *, scale: float = PANEL_MASK_SCALE,
                           max_cover: float = PLATE_ON_PANEL_MAX) -> tuple[list[Box], int]:
    """Drop every plate box (whatever its source, the parts model's own included) whose area is mostly
    (>= max_cover) on body panels where plates are not mounted. Seen in the smoke test: door decals with
    text ("TOYOTA CAMRY", a phone number) were called plates by both the plate model and the parts model.
    A real plate sits on a bumper or the trunk lid, which are not body panels here. Returns (kept, dropped)."""
    mh, mw = mask.shape[:2]
    kept: list[Box] = []
    dropped = 0
    for b in boxes:
        if b.kind == "plate":
            x1, y1 = max(0, int(b.x1 * scale)), max(0, int(b.y1 * scale))
            x2, y2 = min(mw, max(x1 + 1, math.ceil(b.x2 * scale))), min(mh, max(y1 + 1, math.ceil(b.y2 * scale)))
            cover = float(mask[y1:y2, x1:x2].mean()) if x2 > x1 and y2 > y1 else 0.0
            if cover >= max_cover:
                dropped += 1
                continue
        kept.append(b)
    return kept, dropped


def make_parts_plate_detector(
    parts_model: Any,
    *,
    device: str,
    imgsz: int = 768,
    conf: float = 0.25,
    class_names: Sequence[str] = ("license-plate", "license_plate", "plate"),
) -> Detector:
    """Detector that runs an already-loaded ultralytics model (e.g. mitbersh car-parts-segmentation)
    and returns its plate-class boxes as Box(kind="plate", source="parts:<class name>").
    Raises PrivacyUnavailable if none of ``class_names`` (case-insensitive) is in ``model.names``,
    so a wrong class map fails closed instead of silently finding no plates."""
    names = getattr(parts_model, "names", None)
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise PrivacyUnavailable("parts model has no class names")
    wanted = {n.lower() for n in class_names}
    ids = {int(k): str(v) for k, v in names.items() if str(v).lower() in wanted}
    if not ids:
        known = sorted(map(str, names.values()))
        raise PrivacyUnavailable(f"none of {list(class_names)} in the parts model classes {known}")
    panels = tuple(int(k) for k, v in names.items()
                   if int(k) not in ids and any(kw in str(v).lower() for kw in PANEL_KEYWORDS))
    return _PartsPlateDetector(parts_model, device, imgsz, conf, ids, panels)


class Anonymizer:
    """Plate + face detection and pixelation; fail closed (see the module docstring)."""

    def __init__(
        self,
        *,
        device: str = "cpu",                  # "cpu" | "cuda" | "cuda:0"
        plate_model: Path | None = None,       # default: .pt on cuda, .onnx on cpu
        face_model: Path | None = None,        # default: chosen by cv2 major version
        plate_conf: float = 0.25,
        plate_conf_hi: float = 0.10,
        plate_imgsz: int = 640,
        plate_imgsz_hi: int = 1536,
        face_score: float = 0.5,
        pad: float = DEFAULT_PAD,
        extra_detectors: Sequence[Detector] = (),
    ) -> None:
        """Load both detectors and run each once on a blank image; raise PrivacyUnavailable on any failure.
        Raises ValueError for settings that would silently disable a detector (a threshold >= 1, pad < 0)."""
        dev = str(device).lower()
        if dev != "cpu" and not dev.startswith("cuda"):
            raise ValueError(f"device must be 'cpu', 'cuda' or 'cuda:N', got {device!r}")
        for what, v in (("plate_conf", plate_conf), ("plate_conf_hi", plate_conf_hi), ("face_score", face_score)):
            if not 0 <= v < 1:
                raise ValueError(f"{what} must be in [0, 1), got {v}")
        if min(plate_imgsz, plate_imgsz_hi) < 32:
            raise ValueError(f"plate image sizes must be >= 32, got {plate_imgsz}, {plate_imgsz_hi}")
        _check_pad(pad)
        self.device = dev
        self.plate_conf, self.plate_conf_hi = plate_conf, plate_conf_hi
        self.plate_imgsz, self.plate_imgsz_hi = plate_imgsz, plate_imgsz_hi
        self.face_score = face_score
        self.pad = pad
        self.extra_detectors = tuple(extra_detectors)
        cuda = dev.startswith("cuda")
        self.plate_model_path = Path(plate_model) if plate_model else (DEFAULT_PLATE_PT if cuda else DEFAULT_PLATE_ONNX)
        self.face_model_path = Path(face_model) if face_model else _default_face_model()
        for what, path in (("plate", self.plate_model_path), ("face", self.face_model_path)):
            if not path.is_file():
                raise PrivacyUnavailable(f"{what} detector model missing: {path} (run scripts/download_models.py)")
        self._face = self._load_face()
        self._plate = self._load_plate()
        self._passes: list[_Pass] = [self._detect_plates, self._detect_faces]
        self._passes += [_extra_pass(d, i) for i, d in enumerate(self.extra_detectors)]

    @classmethod
    def from_detectors(cls, detectors: Sequence[Detector], *, pad: float = DEFAULT_PAD) -> Anonymizer:
        """An Anonymizer that runs ONLY the given detectors, with the same fail-closed anonymize().
        For tests and custom detector sets; production code uses Anonymizer(...), which loads the
        plate and face models."""
        if not detectors:
            raise ValueError("at least one detector is required")
        _check_pad(pad)
        self = cls.__new__(cls)
        self.device, self.pad, self.extra_detectors = "cpu", pad, tuple(detectors)
        self.plate_conf = self.plate_conf_hi = self.plate_imgsz = self.plate_imgsz_hi = self.face_score = None
        self.plate_model_path = self.face_model_path = None
        self._face = self._plate = None
        self._passes = [_extra_pass(d, i) for i, d in enumerate(self.extra_detectors)]
        return self

    def _load_face(self) -> Any:
        path = self.face_model_path
        try:
            det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), float(self.face_score), FACE_NMS_IOU, FACE_TOP_K)
            det.setInputSize((64, 64))
            det.detect(np.zeros((64, 64, 3), np.uint8))  # fails here if the network cannot run
        except Exception as exc:
            raise PrivacyUnavailable(f"face detector could not be loaded from {path}: {exc}") from exc
        return det

    def _load_plate(self) -> Any:
        path = self.plate_model_path
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        threads = cv2.getNumThreads()
        try:
            from ultralytics import YOLO

            # importing ultralytics calls cv2.setNumThreads(0) (a training/DataLoader setting), which
            # makes the YuNet tiles ~2x slower; keep the caller's OpenCV thread count instead
            cv2.setNumThreads(threads)
            model = YOLO(str(path), task="detect")
            # ONNX sessions / CUDA load lazily: one blank prediction makes a broken backend fail here
            blank = np.zeros((64, 64, 3), np.uint8)
            _yolo_rows(model, blank, imgsz=self.plate_imgsz, conf=self.plate_conf, device=self.device)
        except Exception as exc:
            raise PrivacyUnavailable(f"plate detector could not be loaded from {path} on {self.device}: "
                                     f"{type(exc).__name__}: {exc}") from exc
        return model

    def _detect_plates(self, img: np.ndarray) -> tuple[list[Box], dict[str, int]]:
        boxes: list[Box] = []
        passes: dict[str, int] = {}
        for imgsz, conf in ((self.plate_imgsz, self.plate_conf), (self.plate_imgsz_hi, self.plate_conf_hi)):
            name = f"plate@{imgsz}"
            rows = _yolo_rows(self._plate, img, imgsz=imgsz, conf=conf, device=self.device)
            boxes += [Box(x1, y1, x2, y2, s, "plate", name) for x1, y1, x2, y2, s, _ in rows]
            passes[name] = passes.get(name, 0) + len(rows)
        if max(img.shape[:2]) > PLATE_TILE_MIN_SIDE:
            def run(view: np.ndarray) -> list[tuple[float, float, float, float, float, int]]:
                return _yolo_rows(self._plate, view, imgsz=self.plate_imgsz, conf=self.plate_conf, device=self.device)
            rows = _tiled_rows(run, img)
            boxes += [Box(x1, y1, x2, y2, s, "plate", "plate@tiles") for x1, y1, x2, y2, s, _ in rows]
            passes["plate@tiles"] = len(rows)
        return boxes, passes

    def _run_yunet(self, view: np.ndarray) -> np.ndarray | None:
        self._face.setInputSize((view.shape[1], view.shape[0]))
        return self._face.detect(view)[1]

    def _detect_faces(self, img: np.ndarray) -> tuple[list[Box], dict[str, int]]:
        return _detect_faces_multiscale(img, self._run_yunet)

    def detect(self, img: np.ndarray) -> tuple[list[Box], dict[str, int]]:
        """All boxes from all passes in full-image coordinates, plus per-pass counts. May raise."""
        check_bgr8(img)
        boxes: list[Box] = []
        passes: dict[str, int] = {}
        for run in self._passes:
            found, counts = run(img)
            boxes += found
            for k, v in counts.items():
                passes[k] = passes.get(k, 0) + v
        if self.plate_imgsz_hi is not None:
            boxes, dropped = _gate_secondary_plate_boxes(
                boxes, max(img.shape[:2]), primary_sources=(f"plate@{self.plate_imgsz}",),
                hi_source=f"plate@{self.plate_imgsz_hi}")
            passes["plate:secondary_dropped"] = dropped
        for det in self.extra_detectors:
            panel_mask = getattr(det, "panel_mask", None)
            if panel_mask is not None:
                boxes, dropped = _drop_plates_on_panels(boxes, panel_mask(img))
                passes["plate:on_body_panel_dropped"] = passes.get("plate:on_body_panel_dropped", 0) + dropped
        return boxes, passes

    def anonymize(self, img: np.ndarray) -> tuple[np.ndarray | None, PrivacyReport]:
        """Pixelated copy + report; (None, report.ok=False) if any detector failed (fail closed)."""
        try:
            boxes, passes = self.detect(img)
            merged = _merge_boxes(boxes)          # one entry per object, for counting and the report
            out = blur_boxes(img, boxes, pad=self.pad)  # blur the detected boxes, not their bounding unions
        except Exception as exc:  # fail closed: the caller must drop this image
            log.warning("privacy detection failed; image must be dropped: %s: %s", type(exc).__name__, exc)
            return None, PrivacyReport(ok=False, errors=[f"{type(exc).__name__}: {exc}"])
        return out, PrivacyReport(
            ok=True,
            n_plates=sum(b.kind == "plate" for b in merged),
            n_faces=sum(b.kind == "face" for b in merged),
            boxes=merged,
            passes=passes,
        )
