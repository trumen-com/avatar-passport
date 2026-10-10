"""Phase 2 evaluation on our own labelled, BLURRED local test set (CLAUDE.md section 3, Phase 2).

Inputs (see docs/research/labelling_guide.md):
  data/local_testset/annotations.json    COCO 1.0 export, damage POLYGONS (no RLE, no iscrowd), each with a
                                         "part" attribute (parts-model vocabulary)
  data/local_testset/slices.csv          file, slice (day|night|wet), brand, car, no_damage, notes
  data/local_testset/images/<slice>/*.jpg the blurred photos (scripts/blur_testset.py)
  data/raw/testset/vin_ground_truth.csv  optional, NEVER committed: file, vin  (VIN photos in data/raw/testset/vin/)

Metrics, per damage model (segment.MODEL_SPECS keys) and per slice (all, day, night, wet, brand:<X>):
  - mask AP50 per class and the mean over classes with ground truth (101-point interpolated, COCO style;
    predictions matched greedily in descending score order to the unmatched GT of the same class and image
    with the highest mask IoU, a match needs IoU >= 0.5). Predictions are taken down to --min-conf.
  - part-level recall: a labelled damaged (image, part) counts as found if any kept prediction of ANY damage
    class (a) is assigned to that part by mapping.assign with the parts model, or (b) has mask IoU >= 0.1
    with a GT polygon labelled with that part.
  - false positives per car on cars tagged no_damage, and unmatched predictions per image on all images.
  - per-class confidence threshold calibration (max F1 at IoU 0.5), baseline conf 0.25 vs calibrated;
    --split kfold:K | holdout:F reports the calibrated metrics on held-out cars too (in-sample calibration
    is optimistic).
  - latency per image (median, p90) at the baseline conf on the current device.
  - VIN exact-match rate (rates only; VINs are never printed or written).
  - failure-case gallery: docs/assets/failure_cases/<model>_<n>.jpg (missed GT red, FP yellow, TP green).

Outputs: outputs/eval/<run>/metrics.json and the tables between "<!-- BEGIN:<block> -->" / "<!-- END:<block> -->"
in docs/research/evaluation.md. Without labelled data nothing is written (exit 3, the missing inputs are printed).

Usage (from the repo root):
    YOLO_AUTOINSTALL=false .venv/bin/python scripts/eval.py --check          # what is present / missing
    YOLO_AUTOINSTALL=false .venv/bin/python scripts/eval.py --split kfold:5  [--device auto|cpu|cuda]

Exit codes: 0 done; 1 finished with errors (a model failed to load, images unreadable); 3 no labelled data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
try:
    import avtopassport  # noqa: F401
except ImportError:  # not installed: use the source tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from avtopassport.paths import DATA, OUTPUTS, RAW, REPO_ROOT  # noqa: E402

LOCAL = DATA / "local_testset"
ANNOTATIONS = LOCAL / "annotations.json"
SLICES_CSV = LOCAL / "slices.csv"
IMAGES = LOCAL / "images"
VIN_DIR = RAW / "testset" / "vin"
VIN_GT = RAW / "testset" / "vin_ground_truth.csv"
DOC = REPO_ROOT / "docs" / "research" / "evaluation.md"
GALLERY = REPO_ROOT / "docs" / "assets" / "failure_cases"
EVAL_OUT = OUTPUTS / "eval"

DAMAGE_MODELS = ("damage_mitbersh", "damage_danishali", "damage_harpreet")
PARTS_MODEL = "parts_mitbersh"
IOU_MATCH = 0.5          # mask AP50 / TP definition
IOU_PART = 0.1           # part-recall rule (b): prediction overlaps the GT polygon of that part
BASELINE_CONF = 0.25     # ultralytics' default conf = what the pipeline runs today (uncalibrated)
MIN_CONF = 0.01          # predictions kept for AP / calibration (ultralytics val uses 0.001; 0.01 bounds runtime)
SLICE_NAMES = ("day", "night", "wet")
NO_DAMAGE = "no_damage"
UNKNOWN = "unknown"
BLOCKS = ("summary", "per_class", "per_slice", "part_recall", "fp_per_car", "calibration", "vin", "latency",
          "failures")
SLICES_FIELDS = ("file", "slice", "brand", "car", "no_damage", "notes")
EXIT_OK, EXIT_ERRORS, EXIT_NO_DATA = 0, 1, 3
_TRUE = {"1", "true", "yes", "y", "x"}


class DataError(Exception):
    """Inputs are missing or unusable: nothing is evaluated or written (exit 3)."""


# ------------------------------------------------------------------------------------------------ names

def norm_cls(name: str) -> str:
    """Damage class key: lower case, '_' / '-' / runs of spaces -> one space ('Glass_Shatter' -> 'glass shatter')."""
    return " ".join(re.split(r"[\s_\-]+", str(name).strip().lower())).strip()


def part_key(name: str | None) -> str | None:
    """Part key for comparisons: lower case, separators -> '-' ('left_front-door' == 'Left front door')."""
    if name is None or not str(name).strip():
        return None
    return "-".join(p for p in re.split(r"[\s_\-]+", str(name).strip().lower()) if p)


# ------------------------------------------------------------------------------------------------ masks

@dataclass(frozen=True)
class CropMask:
    """A bool mask stored as its tight bounding crop: rows y0.., cols x0.. of an (h, w) image."""

    y0: int
    x0: int
    m: np.ndarray
    area: int

    @staticmethod
    def from_full(mask: np.ndarray, box: Sequence[float] | None = None) -> CropMask | None:
        """Tight crop of a full-image mask; ``box`` (x1, y1, x2, y2) only narrows the search (2 px margin)."""
        mask = np.asarray(mask, dtype=bool)
        by, bx = 0, 0
        if box is not None:
            h, w = mask.shape
            bx, by = max(0, int(box[0]) - 2), max(0, int(box[1]) - 2)
            ex, ey = min(w, int(math.ceil(box[2])) + 2), min(h, int(math.ceil(box[3])) + 2)
            if ex <= bx or ey <= by or int(np.count_nonzero(mask[by:ey, bx:ex])) != int(np.count_nonzero(mask)):
                bx, by = 0, 0          # mask leaks outside the box: search the whole image
            else:
                mask = mask[by:ey, bx:ex]
        rows = np.flatnonzero(mask.any(axis=1))
        if rows.size == 0:
            return None
        cols = np.flatnonzero(mask.any(axis=0))
        y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        crop = mask[y0:y1, x0:x1].copy()
        return CropMask(by + y0, bx + x0, crop, int(crop.sum()))

    def full(self, h: int, w: int) -> np.ndarray:
        out = np.zeros((h, w), bool)
        hh, ww = min(self.m.shape[0], h - self.y0), min(self.m.shape[1], w - self.x0)
        if hh > 0 and ww > 0:
            out[self.y0:self.y0 + hh, self.x0:self.x0 + ww] = self.m[:hh, :ww]
        return out


def mask_iou(a: CropMask | None, b: CropMask | None) -> float:
    """Mask IoU of two crop masks of the same image (0 when either is empty)."""
    if a is None or b is None or a.area == 0 or b.area == 0:
        return 0.0
    y0, x0 = max(a.y0, b.y0), max(a.x0, b.x0)
    y1 = min(a.y0 + a.m.shape[0], b.y0 + b.m.shape[0])
    x1 = min(a.x0 + a.m.shape[1], b.x0 + b.m.shape[1])
    inter = 0
    if y1 > y0 and x1 > x0:
        sa = a.m[y0 - a.y0:y1 - a.y0, x0 - a.x0:x1 - a.x0]
        sb = b.m[y0 - b.y0:y1 - b.y0, x0 - b.x0:x1 - b.x0]
        inter = int(np.count_nonzero(sa & sb))
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def rasterize_polygon(poly: Sequence[float] | np.ndarray, h: int, w: int) -> np.ndarray:
    """bool (h, w) mask of one polygon (flat [x1, y1, x2, y2, ...] or (N, 2) pixel coordinates).

    Pure numpy, even-odd rule, a pixel is inside when its CENTRE (x + 0.5, y + 0.5) is inside the polygon
    (half-open edges, so two polygons sharing an edge do not share pixels). < 3 points -> empty mask.
    """
    out = np.zeros((h, w), bool)
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3 or h <= 0 or w <= 0 or not np.all(np.isfinite(pts)):
        return out
    xa, ya = pts[:, 0], pts[:, 1]
    xb, yb = np.roll(xa, -1), np.roll(ya, -1)
    r0 = max(0, int(math.floor(ya.min() - 0.5)))
    r1 = min(h, int(math.ceil(ya.max() + 0.5)))
    if r1 <= r0:
        return out
    yc = (np.arange(r0, r1, dtype=np.float64) + 0.5)[:, None]          # (R, 1) pixel-centre rows
    cross = ((ya <= yc) & (yb > yc)) | ((yb <= yc) & (ya > yc))         # (R, E) half-open edge rule
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = np.where(cross, xa + (yc - ya) * (xb - xa) / (yb - ya), np.inf)
    n_max = int(cross.sum(axis=1).max())
    if n_max < 2:
        return out
    xs = np.sort(xs, axis=1)[:, :n_max - n_max % 2]
    starts, ends = xs[:, 0::2], xs[:, 1::2]
    ok = np.isfinite(starts) & np.isfinite(ends)
    s = np.clip(np.ceil(starts[ok] - 0.5), 0, w).astype(np.int64)
    e = np.clip(np.ceil(ends[ok] - 0.5), 0, w).astype(np.int64)
    rows = np.broadcast_to(np.arange(r1 - r0)[:, None], starts.shape)[ok]
    diff = np.zeros((r1 - r0, w + 1), np.int32)
    np.add.at(diff, (rows, s), 1)
    np.add.at(diff, (rows, e), -1)
    out[r0:r1] = np.cumsum(diff[:, :w], axis=1) > 0
    return out


def rasterize_polygons(polys: Iterable[Sequence[float]], h: int, w: int) -> np.ndarray:
    """Union of several polygons (a COCO 'segmentation' list)."""
    out = np.zeros((h, w), bool)
    for p in polys:
        out |= rasterize_polygon(p, h, w)
    return out


# ------------------------------------------------------------------------------------------------ dataset

@dataclass
class GtInstance:
    cls: str                 # norm_cls
    part: str | None         # part_key, None when the polygon has no part attribute
    part_label: str | None   # the part as written by the labeller
    mask: CropMask


@dataclass
class GtImage:
    file: str                # path relative to data/local_testset/images (as in slices.csv)
    path: Path | None        # resolved image file (None: not found)
    width: int
    height: int
    slice: str
    brand: str
    car: str
    no_damage: bool
    instances: list[GtInstance] = field(default_factory=list)

    @property
    def labelled(self) -> bool:
        return self.no_damage or bool(self.instances)


@dataclass
class Dataset:
    images: list[GtImage]
    classes: list[str]                      # GT damage classes (norm_cls), sorted
    warnings: list[str] = field(default_factory=list)


def read_slices(path: Path) -> dict[str, dict[str, str]]:
    """slices.csv rows keyed by 'file' ({} when the file does not exist)."""
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {(r.get("file") or "").strip(): {k: (v or "").strip() for k, v in r.items() if k}
                for r in csv.DictReader(f) if (r.get("file") or "").strip()}


def _resolve_image(file_name: str, root: Path, by_name: dict[str, list[Path]]) -> Path | None:
    p = root / file_name
    if p.is_file():
        return p
    hits = by_name.get(Path(file_name).name, [])
    return hits[0] if len(hits) == 1 else None


def _image_tags(img: dict[str, Any]) -> set[str]:
    """Image-level tags a COCO exporter may carry (CVAT / Roboflow variants); normalised with norm_cls."""
    tags: set[str] = set()
    for key in ("tags", "labels"):
        v = img.get(key)
        if isinstance(v, list):
            tags |= {norm_cls(t.get("name", "") if isinstance(t, dict) else t) for t in v}
    attrs = img.get("attributes")
    if isinstance(attrs, dict):
        tags |= {norm_cls(k) for k, v in attrs.items() if v is True or str(v).strip().lower() in _TRUE}
    if img.get(NO_DAMAGE) is True:
        tags.add(norm_cls(NO_DAMAGE))
    return tags


def _ann_part(ann: dict[str, Any], cat_name: str) -> tuple[str | None, str]:
    """(part label, damage class name): attribute 'part' (CVAT), ann['part'], or category 'damage__part'."""
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    part = attrs.get("part") or ann.get("part")
    if "__" in cat_name:
        dmg, _, cat_part = cat_name.partition("__")
        return (str(part) if part else cat_part), dmg
    return (str(part) if part else None), cat_name


def parse_coco(coco: dict[str, Any], slices: dict[str, dict[str, str]], images_root: Path) -> Dataset:
    """GT images from a COCO dict. Raises DataError on RLE / iscrowd segmentations (polygons only)."""
    if not isinstance(coco, dict) or not isinstance(coco.get("images"), list):
        raise DataError("annotations.json is not a COCO file (no 'images' list)")
    cats = {int(c["id"]): str(c.get("name", c["id"])) for c in coco.get("categories", [])}
    by_name: dict[str, list[Path]] = {}
    if images_root.is_dir():
        for p in images_root.rglob("*"):
            if p.is_file():
                by_name.setdefault(p.name, []).append(p)
    no_dmg_key = norm_cls(NO_DAMAGE)
    images: dict[int, GtImage] = {}
    warnings: list[str] = []
    for im in coco["images"]:
        fname = str(im.get("file_name", "")).replace("\\", "/")
        path = _resolve_image(fname, images_root, by_name)
        rel = fname
        if path is not None:
            try:
                rel = path.resolve().relative_to(images_root.resolve()).as_posix()
            except ValueError:
                rel = fname
        row = slices.get(rel) or slices.get(fname) or next(
            (r for k, r in slices.items() if Path(k).name == Path(fname).name), {})
        parent = Path(rel).parent.name
        slc = (row.get("slice") or (parent if parent in SLICE_NAMES else UNKNOWN)).lower()
        images[int(im["id"])] = GtImage(
            file=rel, path=path, width=int(im.get("width") or 0), height=int(im.get("height") or 0),
            slice=slc, brand=row.get("brand", "").strip(), car=row.get("car", "").strip() or Path(rel).stem,
            no_damage=(row.get("no_damage", "").lower() in _TRUE) or no_dmg_key in _image_tags(im))
    n_rect = n_unknown_img = 0
    for ann in coco.get("annotations", []):
        img = images.get(int(ann.get("image_id", -1)))
        if img is None:
            n_unknown_img += 1
            continue
        cat_name = cats.get(int(ann.get("category_id", -1)), "")
        part_label, dmg_name = _ann_part(ann, cat_name)
        cls = norm_cls(dmg_name)
        if cls == no_dmg_key:
            img.no_damage = True
            continue
        seg = ann.get("segmentation")
        if isinstance(seg, dict) or int(ann.get("iscrowd", 0) or 0):
            raise DataError(f"annotation {ann.get('id')} on {img.file} is RLE / iscrowd: export polygons "
                            "(CVAT: COCO 1.0, no masks; see docs/research/labelling_guide.md)")
        polys = [p for p in (seg or []) if isinstance(p, list) and len(p) >= 6]
        if not polys:
            n_rect += 1
            continue
        if img.width <= 0 or img.height <= 0:
            raise DataError(f"{img.file}: width/height missing in annotations.json")
        mask = CropMask.from_full(rasterize_polygons(polys, img.height, img.width))
        if mask is None:
            warnings.append(f"annotation {ann.get('id')} on {img.file} rasterises to 0 px: ignored")
            continue
        img.instances.append(GtInstance(cls, part_key(part_label), part_label, mask))
    if n_rect:
        warnings.append(f"{n_rect} annotation(s) without a polygon (rectangles / empty) were ignored")
    if n_unknown_img:
        warnings.append(f"{n_unknown_img} annotation(s) reference an unknown image id")
    n_no_part = sum(1 for i in images.values() for g in i.instances if g.part is None)
    if n_no_part:
        warnings.append(f"{n_no_part} damage polygon(s) have no 'part' attribute: excluded from part recall")
    clash = [i.file for i in images.values() if i.no_damage and i.instances]
    if clash:
        warnings.append(f"{len(clash)} image(s) tagged no_damage also have damage polygons: treated as damaged")
        for i in images.values():
            if i.no_damage and i.instances:
                i.no_damage = False
    classes = sorted({g.cls for i in images.values() for g in i.instances})
    return Dataset(list(images.values()), classes, warnings)


def data_status(annotations: Path = ANNOTATIONS, slices_csv: Path = SLICES_CSV, images_root: Path = IMAGES,
                vin_gt: Path = VIN_GT, vin_dir: Path = VIN_DIR) -> tuple[Dataset | None, list[str]]:
    """(dataset or None, human-readable lines saying what is present and what is missing). Never raises."""
    lines: list[str] = []
    n_img_files = sum(1 for p in images_root.rglob("*.jpg")) if images_root.is_dir() else 0
    lines.append(f"blurred images in {_rel(images_root)}: {n_img_files}"
                 + ("" if n_img_files else "  -> MISSING: run scripts/blur_testset.py"))
    lines.append(f"slices file {_rel(slices_csv)}: " + ("present" if slices_csv.is_file()
                 else "MISSING (written by scripts/blur_testset.py)"))
    ds: Dataset | None = None
    if not annotations.is_file():
        lines.append(f"annotations {_rel(annotations)}: MISSING -> label the blurred images and export COCO 1.0 "
                     "(docs/research/labelling_guide.md)")
    else:
        try:
            ds = parse_coco(json.loads(annotations.read_text(encoding="utf-8")), read_slices(slices_csv),
                            images_root)
        except (ValueError, KeyError, TypeError, DataError) as exc:
            lines.append(f"annotations {_rel(annotations)}: UNUSABLE ({type(exc).__name__}: {exc})")
        else:
            usable = [i for i in ds.images if i.labelled and i.path is not None]
            missing = [i for i in ds.images if i.labelled and i.path is None]
            unlabelled = [i for i in ds.images if not i.labelled]
            per_slice = {s: sum(1 for i in usable if i.slice == s) for s in sorted({i.slice for i in usable})}
            lines.append(f"annotations {_rel(annotations)}: {len(ds.images)} images, {len(usable)} labelled with an "
                         f"image file {per_slice}, {sum(len(i.instances) for i in usable)} damage polygons, "
                         f"{sum(i.no_damage for i in usable)} tagged no_damage")
            if missing:
                lines.append(f"  {len(missing)} labelled image(s) have no file under {_rel(images_root)}")
            if unlabelled:
                lines.append(f"  {len(unlabelled)} image(s) have no polygon and no no_damage tag: not evaluated")
            lines += [f"  warning: {w}" for w in ds.warnings]
    lines.append(f"VIN ground truth {_rel(vin_gt)}: " + ("present" if vin_gt.is_file() else "missing (VIN rate skipped)")
                 + f"; VIN photos in {_rel(vin_dir)}: "
                 + str(sum(1 for p in vin_dir.iterdir() if p.is_file()) if vin_dir.is_dir() else 0))
    return ds, lines


def usable_images(ds: Dataset | None) -> list[GtImage]:
    return [] if ds is None else [i for i in ds.images if i.labelled and i.path is not None]


# ------------------------------------------------------------------------------------------------ matching

@dataclass
class Pred:
    cls: str                 # norm_cls
    score: float
    mask: CropMask
    part: str | None         # part_key from mapping.assign (None when 'unknown')


@dataclass
class MatchTable:
    """Every prediction of one model on the evaluated images, matched once at IoU_MATCH (greedy by score).

    Greedy matching in descending score order means that the matches of any score prefix (any confidence
    threshold) are the prefix of these matches, so thresholds and image subsets are applied afterwards.
    """

    images: list[GtImage]
    preds: list[list[Pred]]
    p_img: np.ndarray        # (P,) image index
    p_cls: list[str]
    p_score: np.ndarray      # (P,)
    p_tp: np.ndarray         # (P,) bool, matched at IoU >= IOU_MATCH
    p_gt: np.ndarray         # (P,) global GT index matched (-1)
    p_pos: list[tuple[int, int]]       # (image index, index in preds[image])
    p_hits: list[frozenset[str]]       # (image, part) keys this prediction finds, rule (a) or (b)
    p_hits_map: list[frozenset[str]]   # rule (a) only
    g_img: np.ndarray        # (G,) image index
    g_cls: list[str]
    g_part: list[str | None]


def match(images: Sequence[GtImage], preds: Sequence[Sequence[Pred]], iou_thr: float = IOU_MATCH,
          part_iou: float = IOU_PART) -> MatchTable:
    """Greedy per-class matching on each image + the part-recall hits of every prediction."""
    g_img, g_cls, g_part, g_masks, g_offset = [], [], [], [], []
    for k, im in enumerate(images):
        g_offset.append(len(g_img))
        for g in im.instances:
            g_img.append(k), g_cls.append(g.cls), g_part.append(g.part), g_masks.append(g.mask)
    rows: list[tuple[int, int, float, str]] = [(k, j, p.score, p.cls) for k, ps in enumerate(preds) for j, p in
                                               enumerate(ps)]
    rows.sort(key=lambda r: -r[2])  # stable: ties keep image / model order
    tp = np.zeros(len(rows), bool)
    gt_idx = np.full(len(rows), -1, np.int64)
    used = np.zeros(len(g_img), bool)
    hits, hits_map = [], []
    for r, (k, j, _score, cls) in enumerate(rows):
        im, p = images[k], preds[k][j]
        best, best_g = -1.0, -1
        found: set[str] = set()
        for li, g in enumerate(im.instances):
            gi = g_offset[k] + li
            iou = mask_iou(p.mask, g.mask)
            if g.part is not None and iou >= part_iou:
                found.add(g.part)
            if g.cls == cls and not used[gi] and iou >= iou_thr and iou > best:   # ties: first GT
                best, best_g = iou, gi
        if best_g >= 0:
            used[best_g], tp[r], gt_idx[r] = True, True, best_g
        gt_parts = {g.part for g in im.instances if g.part is not None}
        by_map = {p.part} & gt_parts if p.part is not None else set()
        hits.append(frozenset(found | by_map))
        hits_map.append(frozenset(by_map))
    return MatchTable(
        images=list(images), preds=[list(p) for p in preds],
        p_img=np.array([r[0] for r in rows], np.int64), p_cls=[r[3] for r in rows],
        p_score=np.array([r[2] for r in rows], np.float64), p_tp=tp, p_gt=gt_idx,
        p_pos=[(r[0], r[1]) for r in rows], p_hits=hits, p_hits_map=hits_map,
        g_img=np.array(g_img, np.int64), g_cls=g_cls, g_part=g_part)


def average_precision(tp_sorted: np.ndarray, n_gt: int) -> float | None:
    """101-point interpolated AP (COCO) from TP flags in descending score order; None when n_gt == 0."""
    if n_gt <= 0:
        return None
    tp = np.asarray(tp_sorted, bool)
    if tp.size == 0:
        return 0.0
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, tp.size + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0.0, 1.0, 101), side="left")
    q = np.where(idx < tp.size, precision[np.minimum(idx, tp.size - 1)], 0.0)
    return float(q.mean())


def calibrate_threshold(scores: np.ndarray, tp: np.ndarray, n_gt: int, default: float = BASELINE_CONF
                        ) -> tuple[float, str]:
    """Confidence threshold maximising F1 at IoU 0.5 for one class: keep predictions with score >= threshold.

    Candidate thresholds are the prediction scores (all predictions with an equal score are kept or dropped
    together); ties in F1 go to the HIGHER threshold. No GT or no TP -> (default, reason).
    """
    scores, tp = np.asarray(scores, np.float64), np.asarray(tp, bool)
    if n_gt <= 0:
        return default, "no GT in the calibration images: baseline kept"
    if scores.size == 0 or not tp.any():
        return default, "no true positive in the calibration images: baseline kept"
    order = np.argsort(-scores, kind="mergesort")
    s, t = scores[order], tp[order]
    f1 = 2 * np.cumsum(t) / (np.arange(1, s.size + 1) + n_gt)
    cut = np.r_[s[1:] < s[:-1], True]          # last prediction of each equal-score group
    f1 = np.where(cut, f1, -1.0)
    best = int(np.argmax(f1))                  # first max = highest threshold
    return float(s[best]), f"max F1 {f1[best]:.3f}"


def _subset(t: MatchTable, img_mask: np.ndarray) -> np.ndarray:
    return img_mask[t.p_img] if t.p_img.size else np.zeros(0, bool)


def ap_table(t: MatchTable, classes: Sequence[str], img_mask: np.ndarray) -> dict[str, float | None]:
    """AP50 per class on the images where img_mask is True (predictions already score-sorted)."""
    sel = _subset(t, img_mask)
    g_sel = img_mask[t.g_img] if t.g_img.size else np.zeros(0, bool)
    out: dict[str, float | None] = {}
    p_cls, g_cls = np.array(t.p_cls, dtype=object), np.array(t.g_cls, dtype=object)
    for c in classes:
        n_gt = int(np.count_nonzero(g_sel & (g_cls == c))) if g_cls.size else 0
        m = sel & (p_cls == c) if p_cls.size else np.zeros(0, bool)
        out[c] = average_precision(t.p_tp[m], n_gt)
    return out


def mean_ap(aps: dict[str, float | None]) -> float | None:
    vals = [v for v in aps.values() if v is not None]
    return float(np.mean(vals)) if vals else None


def keep_mask(t: MatchTable, thresholds: dict[str, float], default: float = BASELINE_CONF) -> np.ndarray:
    return np.array([s >= thresholds.get(c, default) for s, c in zip(t.p_score, t.p_cls)], bool)


def calibrate(t: MatchTable, classes: Sequence[str], img_mask: np.ndarray, default: float = BASELINE_CONF
              ) -> dict[str, dict[str, Any]]:
    """Per-class calibrated threshold on the images where img_mask is True."""
    sel = _subset(t, img_mask)
    g_sel = img_mask[t.g_img] if t.g_img.size else np.zeros(0, bool)
    p_cls, g_cls = np.array(t.p_cls, dtype=object), np.array(t.g_cls, dtype=object)
    out: dict[str, dict[str, Any]] = {}
    for c in classes:
        m = sel & (p_cls == c) if p_cls.size else np.zeros(0, bool)
        n_gt = int(np.count_nonzero(g_sel & (g_cls == c))) if g_cls.size else 0
        thr, why = calibrate_threshold(t.p_score[m], t.p_tp[m], n_gt, default)
        out[c] = {"threshold": thr, "rule": why, "n_gt": n_gt}
    return out


def threshold_metrics(t: MatchTable, keep: np.ndarray, img_mask: np.ndarray, classes: Sequence[str]
                      ) -> dict[str, Any]:
    """P/R/F1 per class, part recall, FP per clean car and per image for the kept predictions on img_mask."""
    sel = _subset(t, img_mask) & keep if t.p_img.size else np.zeros(0, bool)
    g_sel = img_mask[t.g_img] if t.g_img.size else np.zeros(0, bool)
    p_cls, g_cls = np.array(t.p_cls, dtype=object), np.array(t.g_cls, dtype=object)
    per_class: dict[str, dict[str, Any]] = {}
    for c in classes:
        m = sel & (p_cls == c) if p_cls.size else np.zeros(0, bool)
        n_tp = int(np.count_nonzero(t.p_tp[m]))
        n_fp = int(np.count_nonzero(m)) - n_tp
        n_gt = int(np.count_nonzero(g_sel & (g_cls == c))) if g_cls.size else 0
        p = n_tp / (n_tp + n_fp) if n_tp + n_fp else None
        r = n_tp / n_gt if n_gt else None
        f1 = 2 * n_tp / (2 * n_tp + n_fp + (n_gt - n_tp)) if (2 * n_tp + n_fp + n_gt - n_tp) else None
        per_class[c] = {"n_gt": n_gt, "tp": n_tp, "fp": n_fp, "precision": p, "recall": r, "f1": f1}
    # part recall
    idx = [k for k in range(len(t.images)) if img_mask[k]]
    gt_parts = {(k, g.part) for k in idx for g in t.images[k].instances if g.part is not None}
    found, found_map = set(), set()
    for r in np.flatnonzero(sel):
        k = int(t.p_img[r])
        found |= {(k, p) for p in t.p_hits[r]}
        found_map |= {(k, p) for p in t.p_hits_map[r]}
    found &= gt_parts
    found_map &= gt_parts
    # false positives
    clean = [k for k in idx if t.images[k].no_damage]
    clean_cars = {t.images[k].car for k in clean}
    fp_clean = sum(int(np.count_nonzero(sel & (t.p_img == k))) for k in clean)
    fp_all = int(np.count_nonzero(sel & ~t.p_tp))
    return {
        "per_class": per_class,
        "part_recall": {"n_parts": len(gt_parts), "found": len(found), "found_by_mapping": len(found_map),
                        "recall": len(found) / len(gt_parts) if gt_parts else None},
        "fp": {"clean_images": len(clean), "clean_cars": len(clean_cars), "fp_on_clean": fp_clean,
               "fp_per_clean_car": fp_clean / len(clean_cars) if clean_cars else None,
               "n_images": len(idx), "fp_all": fp_all, "fp_per_image": fp_all / len(idx) if idx else None},
    }


# ------------------------------------------------------------------------------------------------ splits

def split_folds(cars: Sequence[str], spec: str, seed: int = 0) -> list[set[str]]:
    """Held-out car groups: 'kfold:K' -> K disjoint groups covering all cars; 'holdout:F' -> one group with
    ceil(F * n_cars) cars; 'none' -> []. Deterministic for a seed. Split by car, never by image."""
    spec = (spec or "none").strip().lower()
    if spec == "none":
        return []
    kind, _, val = spec.partition(":")
    uniq = sorted(set(cars))
    random.Random(seed).shuffle(uniq)
    if kind == "kfold":
        k = int(val)
        if k < 2 or k > len(uniq):
            raise ValueError(f"kfold needs 2 <= K <= number of cars ({len(uniq)}), got {k}")
        return [set(uniq[i::k]) for i in range(k)]
    if kind == "holdout":
        f = float(val)
        if not 0 < f < 1:
            raise ValueError(f"holdout fraction must be in (0, 1), got {f}")
        n = min(len(uniq) - 1, max(1, math.ceil(f * len(uniq))))
        if n < 1:
            raise ValueError("holdout needs at least 2 cars")
        return [set(uniq[:n])]
    raise ValueError(f"--split must be none, kfold:K or holdout:F, got {spec!r}")


def slice_masks(images: Sequence[GtImage]) -> dict[str, np.ndarray]:
    """all, day / night / wet (and 'unknown' when present), brand:<X> for every non-empty brand."""
    out = {"all": np.ones(len(images), bool)}
    for s in (*SLICE_NAMES, *sorted({i.slice for i in images} - set(SLICE_NAMES))):
        m = np.array([i.slice == s for i in images], bool)
        if m.any() or s in SLICE_NAMES:
            out[s] = m
    for b in sorted({i.brand for i in images if i.brand}, key=str.lower):
        out[f"brand:{b}"] = np.array([i.brand == b for i in images], bool)
    return out


# ------------------------------------------------------------------------------------------------ evaluation

def evaluate_model(t: MatchTable, model_classes: Sequence[str], gt_classes: Sequence[str], *,
                   split: str = "none", seed: int = 0, baseline: float = BASELINE_CONF) -> dict[str, Any]:
    """All metrics of one model from its MatchTable (pure; no models, no files)."""
    classes = [c for c in gt_classes if c in set(model_classes)]
    all_classes = sorted(set(model_classes) | set(gt_classes))
    slices = slice_masks(t.images)
    everything = slices["all"]
    base_thr = {c: baseline for c in all_classes}
    keep_base = keep_mask(t, base_thr, baseline)
    calib = calibrate(t, model_classes, everything, baseline)
    cal_thr = {c: v["threshold"] for c, v in calib.items()}
    keep_cal = keep_mask(t, cal_thr, baseline)
    res: dict[str, Any] = {
        "eval_classes": classes,
        "gt_classes_not_in_model": [c for c in gt_classes if c not in set(model_classes)],
        "slices": {},
        "calibration": {"thresholds": calib, "baseline_conf": baseline},
    }
    for name, m in slices.items():
        aps = ap_table(t, classes, m)
        res["slices"][name] = {
            "n_images": int(m.sum()),
            "n_gt": int(np.count_nonzero(m[t.g_img])) if t.g_img.size else 0,
            "ap50": aps, "map50": mean_ap(aps),
            "baseline": threshold_metrics(t, keep_base, m, all_classes),
            "calibrated_in_sample": threshold_metrics(t, keep_cal, m, all_classes),
        }
    folds = split_folds([i.car for i in t.images], split, seed)
    if folds:
        held = np.zeros(len(t.images), bool)
        keep_held = np.zeros(t.p_score.size, bool)
        fold_thr = []
        for fold in folds:
            test = np.array([i.car in fold for i in t.images], bool)
            c = calibrate(t, model_classes, ~test, baseline)
            thr = {k: v["threshold"] for k, v in c.items()}
            fold_thr.append(thr)
            in_test = test[t.p_img] if t.p_img.size else np.zeros(0, bool)
            keep_held |= in_test & keep_mask(t, thr, baseline)
            held |= test
        res["calibration"]["split"] = {
            "spec": split, "seed": seed, "n_folds": len(folds), "held_out_images": int(held.sum()),
            "fold_thresholds": fold_thr,
            "baseline_on_held_out": threshold_metrics(t, keep_base, held, all_classes),
            "calibrated_held_out": threshold_metrics(t, keep_held, held, all_classes),
        }
    return res


def failure_cases(t: MatchTable, keep: np.ndarray, n: int) -> list[dict[str, Any]]:
    """Up to n images with the most errors (missed GT + FP) at the kept predictions; ties by file name."""
    matched = set(int(g) for g in t.p_gt[keep & t.p_tp])
    rows = []
    g_off = 0
    for k, im in enumerate(t.images):
        missed = [li for li in range(len(im.instances)) if (g_off + li) not in matched]
        g_off += len(im.instances)
        fp = [t.p_pos[r][1] for r in np.flatnonzero(keep & ~t.p_tp & (t.p_img == k))]
        tp = [t.p_pos[r][1] for r in np.flatnonzero(keep & t.p_tp & (t.p_img == k))]
        if missed or fp:
            rows.append({"image": k, "file": im.file, "missed": missed, "fp": fp, "tp": tp,
                         "n_errors": len(missed) + len(fp)})
    rows.sort(key=lambda r: (-r["n_errors"], r["file"]))
    return rows[:n]


# ------------------------------------------------------------------------------------------------ inference

def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_inference(images: Sequence[GtImage], model_keys: Sequence[str], *, device: str, min_conf: float,
                  baseline: float, latency_n: int, parts_key: str = PARTS_MODEL,
                  log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run the parts model and every damage model on the blurred images.

    Returns {"preds": {model: [[Pred]...]}, "classes": {model: [norm_cls]}, "latency": {...},
    "errors": [...], "ok_images": [bool per image]}.
    """
    from avtopassport.io import load_image
    from avtopassport.pipeline.mapping import UNKNOWN_PART, assign
    from avtopassport.pipeline.segment import load_segmenter

    errors: list[str] = []
    segs: dict[str, Any] = {}
    for key in (parts_key, *model_keys):
        try:
            segs[key] = load_segmenter(key, device=device)
        except Exception as exc:
            errors.append(f"{key}: could not load ({type(exc).__name__}: {exc})")
            log(f"WARNING {errors[-1]}")
    parts = segs.get(parts_key)
    dmg = {k: segs[k] for k in model_keys if k in segs}
    for s in dmg.values():
        s.conf = min_conf
    preds: dict[str, list[list[Pred]]] = {k: [] for k in dmg}
    ok: list[bool] = []
    for n, im in enumerate(images):
        try:
            img = load_image(im.path) if im.path is not None else None
            if img is None:
                raise FileNotFoundError(im.file)
            if img.shape[0] != im.height or img.shape[1] != im.width:
                raise ValueError(f"image is {img.shape[1]}x{img.shape[0]}, annotations say {im.width}x{im.height}")
            part_segs = parts.segment(img) if parts is not None else []
            per_model = {}
            for key, s in dmg.items():
                found = s.segment(img)
                obs = assign(im.file, found, part_segs)
                out = []
                for sg, ob in zip(found, obs):
                    mask = sg.mask if sg.mask is not None else _box_mask(sg.box, img.shape[0], img.shape[1])
                    cm = CropMask.from_full(mask, sg.box)
                    if cm is not None:
                        out.append(Pred(norm_cls(sg.cls), float(sg.score), cm,
                                        None if ob.part == UNKNOWN_PART else part_key(ob.part)))
                per_model[key] = out
        except Exception as exc:
            errors.append(f"{im.file}: {type(exc).__name__}: {exc}")
            log(f"WARNING {errors[-1]}")
            ok.append(False)
            continue
        ok.append(True)
        for key, out in per_model.items():
            preds[key].append(out)
        if (n + 1) % 10 == 0 or n + 1 == len(images):
            log(f"inference {n + 1}/{len(images)} images")
    latency = measure_latency([im for im, good in zip(images, ok) if good], {**dmg, **({parts_key: parts} if parts
                              else {})}, baseline=baseline, n=latency_n, device=device, parts_key=parts_key)
    for s in dmg.values():
        s.conf = min_conf
    return {"preds": preds, "classes": {k: sorted({norm_cls(v) for v in s.names.values()}) for k, s in dmg.items()},
            "latency": latency, "errors": errors, "ok_images": ok}


def _box_mask(box: Sequence[float], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
    return m


def measure_latency(images: Sequence[GtImage], segs: dict[str, Any], *, baseline: float, n: int, device: str,
                    parts_key: str = PARTS_MODEL) -> dict[str, Any]:
    """Median / p90 ms of segment() per image at the baseline conf (parts model at its own conf), after one
    warm-up image. n = 0 disables."""
    from avtopassport.io import load_image

    out: dict[str, Any] = {}
    sample = list(images[:n]) if n > 0 else []
    if not sample:
        return out
    imgs = [load_image(i.path) for i in sample]
    for key, s in segs.items():
        if key != parts_key:
            s.conf = baseline
        s.segment(imgs[0])  # warm-up
        times = []
        for img in imgs:
            t0 = time.perf_counter()
            s.segment(img)
            times.append((time.perf_counter() - t0) * 1000.0)
        out[key] = {"device": device, "n_images": len(times), "conf": s.conf, "imgsz": s.imgsz,
                    "median_ms": statistics.median(times), "p90_ms": float(np.percentile(times, 90)),
                    "image_px": f"{imgs[0].shape[1]}x{imgs[0].shape[0]}"}
    return out


# ------------------------------------------------------------------------------------------------ VIN

def eval_vin(gt_csv: Path = VIN_GT, vin_dir: Path = VIN_DIR) -> dict[str, Any]:
    """VIN exact-match rate on data/raw/testset/vin (rates and counts only; no VIN leaves this function)."""
    if not gt_csv.is_file():
        return {"status": "skipped", "reason": f"{_rel(gt_csv)} not found"}
    with gt_csv.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("file") or "").strip() and (r.get("vin") or "").strip()]
    if not rows:
        return {"status": "skipped", "reason": f"{_rel(gt_csv)} has no rows with file and vin"}
    try:
        from avtopassport.io import load_image
        from avtopassport.vin.ocr import read_vin
    except Exception as exc:
        return {"status": "skipped", "reason": f"VIN OCR unavailable ({type(exc).__name__})"}
    counts = {"read": 0, "ambiguous": 0, "could_not_read": 0}
    n = exact = top_exact = missing = 0
    for r in rows:
        p = vin_dir / r["file"].strip()
        if not p.is_file():
            missing += 1
            continue
        truth = re.sub(r"\s+", "", r["vin"]).upper()
        try:
            res = read_vin(load_image(p))
        except Exception:
            counts["could_not_read"] += 1
            n += 1
            continue
        n += 1
        counts[res.status] = counts.get(res.status, 0) + 1
        best = res.best.text if res.best is not None else None
        exact += int(res.status == "read" and best == truth)
        top_exact += int(best == truth)
    return {"status": "done" if n else "skipped", "n_photos": n, "missing_files": missing,
            "exact_match": exact, "exact_match_rate": exact / n if n else None,
            "top_candidate_match_rate": top_exact / n if n else None, "status_counts": counts,
            "rule": "exact = status 'read' and the 17 characters equal the ground truth; ambiguous counts as a miss"}


# ------------------------------------------------------------------------------------------------ gallery

def draw_gallery(t: MatchTable, keep: np.ndarray, model: str, n: int, out_dir: Path = GALLERY
                 ) -> list[dict[str, Any]]:
    """Annotated BLURRED images: missed GT red, FP yellow, TP green. Replaces <model>_*.jpg in out_dir."""
    import cv2

    from avtopassport.io import load_image, save_jpeg

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{model}_*.jpg"):
        old.unlink()
    written = []
    for i, case in enumerate(failure_cases(t, keep, n), 1):
        im = t.images[case["image"]]
        img = load_image(im.path)
        h, w = img.shape[:2]
        lw = max(2, round(max(h, w) / 500))

        def outline(cm: CropMask, colour: tuple[int, int, int], label: str) -> None:
            m = cm.full(h, w).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, colour, lw)
            cv2.putText(img, label, (cm.x0, max(12, cm.y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, lw * 0.35, colour, lw)

        for j in case["tp"]:
            p = t.preds[case["image"]][j]
            outline(p.mask, (0, 200, 0), f"TP {p.cls} {p.score:.2f}")
        for j in case["fp"]:
            p = t.preds[case["image"]][j]
            outline(p.mask, (0, 255, 255), f"FP {p.cls} {p.score:.2f}")
        for li in case["missed"]:
            g = im.instances[li]
            outline(g.mask, (0, 0, 255), f"MISSED {g.cls}" + (f" @{g.part_label}" if g.part_label else ""))
        path = save_jpeg(img, out_dir / f"{model}_{i}.jpg", max_side=1280)
        written.append({"file": _rel(path), "source": im.file, "slice": im.slice, "missed": len(case["missed"]),
                        "fp": len(case["fp"]), "tp": len(case["tp"])})
    return written


# ------------------------------------------------------------------------------------------------ rendering

def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def f3(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(v: Any) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def render_blocks(metrics: dict[str, Any]) -> dict[str, str]:
    """Markdown body of every block from metrics.json (pure)."""
    run, models = metrics["run"], metrics["models"]
    head = f"<!-- generated by scripts/eval.py, run {run}; do not edit by hand -->\n"
    ds = metrics["dataset"]
    split = metrics["config"]["split"]
    b: dict[str, str] = {}
    rows = []
    for k, m in models.items():
        a = m["slices"]["all"]
        sp = m["calibration"].get("split")
        cal = sp["calibrated_held_out"] if sp else a["calibrated_in_sample"]
        lat = metrics["latency"].get(k, {})
        rows.append([k, f3(a["map50"]), f3(a["baseline"]["part_recall"]["recall"]), f3(cal["part_recall"]["recall"]),
                     f3(a["baseline"]["fp"]["fp_per_clean_car"]), f3(cal["fp"]["fp_per_clean_car"]),
                     f3(lat.get("median_ms"))])
    b["summary"] = head + (
        f"Run `{run}` on `{metrics['config']['device']}`, {metrics['generated_utc']}. Test set: {ds['n_images']} labelled "
        f"images ({', '.join(f'{s}: {n}' for s, n in ds['per_slice'].items())}), {ds['n_cars']} cars, "
        f"{ds['n_gt']} damage polygons, {ds['n_parts']} damaged parts, {ds['n_clean_images']} images tagged no_damage. "
        f"Calibrated columns: {'held-out (' + split + ')' if split != 'none' else 'IN-SAMPLE (optimistic; use --split)'}."
        "\n\n" + md_table(["model", "mask mAP50 (all)", "part recall @0.25", "part recall calibrated",
                           "FP / clean car @0.25", "FP / clean car calibrated", "latency median ms"], rows))
    rows = []
    for k, m in models.items():
        for c, ap in m["slices"]["all"]["ap50"].items():
            pc = m["slices"]["all"]["baseline"]["per_class"].get(c, {})
            rows.append([k, c, pc.get("n_gt", 0), f3(ap)])
        for c in m["gt_classes_not_in_model"]:
            rows.append([k, c, "", "not in model vocabulary"])
    b["per_class"] = head + md_table(["model", "class", "GT instances", "mask AP50"], rows)
    rows = []
    for k, m in models.items():
        for s, v in m["slices"].items():
            rows.append([k, s, v["n_images"], v["n_gt"], f3(v["map50"]), f3(v["baseline"]["part_recall"]["recall"]),
                         f3(v["baseline"]["fp"]["fp_per_image"])])
    b["per_slice"] = head + md_table(["model", "slice", "images", "GT", "mask mAP50", "part recall @0.25",
                                      "FP / image @0.25"], rows)
    rows = []
    for k, m in models.items():
        for s, v in m["slices"].items():
            pb, pc = v["baseline"]["part_recall"], v["calibrated_in_sample"]["part_recall"]
            rows.append([k, s, pb["n_parts"], pb["found"], pb["found_by_mapping"], f3(pb["recall"]), f3(pc["recall"])])
    b["part_recall"] = head + (
        f"Rule: a labelled damaged (image, part) is found when a kept prediction of any damage class is assigned to "
        f"that part by `mapping.assign`, or has mask IoU >= {metrics['config']['iou_part']} with a GT polygon of that "
        "part. Calibrated column is in-sample.\n\n" + md_table(
            ["model", "slice", "damaged parts", "found @0.25", "of which by mapping", "recall @0.25",
             "recall calibrated (in-sample)"], rows))
    rows = []
    for k, m in models.items():
        a = m["slices"]["all"]
        sp = m["calibration"].get("split")
        for label, v in (("baseline 0.25", a["baseline"]), ("calibrated in-sample", a["calibrated_in_sample"]),
                         *((("baseline 0.25, held-out", sp["baseline_on_held_out"]),
                            ("calibrated, held-out", sp["calibrated_held_out"])) if sp else ())):
            fp = v["fp"]
            rows.append([k, label, fp["clean_cars"], fp["fp_on_clean"], f3(fp["fp_per_clean_car"]), fp["n_images"],
                         f3(fp["fp_per_image"])])
    b["fp_per_car"] = head + md_table(["model", "thresholds", "clean cars", "FP on clean cars", "FP / clean car",
                                       "images", "FP / image (all)"], rows)
    rows = []
    for k, m in models.items():
        cal = m["calibration"]
        sp = cal.get("split")
        a = m["slices"]["all"]
        for c, th in cal["thresholds"].items():
            pb = a["baseline"]["per_class"].get(c, {})
            pc = a["calibrated_in_sample"]["per_class"].get(c, {})
            ph = sp["calibrated_held_out"]["per_class"].get(c, {}) if sp else {}
            pbh = sp["baseline_on_held_out"]["per_class"].get(c, {}) if sp else {}
            rows.append([k, c, th["n_gt"], f3(th["threshold"]), th["rule"], f3(pb.get("f1")), f3(pc.get("f1")),
                         f3(pbh.get("f1")) if sp else "", f3(ph.get("f1")) if sp else ""])
    b["calibration"] = head + (
        f"Threshold per class = the score maximising F1 at IoU {metrics['config']['iou_match']} on all images "
        "(in-sample, optimistic). Held-out columns: thresholds calibrated without the held-out cars "
        f"(split `{split}`, seed {metrics['config']['seed']}).\n\n" + md_table(
            ["model", "class", "GT", "calibrated threshold", "rule", "F1 @0.25", "F1 calibrated (in-sample)",
             "F1 @0.25 held-out", "F1 calibrated held-out"], rows))
    v = metrics["vin"]
    b["vin"] = head + (f"_not measured: {v.get('reason')}_" if v.get("status") != "done" else md_table(
        ["VIN photos", "exact match", "exact-match rate", "top-candidate match rate", "read / ambiguous / could not read",
         "missing files"],
        [[v["n_photos"], v["exact_match"], f3(v["exact_match_rate"]), f3(v["top_candidate_match_rate"]),
          " / ".join(str(v["status_counts"].get(s, 0)) for s in ("read", "ambiguous", "could_not_read")),
          v["missing_files"]]]) + f"\n\nRule: {v['rule']}.")
    rows = [[k, l["device"], l["imgsz"], l["conf"], l["image_px"], l["n_images"], f3(l["median_ms"]), f3(l["p90_ms"])]
            for k, l in metrics["latency"].items()]
    b["latency"] = head + ("_not measured (--latency-n 0)_" if not rows else (
        "Per image, `segment()` end to end (predict + mask rasterisation), after one warm-up image; damage models at "
        "conf 0.25. Full-inspection latency on the deploy CPU is measured in Phase 4.\n\n" + md_table(
            ["model", "device", "imgsz", "conf", "image size", "images", "median ms", "p90 ms"], rows)))
    rows = [[k, f"[{Path(g['file']).name}](../assets/failure_cases/{Path(g['file']).name})", g["slice"], g["missed"],
             g["fp"], g["tp"]] for k, gs in metrics["gallery"].items() for g in gs]
    b["failures"] = head + ("_no failure cases written_" if not rows else (
        "Calibrated (in-sample) thresholds; red = missed GT, yellow = false positive, green = true positive. "
        "Images are the blurred test-set copies.\n\n" + md_table(
            ["model", "image", "slice", "missed GT", "FP", "TP"], rows)))
    return b


def replace_blocks(text: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """Replace each block body between its markers; returns (text, names skipped because markers are broken)."""
    skipped = []
    for name, body in blocks.items():
        begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
        if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
            skipped.append(name)
            continue
        i, j = text.index(begin) + len(begin), text.index(end)
        text = text[:i] + "\n" + body.strip("\n") + "\n" + text[j:]
    return text, skipped


# ------------------------------------------------------------------------------------------------ main

def dataset_summary(images: Sequence[GtImage]) -> dict[str, Any]:
    per_slice: dict[str, int] = {}
    for i in images:
        per_slice[i.slice] = per_slice.get(i.slice, 0) + 1
    return {"n_images": len(images), "n_cars": len({i.car for i in images}), "per_slice": per_slice,
            "brands": sorted({i.brand for i in images if i.brand}),
            "n_gt": sum(len(i.instances) for i in images),
            "n_parts": len({(n, g.part) for n, i in enumerate(images) for g in i.instances if g.part}),
            "n_clean_images": sum(i.no_damage for i in images),
            "gt_per_class": {c: sum(g.cls == c for i in images for g in i.instances)
                             for c in sorted({g.cls for i in images for g in i.instances})}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only print which inputs are present / missing")
    ap.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    ap.add_argument("--slices", type=Path, default=SLICES_CSV)
    ap.add_argument("--images", type=Path, default=IMAGES)
    ap.add_argument("--models", default=",".join(DAMAGE_MODELS), help="comma-separated segment.MODEL_SPECS keys")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--min-conf", type=float, default=MIN_CONF, help="lowest score kept for AP / calibration")
    ap.add_argument("--baseline-conf", type=float, default=BASELINE_CONF)
    ap.add_argument("--split", default="none", help="none | kfold:K | holdout:F (by car) for held-out calibration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latency-n", type=int, default=20, help="images timed per model (0: skip)")
    ap.add_argument("--gallery-n", type=int, default=12, help="failure-case images per model (0: none)")
    ap.add_argument("--no-vin", action="store_true", help="skip the VIN exact-match rate")
    ap.add_argument("--no-docs", action="store_true", help="do not touch docs/research/evaluation.md")
    ap.add_argument("--run", default=None, help="run id (default: UTC time + device)")
    ap.add_argument("--out", type=Path, default=EVAL_OUT, help=argparse.SUPPRESS)
    ap.add_argument("--doc", type=Path, default=DOC, help=argparse.SUPPRESS)
    ap.add_argument("--gallery-dir", type=Path, default=GALLERY, help=argparse.SUPPRESS)
    ap.add_argument("--vin-gt", type=Path, default=VIN_GT, help=argparse.SUPPRESS)
    ap.add_argument("--vin-dir", type=Path, default=VIN_DIR, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    ds, status = data_status(args.annotations, args.slices, args.images, args.vin_gt, args.vin_dir)
    print("\n".join(status))
    images = usable_images(ds)
    if not images:
        print(f"REFUSED: no labelled image to evaluate (need {_rel(args.annotations)} with at least one image that "
              f"has damage polygons or the no_damage tag, and its blurred file under {_rel(args.images)}). "
              "Nothing written; see docs/research/labelling_guide.md.", file=sys.stderr)
        return EXIT_NO_DATA
    if args.check:
        return EXIT_OK
    try:
        split_folds([i.car for i in images], args.split, args.seed)
    except ValueError as exc:
        print(f"bad --split: {exc}", file=sys.stderr)
        return EXIT_ERRORS
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    device = resolve_device(args.device)
    run = args.run or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{device.replace(':', '')}"
    t0 = time.perf_counter()
    inf = run_inference(images, model_keys, device=device, min_conf=args.min_conf, baseline=args.baseline_conf,
                        latency_n=args.latency_n)
    images = [im for im, good in zip(images, inf["ok_images"]) if good]
    if not images or not inf["preds"]:
        print("REFUSED: no image could be evaluated or no damage model loaded; nothing written", file=sys.stderr)
        print("\n".join(inf["errors"]), file=sys.stderr)
        return EXIT_NO_DATA
    gt_classes = sorted({g.cls for i in images for g in i.instances})
    models: dict[str, Any] = {}
    gallery: dict[str, list[dict[str, Any]]] = {}
    for key, preds in inf["preds"].items():
        t = match(images, preds)
        models[key] = evaluate_model(t, inf["classes"][key], gt_classes, split=args.split, seed=args.seed,
                                     baseline=args.baseline_conf)
        if args.gallery_n > 0:
            thr = {c: v["threshold"] for c, v in models[key]["calibration"]["thresholds"].items()}
            gallery[key] = draw_gallery(t, keep_mask(t, thr, args.baseline_conf), key, args.gallery_n,
                                        args.gallery_dir)
    metrics = {
        "run": run, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/eval.py",
        "config": {"device": device, "models": model_keys, "parts_model": PARTS_MODEL, "min_conf": args.min_conf,
                   "baseline_conf": args.baseline_conf, "iou_match": IOU_MATCH, "iou_part": IOU_PART,
                   "split": args.split, "seed": args.seed, "latency_n": args.latency_n,
                   "annotations": _rel(args.annotations)},
        "dataset": dataset_summary(images),
        "models": models, "latency": inf["latency"],
        "vin": {"status": "skipped", "reason": "--no-vin"} if args.no_vin else eval_vin(args.vin_gt, args.vin_dir),
        "gallery": gallery, "errors": inf["errors"], "warnings": ds.warnings if ds else [],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    out_dir = args.out / run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default) + "\n",
                                          encoding="utf-8")
    print(f"wrote {_rel(out_dir / 'metrics.json')}")
    blocks = render_blocks(metrics)
    if not args.no_docs:
        if args.doc.is_file():
            text, skipped = replace_blocks(args.doc.read_text(encoding="utf-8"), blocks)
            args.doc.write_text(text, encoding="utf-8")
            print(f"updated {_rel(args.doc)}" + (f" (markers broken, NOT updated: {skipped})" if skipped else ""))
        else:
            print(f"WARNING: {_rel(args.doc)} not found; tables only in metrics.json")
    print(blocks["summary"])
    return EXIT_ERRORS if inf["errors"] else EXIT_OK


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


if __name__ == "__main__":
    sys.exit(main())
