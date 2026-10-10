"""VIN OCR with RapidOCR (ONNX Runtime) + our validation.

STUB written by main; the vin writer implements the bodies without changing signatures.
Approach: detect text at RapidOCR's default size, scale boxes back to full resolution, crop each
line (padded) from the FULL-RES array, recognise crops with use_det=False (upscale only crops
< 32 px high); also try 90/270-degree rotations for vertical stickers. Build 17-char windows from
line text (strip spaces), validate each, rank:
  1. check digit valid where mandatory, 2. known WMI, 3. single-token (not stitched), 4. OCR score.
status: 'read' (one clear best), 'ambiguous' (>1 plausible, candidates returned), 'could_not_read'.

Implementation notes:
- rapidocr 3.10 keeps the use_det/use_cls/use_rec flags of the last call on the engine; every call
  here passes all three and restores the engine's previous flags afterwards, under one module lock:
  without it, concurrent read_vin calls on a shared engine (a web server's thread pool) interleaved
  their save/restore and left the engine with use_cls=use_rec=False.
- Detection input is downscaled with INTER_AREA to RapidOCR's ``Global.max_side_len`` (2000). A tall,
  thin input (more than DET_MAX_RATIO:1, or at most DET_MIN_SIDE px wide) is padded left/right first, as
  RapidOCR itself pads wide ones top/bottom: its detector scales the SHORT side up to 736 px with no cap
  on the long side, so a VIN strip turned 90 degrees became tens of megapixels. (Measured on the dev
  machine with synthetic renders: read_vin on a 60x3000 px (HxW) strip peaked at 5.0 GB RSS in 12.5 s and
  a 20x2000 px strip passed 6 GB and was stopped; padded, each took under 2 s and 0.6 GB.)
  Crops are perspective-warped from the full-res array, padded left/right (PAD_X); a crop that is
  1.5x taller than wide is turned 90 degrees (as RapidOCR does). Each crop is recognised upright and
  turned 180 degrees and the higher recogniser score wins: rapidocr's 0/180 classifier (use_cls)
  flipped a clean, upright synthetic VIN render, so it is not used.
- Windows also come from 2..MAX_STITCH consecutive lines in reading order (VIN split across boxes);
  such windows touch every joined line and are marked stitched ("rot0/line2+3").
- Rank key 1 counts only for a window that starts or ends on a token boundary (line end, space or
  separator), and for a stitched window only if both ends do: a 17-char slice cut across words passes
  a check digit 1 time in 11 by chance. Ties after the four documented keys go to token-aligned
  windows, then to a verified check digit, then to fewer corrections.
- A plausible candidate (status valid / valid_unverified) is *credible* if it has key 1, or key 2 on a
  window anchored at a token boundary, or if it is exactly one whole token of one line ("GAZ LOT
  12345678901" is 17 characters of whole words, but three tokens). 'read' = exactly one
  credible candidate; 'ambiguous' = two or more; 'could_not_read' = none (a lone window cut across
  words, e.g. "...UZBEKISTAN" + "MAX WEIGHT 1690", is not a reading). Credible candidates are
  returned first; could_not_read still lists the other plausible and the 'invalid' windows so a
  person can see what was read.
- Lines scoring below the engine's ``text_score`` (rapidocr default 0.5, the filter its own one-call
  pipeline applies) stay in ``raw_lines`` but do not feed candidates.
- Rotations stop early once a strong candidate is 'read' (``early_exit``); at most MAX_LINES lines
  (the widest) are recognised per rotation.
- VinOcrResult has no reason field: a could_not_read (or a failed rotation) adds an entry to
  ``raw_lines`` whose text starts with REASON_PREFIX and whose score is 0.0.
- ``_vin_from_lines`` applies the same windows/validation/ranking/status to text lines from another
  OCR engine (e.g. the EasyOCR comparison), so engines are compared with identical post-processing.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import unicodedata
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Literal

import cv2
import numpy as np

from ..timing import StepTimer
from .validate import VinValidation, _is_separator, _wmi_set, normalize, validate

log = logging.getLogger("avtopassport.vin.ocr")

OcrStatus = Literal["read", "ambiguous", "could_not_read"]

ROTATIONS: tuple[int, ...] = (0, 90, 270)   # degrees clockwise
REASON_PREFIX = "[reason] "
DET_MAX_SIDE = 2000      # rapidocr 3.10 default Global.max_side_len (used if the engine does not say)
DET_MAX_RATIO = 8        # rapidocr 3.10 Global.width_height_ratio: it pads wider inputs top/bottom
DET_MIN_SIDE = 30        # rapidocr 3.10 Global.min_height / min_side_len
TEXT_SCORE = 0.5         # rapidocr 3.10 default Global.text_score (used if the engine does not say)
REC_HEIGHT = 48          # rapidocr 3.10 recogniser input height (Rec.rec_img_shape)
MIN_CROP_HEIGHT = 32     # crops lower than this are upscaled to REC_HEIGHT (module docstring rule)
PAD_X = 0.20             # crop padding left/right, fraction of line height (uncalibrated default)
PAD_Y = 0.0              # no padding above/below: on synthetic renders it made the recogniser drop
                         # characters (the 48 px input gets narrower); DB boxes already have a margin
MAX_STITCH = 3           # consecutive lines joined when looking for a split VIN (uncalibrated default)
MAX_CANDIDATES = 10      # candidates returned in a result (uncalibrated default)
MAX_LINES = 60           # widest text lines recognised per rotation, bounds latency (uncalibrated default)

_FLAGS = ("use_det", "use_cls", "use_rec")
_ENGINE_LOCK = threading.Lock()   # rapidocr stores the stage flags on the engine: one call at a time
_PLAUSIBLE = ("valid", "valid_unverified")
_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


@dataclass(frozen=True)
class OcrCandidate:
    text: str                 # normalised 17-char string
    score: float              # mean OCR confidence of the contributing text
    source: str               # e.g. "rot0/line3" or "rot90/line1+2"
    validation: VinValidation


@dataclass
class VinOcrResult:
    status: OcrStatus
    best: OcrCandidate | None
    candidates: list[OcrCandidate] = field(default_factory=list)
    raw_lines: list[tuple[str, float]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    engine: str = "rapidocr"


@dataclass(frozen=True)
class _Line:
    no: int        # 1-based reading order within one rotation
    text: str
    score: float


@dataclass(frozen=True)
class _Ranked:
    cand: OcrCandidate
    key: tuple[Any, ...]
    strong: bool     # anchored window with a verified mandatory check digit or a known WMI
    exact: bool      # the window is exactly one whole token of one line
    anchored: bool   # at least one window end on a token boundary

    @property
    def credible(self) -> bool:
        return self.strong or self.exact


def make_engine(model_root: Path | None = None, *, threads: int | None = None) -> Any:
    """Create a RapidOCR engine (CPU, onnxruntime).

    model_root: folder with the ONNX models; None uses the models bundled in the rapidocr wheel
    (rapidocr downloads a missing default model into model_root on first use).
    threads: onnxruntime intra-op threads (None = onnxruntime default).
    """
    from rapidocr import EngineType, RapidOCR

    params: dict[str, Any] = {
        "Global.log_level": "error",
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Cls.engine_type": EngineType.ONNXRUNTIME,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "EngineConfig.onnxruntime.use_cuda": False,
    }
    if model_root is not None:
        params["Global.model_root_dir"] = str(model_root)
    if threads is not None:
        params["EngineConfig.onnxruntime.intra_op_num_threads"] = int(threads)
    return RapidOCR(params=params)


@lru_cache(maxsize=1)
def _default_engine() -> Any:
    return make_engine()


def _default_known_wmis() -> frozenset[str]:
    try:
        from .decode import known_wmis

        return known_wmis()
    except Exception as e:  # missing/broken table: rank without WMI knowledge
        log.warning("WMI table unavailable for OCR ranking: %s", e)
        return frozenset()


def _as_bgr(image: Any) -> tuple[np.ndarray | None, str | None]:
    """Contract check: uint8 HxWx3 BGR (gray and BGRA are converted). Returns (image, problem)."""
    if not isinstance(image, np.ndarray):
        return None, f"expected a numpy array, got {type(image).__name__}"
    if image.dtype != np.uint8:
        return None, f"expected dtype uint8, got {image.dtype}"
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        return None, f"expected an HxWx3 BGR image, got shape {image.shape}"
    if min(image.shape[:2]) < 8:
        return None, f"image too small ({image.shape[1]}x{image.shape[0]} px)"
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(image), None


def _run(engine: Any, img: np.ndarray, *, use_det: bool, use_cls: bool, use_rec: bool) -> Any:
    """Call the engine with explicit stage flags and restore the flags it had before (serialised)."""
    with _ENGINE_LOCK:
        saved = {k: getattr(engine, k) for k in _FLAGS if hasattr(engine, k)}
        try:
            return engine(img, use_det=use_det, use_cls=use_cls, use_rec=use_rec)
        finally:
            for k, v in saved.items():
                setattr(engine, k, v)


def _det_pad(h: int, w: int) -> int:
    """Left/right padding (px) for a tall, thin detection input, as RapidOCR pads a wide one top/bottom."""
    if h <= w or (h <= DET_MAX_RATIO * w and w > DET_MIN_SIDE):
        return 0
    return max(0, math.ceil((2 * max(h / DET_MAX_RATIO, DET_MIN_SIDE) - w) / 2))


def _detect(engine: Any, img: np.ndarray) -> np.ndarray:
    """Text-line quads (N, 4, 2), detected at RapidOCR's default size, in ``img`` pixel coordinates."""
    h, w = img.shape[:2]
    max_side = int(getattr(engine, "max_side_len", None) or DET_MAX_SIDE)
    scale = min(1.0, max_side / max(h, w))
    small = img if scale == 1.0 else cv2.resize(
        img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    pad = _det_pad(sh, sw)
    det_in = small if not pad else cv2.copyMakeBorder(small, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    boxes = getattr(_run(engine, det_in, use_det=True, use_cls=False, use_rec=False), "boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4, 2), np.float32)
    quads = np.array(boxes, dtype=np.float32).reshape(-1, 4, 2)    # a copy: the engine's output stays intact
    if pad:
        quads[:, :, 0] = np.clip(quads[:, :, 0] - pad, 0, sw)
    return quads * np.array([w / sw, h / sh], np.float32)


def _quad_size(q: np.ndarray) -> tuple[float, float]:
    """(width along the text, height across it) of a quad ordered TL, TR, BR, BL."""
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return float(w), float(h)


def _reading_order(quads: np.ndarray) -> list[int]:
    """Indices of the MAX_LINES widest lines, sorted top-to-bottom by rows (centres within half a line
    height), then left-to-right."""
    if len(quads) > MAX_LINES:
        widths = np.array([_quad_size(q)[0] for q in quads])
        keep = np.sort(np.argsort(-widths, kind="stable")[:MAX_LINES])
        return [int(keep[i]) for i in _reading_order(quads[keep])]
    cy = quads[:, :, 1].mean(axis=1)
    x0 = quads[:, :, 0].min(axis=1)
    heights = [_quad_size(q)[1] for q in quads]
    rows: list[list[int]] = []
    row_cy = row_h = 0.0
    for i in np.argsort(cy, kind="stable"):
        if rows and abs(cy[i] - row_cy) <= 0.5 * max(heights[i], row_h):
            rows[-1].append(int(i))
        else:
            rows.append([int(i)])
            row_cy, row_h = float(cy[i]), heights[i]
    return [i for row in rows for i in sorted(row, key=lambda j: x0[j])]


def _crop_line(img: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Padded, deskewed crop of one text line from the full-res image, ready for recognition."""
    w, h = _quad_size(quad)
    if w < 2 or h < 2:
        return None
    ux = (quad[1] - quad[0]) / max(float(np.linalg.norm(quad[1] - quad[0])), 1e-6)
    uy = (quad[3] - quad[0]) / max(float(np.linalg.norm(quad[3] - quad[0])), 1e-6)
    dx, dy = ux * PAD_X * h, uy * PAD_Y * h
    src = np.float32([quad[0] - dx - dy, quad[1] + dx - dy, quad[2] + dx + dy, quad[3] - dx + dy])
    out_w, out_h = max(2, round(w + 2 * PAD_X * h)), max(2, round(h + 2 * PAD_Y * h))
    dst = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
    crop = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (out_w, out_h),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if out_h >= 1.5 * out_w:
        crop = np.ascontiguousarray(np.rot90(crop))
    if crop.shape[0] < MIN_CROP_HEIGHT:
        f = REC_HEIGHT / crop.shape[0]
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * f)), REC_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return crop


def _recognise(engine: Any, crop: np.ndarray) -> tuple[str, float]:
    """(text, score) of the crop read upright and turned 180 degrees, whichever scores higher."""
    best = ("", 0.0)
    for img in (crop, cv2.rotate(crop, cv2.ROTATE_180)):
        res = _run(engine, img, use_det=False, use_cls=False, use_rec=True)
        txts = getattr(res, "txts", None) or ()
        scores = getattr(res, "scores", None)
        if txts and scores is not None and len(scores) and float(scores[0]) > best[1]:
            best = (str(txts[0]).strip(), float(scores[0]))
    return best


def _clean(text: str) -> tuple[str, frozenset[int]]:
    """Uppercase NFKC text without whitespace/separators, plus the indices that are token boundaries."""
    out: list[str] = []
    bounds = {0}
    for c in unicodedata.normalize("NFKC", text).upper():
        if _is_separator(c):
            bounds.add(len(out))
        elif c.isalnum():
            out.append(c)
        else:  # junk such as '|': kept (windows containing it are unreadable) but it splits tokens
            bounds.add(len(out))
            out.append(c)
            bounds.add(len(out))
    bounds.add(len(out))
    return "".join(out), frozenset(bounds)


def _windows(lines: list[_Line], prefix: str) -> Iterator[tuple[str, float, str, bool, int, bool]]:
    """(window, score, source, stitched, aligned_ends, single_token) for every 17-char window of a line
    or of 2..MAX_STITCH consecutive lines (a stitched window touches every joined line)."""
    cleaned = [_clean(line.text) for line in lines]
    for i in range(len(lines)):
        for k in range(1, MAX_STITCH + 1):
            parts = cleaned[i:i + k]
            if len(parts) < k:
                break
            text = "".join(p for p, _ in parts)
            if len(text) < 17:
                continue
            offsets = [0]
            for p, _ in parts:
                offsets.append(offsets[-1] + len(p))
            bounds = sorted({o + b for o, (_, bs) in zip(offsets, parts) for b in bs})
            bound_set = set(bounds)
            char_scores = np.repeat([line.score for line in lines[i:i + k]], [len(p) for p, _ in parts])
            source = f"{prefix}/line" + "+".join(str(line.no) for line in lines[i:i + k])
            for s in range(len(text) - 16):
                e = s + 17
                if k > 1 and not (s < offsets[1] and e > offsets[k - 1]):
                    continue
                aligned = (s in bound_set) + (e in bound_set)
                single_token = aligned == 2 and bisect_right(bounds, s) >= bisect_left(bounds, e)
                yield text[s:e], float(char_scores[s:e].mean()), source, k > 1, aligned, single_token


def _rank(cand: OcrCandidate, stitched: bool, aligned: int, single_token: bool,
          known: frozenset[str]) -> _Ranked:
    v = cand.validation
    verified = v.status == "valid" and v.check_digit_required and bool(v.check_digit_ok)
    cd = verified and aligned >= (2 if stitched else 1)
    wmi = v.vin is not None and v.vin[:3] in known
    key = (cd, wmi, not stitched, round(cand.score, 4), aligned, verified, -len(v.corrections))
    return _Ranked(cand, key, strong=cd or (wmi and aligned > 0), exact=single_token and not stitched,
                   anchored=aligned > 0)


def _add_lines(pool: dict[str, _Ranked], lines: list[_Line], prefix: str, known: frozenset[str]) -> None:
    """Validate every window of ``lines`` and keep the best-ranked instance of each VIN in ``pool``."""
    for window, score, source, stitched, aligned, single_token in _windows(lines, prefix):
        val = validate(window, known_wmis=known)
        if val.vin is None:
            continue
        cand = OcrCandidate(normalize(window), score, source, val)
        ranked = _rank(cand, stitched, aligned, single_token, known)
        old = pool.get(val.vin)
        if old is None or ranked.key > old.key:
            pool[val.vin] = ranked


def _near_miss(r: _Ranked) -> bool:
    """An 'invalid' window worth showing: anchored, digits in 14-17, failing only check digit/year code."""
    vin = r.cand.validation.vin
    return r.cand.validation.status == "invalid" and r.anchored and vin is not None and vin[13:].isdigit()


def _decide(pool: dict[str, _Ranked]) -> tuple[OcrStatus, list[_Ranked]]:
    """Status and the candidates to return (credible first, each group in rank order)."""
    ranked = sorted(pool.values(), key=lambda r: r.key, reverse=True)
    plausible = [r for r in ranked if r.cand.validation.status in _PLAUSIBLE]
    credible = [r for r in plausible if r.credible]
    others = [r for r in plausible if not r.credible]
    if not credible:
        return "could_not_read", others + [r for r in ranked if _near_miss(r)]
    return ("read" if len(credible) == 1 else "ambiguous"), credible + others


def _fail(reason: str, raw_lines: list[tuple[str, float]], timings: dict[str, float], t0: float,
          engine_name: str = "rapidocr") -> VinOcrResult:
    timings["total"] = (time.perf_counter() - t0) * 1000.0
    lines = raw_lines + [(REASON_PREFIX + reason, 0.0)]
    return VinOcrResult("could_not_read", None, [], lines, timings, engine_name)


def _finish(status: OcrStatus, chosen: list[_Ranked], raw_lines: list[tuple[str, float]],
            timings: dict[str, float], t0: float, no_text_reason: str,
            engine_name: str = "rapidocr") -> VinOcrResult:
    """Assemble the result; a could_not_read keeps its non-credible windows and near misses as candidates."""
    candidates = [r.cand for r in chosen[:MAX_CANDIDATES]]
    if status == "could_not_read":
        reason = "no credible VIN reading; closest windows are in candidates" if candidates else no_text_reason
        result = _fail(reason, raw_lines, timings, t0, engine_name)
        result.candidates = candidates
        return result
    timings["total"] = (time.perf_counter() - t0) * 1000.0
    return VinOcrResult(status, candidates[0], candidates, raw_lines, timings, engine_name)


def _vin_from_lines(lines: list[tuple[str, float]], *, known_wmis: frozenset[str] | None = None,
                    prefix: str = "ext", engine: str = "external") -> VinOcrResult:
    """read_vin's windows, validation, ranking and status for (text, score) lines from another OCR
    engine, given in reading order (e.g. EasyOCR ``readtext`` output sorted top-to-bottom)."""
    t0 = time.perf_counter()
    known = _default_known_wmis() if known_wmis is None else _wmi_set(known_wmis)
    items = [_Line(no, str(text).strip(), float(score)) for no, (text, score) in enumerate(lines, start=1)
             if str(text).strip()]
    pool: dict[str, _Ranked] = {}
    _add_lines(pool, items, prefix, known)
    status, chosen = _decide(pool)
    reason = "no 17-character VIN candidate in the text" if items else "no text"
    timings = {"validate": (time.perf_counter() - t0) * 1000.0}
    return _finish(status, chosen, [(line.text, line.score) for line in items], timings, t0, reason, engine)


def read_vin(image: np.ndarray, *, engine: Any | None = None, known_wmis: frozenset[str] | None = None,
             rotations: tuple[int, ...] = ROTATIONS, early_exit: bool = True) -> VinOcrResult:
    """Read a VIN from a BGR uint8 image. Never raises on OCR failure: returns 'could_not_read'.

    known_wmis: None loads data/reference/wmi_table.csv; pass frozenset() to rank without it.
    rotations: degrees clockwise to try, in order; early_exit stops once a strong candidate is 'read'.
    """
    t0 = time.perf_counter()
    timer = StepTimer()
    raw_lines: list[tuple[str, float]] = []

    def timings() -> dict[str, float]:
        return {name: s["total_ms"] for name, s in timer.summary().items()}

    img, problem = _as_bgr(image)
    if img is None:
        return _fail(problem or "bad image", raw_lines, timings(), t0)
    known = _default_known_wmis() if known_wmis is None else _wmi_set(known_wmis)
    if engine is None:
        try:
            with timer.step("engine_init"):
                engine = _default_engine()
        except Exception as e:
            return _fail(f"OCR engine unavailable ({type(e).__name__}: {e})", raw_lines, timings(), t0)

    pool: dict[str, _Ranked] = {}
    status: OcrStatus = "could_not_read"
    chosen: list[_Ranked] = []
    any_text = False
    completed = 0
    for rot in rotations:
        rot %= 360
        if rot not in (0, *_ROTATE):
            raw_lines.append((f"{REASON_PREFIX}rotation {rot} not supported (multiples of 90 only)", 0.0))
            continue
        try:
            with timer.step("rotate"):
                rimg = img if rot == 0 else cv2.rotate(img, _ROTATE[rot])
            with timer.step("det"):
                quads = _detect(engine, rimg)
            lines: list[_Line] = []
            failed = 0
            min_score = float(getattr(engine, "text_score", None) or TEXT_SCORE)
            with timer.step("rec"):
                for no, idx in enumerate(_reading_order(quads), start=1):
                    crop = _crop_line(rimg, quads[idx])
                    if crop is None:
                        continue
                    try:
                        text, score = _recognise(engine, crop)
                    except Exception:
                        failed += 1
                        continue
                    if text:
                        any_text = True
                        raw_lines.append((text, score))
                        if score >= min_score:
                            lines.append(_Line(no, text, score))
            if failed:
                raw_lines.append((f"{REASON_PREFIX}rot{rot}: recognition failed on {failed} line(s)", 0.0))
            with timer.step("validate"):
                _add_lines(pool, lines, f"rot{rot}", known)
                status, chosen = _decide(pool)
            completed += 1
        except Exception as e:  # one failed rotation must not crash the read
            raw_lines.append((f"{REASON_PREFIX}rot{rot}: OCR failed ({type(e).__name__}: {e})", 0.0))
            continue
        if early_exit and status == "read" and chosen[0].strong:
            break

    if any_text:
        reason = "no 17-character VIN candidate in the recognised text"
    elif completed:
        reason = "no text detected"
    else:
        reason = "OCR failed on every rotation"
    return _finish(status, chosen, raw_lines, timings(), t0, reason)
