"""Inspection orchestrator: walkaround video (+ VIN photo or typed VIN) -> InspectionResult.

Steps (each timed with StepTimer; progress(step, fraction) is called between steps and per frame):
 1 extract_frames (full resolution)  2 filter_frames  3 anonymize each kept frame (fail closed: a frame
 whose anonymisation fails is dropped)  4 save the anonymised working copy (long side work_side) as
 out_dir/frames/f_<index:05d>.jpg  5 damage + parts segmentation on that working copy  6 mapping.assign
 7 dedupe.merge  8 severity.rate_all  9 cost.estimator.estimate  10 VIN: a typed VIN that validates wins;
 otherwise OCR on the VIN photo (loaded in memory only, never saved) -> validate -> decode  11 hash_frames
 over the saved frame files  12 per-frame body-panel masks (out_dir/masks/f_<index:05d>.png, uint8 0/255,
 for the paint try-on) and per-frame damage boxes (extra["pins2d"]) for the spin viewer.

Only anonymised pixels are written; raw frames are released as soon as they are anonymised. One failed
frame (anonymise / save / segment / map) is skipped and counted, never fatal. Downstream modules
(mapping, dedupe, severity, cost, report, tryon) are imported lazily via importlib so this module imports
before they exist and tests can replace them in sys.modules.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..io import load_image, resize_max_side, save_jpeg
from ..paths import REPO_ROOT
from ..schema import DamageObservation, InspectionResult, Segment
from ..timing import StepTimer
from .frames import extract_frames
from .quality import filter_frames

log = logging.getLogger("avtopassport.run")

Progress = Callable[[str, float], None]
COCO_FILE = "models/coco/yolo26n.pt"
# fallback body-mask exclusions when tryon.recolor is unavailable (glass, wheels, lights, plate, grille)
_NON_BODY = ("glass", "window", "windshield", "wheel", "tire", "tyre", "light", "lamp", "plate", "grille")
_PLAUSIBLE = ("valid", "valid_unverified")


@dataclass
class InspectionConfig:
    device: str = "cpu"
    fps: float = 2.5
    max_frames: int = 72
    work_side: int = 1280
    spin_frames: int = 36
    damage_key: str = "damage_mitbersh"
    parts_key: str = "parts_mitbersh"
    use_vpic: bool = True
    vin_side_for_ocr: int | None = None   # None = OCR on the full-resolution VIN photo


def _mod(name: str) -> Any:
    return importlib.import_module(f"avtopassport.{name}")


def mask_vin(vin: str | None) -> str | None:
    """First 11 characters + '******' (serial hidden), or None."""
    return None if not vin else vin[:11] + "******"


def spin_indices(n: int, k: int) -> list[int]:
    """Up to k positions in range(n), evenly spaced, first and last included."""
    if n <= 0 or k <= 0:
        return []
    if n <= k:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


def fallback_body_mask(parts: list[Segment], shape: tuple[int, int]) -> np.ndarray:
    """Union of painted-part masks minus glass / wheel / light / plate / grille masks (bool)."""
    h, w = shape
    body, excluded = np.zeros((h, w), bool), np.zeros((h, w), bool)
    for p in parts:
        if p.mask is None or p.mask.shape != (h, w):
            continue
        if any(k in p.cls.lower() for k in _NON_BODY):
            excluded |= p.mask
        else:
            body |= p.mask
    return body & ~excluded


def body_mask_u8(parts: list[Segment], shape: tuple[int, int]) -> np.ndarray:
    """uint8 0/255 body-panel mask; tryon.recolor.body_mask when importable, else fallback_body_mask."""
    try:
        m = np.asarray(_mod("tryon.recolor").body_mask(parts, shape))
        if m.shape[:2] != tuple(shape):
            raise ValueError(f"body_mask returned shape {m.shape}, expected {shape}")
    except Exception as exc:  # module missing or incompatible: same rule, local implementation
        log.debug("tryon.recolor.body_mask unavailable (%s); using the fallback", exc)
        m = fallback_body_mask(parts, shape)
    return np.where(m > 0, 255, 0).astype(np.uint8)


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class Inspector:
    """Loads every model once; ``run`` is serialised with a lock (ultralytics models are not thread-safe)."""

    def __init__(
        self,
        config: InspectionConfig | None = None,
        *,
        anonymizer: Any | None = None,
        damage: Any | None = None,
        parts: Any | None = None,
        ocr_engine: Any | None = None,
        known_wmis: frozenset[str] | None = None,
    ) -> None:
        """Components passed in are used as is (tests, custom setups); the rest is loaded here.
        Raises PrivacyUnavailable / FileNotFoundError when a required model cannot be loaded."""
        self.config = cfg = config or InspectionConfig()
        self._lock = threading.Lock()
        self.load_timer = StepTimer()
        from .segment import load_segmenter, load_yolo, manifest_sha256

        with self.load_timer.step("load.damage"):
            self.damage = damage if damage is not None else load_segmenter(cfg.damage_key, device=cfg.device)
        with self.load_timer.step("load.parts"):
            self.parts = parts if parts is not None else load_segmenter(cfg.parts_key, device=cfg.device)
        if anonymizer is None:
            from .privacy import Anonymizer, make_parts_plate_detector, make_person_head_detector

            with self.load_timer.step("load.privacy"):
                # separate instances: the privacy detectors call predict with their own classes/imgsz
                privacy_parts = load_yolo(self.parts.model_path)
                coco = load_yolo(REPO_ROOT / COCO_FILE)
                extra = [make_parts_plate_detector(privacy_parts, device=cfg.device),
                         make_person_head_detector(coco, device=cfg.device)]
                anonymizer = Anonymizer(device=cfg.device, extra_detectors=extra)
        self.anonymizer = anonymizer
        if ocr_engine is None:
            with self.load_timer.step("load.ocr"):
                ocr_engine = _mod("vin.ocr").make_engine()
        self.ocr_engine = ocr_engine
        if known_wmis is None:
            try:
                known_wmis = _mod("vin.decode").known_wmis()
            except Exception as exc:
                log.warning("WMI table unavailable: %s", exc)
                known_wmis = frozenset()
        self.known_wmis = known_wmis
        self.model_sha256 = manifest_sha256([cfg.damage_key, cfg.parts_key, "coco_yolo26n", "plate_n_pt",
                                             "plate_n_onnx"])

    # ------------------------------------------------------------------------------------------------ run

    def run(
        self,
        video_path: Path,
        *,
        out_dir: Path,
        vin_image_path: Path | None = None,
        vin_text: str | None = None,
        progress: Progress | None = None,
    ) -> InspectionResult:
        """Full inspection; see the module docstring. Raises only if the video cannot be opened at all or a
        whole-run step (dedupe / severity / hashing) fails."""
        with self._lock:
            return self._run(Path(video_path), Path(out_dir), vin_image_path, vin_text, progress)

    def _run(self, video_path: Path, out_dir: Path, vin_image_path: Path | None, vin_text: str | None,
             progress: Progress | None) -> InspectionResult:
        cfg = self.config
        timer = StepTimer()
        notes: list[str] = []

        def tell(step: str, frac: float) -> None:
            if progress is not None:
                try:
                    progress(step, float(min(1.0, max(0.0, frac))))
                except Exception as exc:  # a broken progress callback must not stop the inspection
                    log.warning("progress callback failed: %s", exc)

        frames_dir, masks_dir = out_dir / "frames", out_dir / "masks"
        frames_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        with timer.step("total"):
            # 1-2 frames + quality
            tell("extract_frames", 0.0)
            with timer.step("extract_frames"):
                frames, info, frame_errors = extract_frames(video_path, target_fps=cfg.fps,
                                                            max_frames=cfg.max_frames)
            notes += [f"frames: {e}" for e in frame_errors]
            tell("filter_frames", 0.08)
            with timer.step("filter_frames"):
                kept, rejected, _ = filter_frames(frames)
            quality = {"sampled": len(frames), "kept": len(kept), "rejected": len(rejected),
                       "reasons": dict(Counter(q.reason for _, q in rejected if q.reason))}
            del frames, rejected  # rejected raw pixels are never kept


            # 3-4 anonymise each kept frame, save the anonymised working copy; raw frames released one by one
            privacy = {"frames_in": len(kept), "frames_ok": 0, "frames_dropped": 0, "save_failed": 0,
                       "plates": 0, "faces": 0}
            saved: list[tuple[str, Path, np.ndarray]] = []   # (frame id, saved file, anonymised working copy)
            n_kept = len(kept)
            for i in range(n_kept):
                fr = kept[i]
                kept[i] = None  # type: ignore[call-overload]  # release the raw frame with this loop
                index = fr.index
                tell("anonymize", 0.1 + 0.4 * i / max(1, n_kept))
                try:
                    with timer.step("anonymize"):
                        blurred, report = self.anonymizer.anonymize(fr.image)
                except Exception as exc:  # anonymize() must not raise; if it does, fail closed
                    blurred, report = None, None
                    notes.append(f"privacy f{index}: {_err(exc)}")
                del fr
                if blurred is None or report is None or not report.ok:
                    privacy["frames_dropped"] += 1
                    continue
                privacy["frames_ok"] += 1
                privacy["plates"] += int(report.n_plates)
                privacy["faces"] += int(report.n_faces)
                work = resize_max_side(blurred, cfg.work_side)
                del blurred
                fid = f"f_{index:05d}"
                try:
                    with timer.step("save_frame"):
                        path = save_jpeg(work, frames_dir / f"{fid}.jpg", max_side=cfg.work_side)
                except Exception as exc:
                    privacy["save_failed"] += 1
                    notes.append(f"save {fid}: {_err(exc)}")
                    continue
                saved.append((fid, path, work))
            del kept
            if not saved:
                notes.append("no frame passed quality + privacy: no damage analysis")

            # 5-6 + 12: segmentation, mapping, body masks and 2D pins per saved frame
            observations: list[DamageObservation] = []
            pins2d: list[dict[str, Any]] = []
            seg_failed = map_failed = mask_failed = 0
            mapping = _mod("pipeline.mapping")
            for i, (fid, path, work) in enumerate(saved):
                tell("segment", 0.5 + 0.35 * i / max(1, len(saved)))
                rel = path.relative_to(out_dir).as_posix()
                try:
                    with timer.step("segment.damage"):
                        damages = self.damage.segment(work)
                    with timer.step("segment.parts"):
                        parts = self.parts.segment(work)
                except Exception as exc:
                    seg_failed += 1
                    notes.append(f"segment {fid}: {_err(exc)}")
                    continue
                try:
                    with timer.step("mapping.assign"):
                        obs = mapping.assign(fid, damages, parts)
                    observations += obs
                    pins2d += [{"frame": rel, "box": [round(float(v), 1) for v in o.box], "damage": o.damage,
                                "part": o.part, "score": round(float(o.score), 4)} for o in obs]
                except Exception as exc:
                    map_failed += 1
                    notes.append(f"mapping {fid}: {_err(exc)}")
                try:
                    with timer.step("body_mask"):
                        m = body_mask_u8(parts, work.shape[:2])
                        if not cv2.imwrite(str(masks_dir / f"{fid}.png"), m):
                            raise OSError("cv2.imwrite returned False")
                except Exception as exc:
                    mask_failed += 1
                    notes.append(f"body mask {fid}: {_err(exc)}")
                del damages, parts
            frame_size = [int(saved[0][2].shape[1]), int(saved[0][2].shape[0])] if saved else None
            rel_frames = [p.relative_to(out_dir).as_posix() for _, p, _ in saved]
            saved_paths = [p for _, p, _ in saved]
            del saved

            # 7-9 findings, severity, cost
            tell("dedupe", 0.86)
            with timer.step("dedupe.merge"):
                findings = _mod("pipeline.dedupe").merge(observations)
            with timer.step("severity.rate_all"):
                findings = _mod("pipeline.severity").rate_all(findings)
            tell("cost", 0.88)
            costs: list[Any] = []
            total: int | None = None
            try:
                with timer.step("cost.estimate"):
                    costs, total = _mod("cost.estimator").estimate(findings)
            except Exception as exc:
                notes.append(f"cost estimate unavailable: {_err(exc)}")

            # 10 VIN
            tell("vin", 0.9)
            with timer.step("vin"):
                vin, vin_status, vin_extra = self._vin(vin_image_path, vin_text, notes)

            # 11 hashes over the saved (anonymised) frame files
            tell("hash_frames", 0.97)
            with timer.step("hash_frames"):
                frames_sha256, frame_hashes = _mod("passport.report").hash_frames(saved_paths)

        summary = timer.summary()
        extra: dict[str, Any] = {
            "frames": rel_frames,
            "spin": spin_indices(len(rel_frames), cfg.spin_frames),
            "pins2d": pins2d,
            "frame_size": frame_size,
            "masks": [f"masks/{Path(r).stem}.png" for r in rel_frames],
            "frame_hashes": frame_hashes,
            "quality": quality,
            "privacy": privacy,
            "analysis": {"frames_segmented": len(rel_frames) - seg_failed, "segment_failed": seg_failed,
                         "mapping_failed": map_failed, "mask_failed": mask_failed,
                         "observations": len(observations)},
            "video": {"width": info.width, "height": info.height, "fps": round(info.fps, 3),
                      "duration_s": round(info.duration_s, 2), "backend": info.backend, "hdr": info.hdr},
            "config": dataclasses.asdict(cfg),
            "models": dict(self.model_sha256),
            "timings": summary,
            **vin_extra,
        }
        tell("done", 1.0)
        return InspectionResult(
            vin=vin,
            vin_status=vin_status,
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            frames_sha256=frames_sha256,
            findings=findings,
            costs=costs,
            total_from_uzs=total,
            timings_ms={k: round(v["total_ms"], 1) for k, v in summary.items()},
            notes=notes,
            extra=extra,
        )

    # ------------------------------------------------------------------------------------------------ VIN

    def _vin(self, image_path: Path | None, vin_text: str | None,
             notes: list[str]) -> tuple[str | None, str, dict[str, Any]]:
        """(vin, vin_status, extra fields). A typed VIN that validates wins; otherwise OCR on the photo.
        The photo is decoded in memory only and never saved."""
        validate = _mod("vin.validate").validate
        vin: str | None = None
        status = "could_not_read"
        extra: dict[str, Any] = {"vin_source": None, "vin_validation": None, "vin_ocr": None}
        if vin_text and vin_text.strip():
            v = validate(vin_text, known_wmis=self.known_wmis)
            extra["vin_validation"] = {"status": v.status, "check_digit_ok": v.check_digit_ok,
                                       "corrections": list(v.corrections)}
            if v.status in _PLAUSIBLE and v.vin:
                vin, status, extra["vin_source"] = v.vin, "manual", "manual"
            else:
                notes.append(f"typed VIN rejected by validation ({v.status})")
        if vin is None and image_path is not None:
            try:
                img = load_image(image_path, max_side=self.config.vin_side_for_ocr)
                try:
                    res = _mod("vin.ocr").read_vin(img, engine=self.ocr_engine, known_wmis=self.known_wmis)
                finally:
                    del img
                status = res.status
                extra["vin_ocr"] = {"status": res.status, "n_candidates": len(res.candidates),
                                    "candidates_masked": [mask_vin(c.text) for c in res.candidates[:5]],
                                    "timings_ms": dict(res.timings_ms)}
                if res.status == "read" and res.best is not None:
                    v = validate(res.best.validation.vin or res.best.text, known_wmis=self.known_wmis)
                    extra["vin_validation"] = {"status": v.status, "check_digit_ok": v.check_digit_ok,
                                               "corrections": list(v.corrections)}
                    if v.status in _PLAUSIBLE and v.vin:
                        vin, extra["vin_source"] = v.vin, "ocr"
                    else:
                        status = "could_not_read"
                        notes.append(f"OCR reading failed re-validation ({v.status})")
            except Exception as exc:  # unreadable photo / OCR crash: structured failure, never a guess
                status = "could_not_read"
                notes.append(f"VIN photo: {_err(exc)}")
        decoded = None
        if vin is not None:
            try:
                decoded = dataclasses.asdict(_mod("vin.decode").decode(vin, use_vpic=self.config.use_vpic))
            except Exception as exc:
                notes.append(f"VIN decode failed: {_err(exc)}")
        extra["vin_decode"] = decoded
        extra["vin_masked"] = mask_vin(vin)
        return vin, status, extra
