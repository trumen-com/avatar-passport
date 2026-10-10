"""Blur licence plates and faces in data/raw/ media and write the safe copies to data/processed/. FAIL CLOSED.

  data/raw/video/*    -> frames at 1 fps (extract_frames, full resolution) -> Anonymizer ->
                         data/processed/frames/<video-stem>_<t>s.jpg   (long side <= 1280, no metadata)
  data/raw/images/*   -> load_image (full resolution, EXIF-rotated) -> Anonymizer ->
                         data/processed/images/<stem>.jpg               (long side <= 1280, no metadata)
  data/raw/vin/*      -> SKIPPED: VIN photos are never copied to data/processed/ (OCR input only)
  data/raw/originals/ -> SKIPPED: camera originals (HEIC / HDR MOV) are only used to test decoding

Detectors: plate YOLO (two passes) + YuNet faces (multi-scale) + the parts model's 'license-plate'
class (models/parts/mitbersh_parts_segmentation.pt). If any of them cannot be loaded, nothing is
written and the exit code is 2. An image or frame whose detection fails is dropped: counted in the
summary and the manifest, never written. data/processed/manifest.json records, per output: the
source file and its sha256, the output sha256, size in bytes, plate/face counts and per-pass counts.

Usage (from the repo root):
    YOLO_AUTOINSTALL=false .venv/bin/python scripts/blur_raw.py --dry-run
    YOLO_AUTOINSTALL=false .venv/bin/python scripts/blur_raw.py [--device auto|cpu|cuda] [--clean]

Exit codes: 0 every input written; 1 finished, but something was dropped or could not be read;
2 a privacy detector is unavailable (nothing written).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
try:
    import avtopassport  # noqa: F401
except ImportError:  # not installed: use the source tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from avtopassport.io import load_image, save_jpeg, sha256_file  # noqa: E402
from avtopassport.paths import MODELS, PROCESSED, RAW, REPO_ROOT  # noqa: E402
from avtopassport.pipeline.frames import extract_frames, probe_video  # noqa: E402
from avtopassport.pipeline.privacy import (  # noqa: E402
    Anonymizer,
    PrivacyReport,
    PrivacyUnavailable,
    make_parts_plate_detector,
    make_person_head_detector,
)

PARTS_MODEL = MODELS / "parts" / "mitbersh_parts_segmentation.pt"
COCO_MODEL = MODELS / "coco" / "yolo26n.pt"  # person-head backstop for faces YuNet misses
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
FRAME_FPS = 1.0
MAX_SIDE = 1280
EXIT_OK, EXIT_PARTIAL, EXIT_PRIVACY_UNAVAILABLE = 0, 1, 2


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def list_files(folder: Path, exts: set[str] | None = None) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and (exts is None or p.suffix.lower() in exts))


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch missing or broken -> CPU (plates then use the ONNX model)
        return "cpu"


def build_anonymizer(device: str) -> Anonymizer:
    """Plate + face detectors plus the parts model's licence-plate class. Raises PrivacyUnavailable."""
    for what, path in (("parts", PARTS_MODEL), ("COCO", COCO_MODEL)):
        if not path.is_file():
            raise PrivacyUnavailable(f"{what} model missing: {path} (run scripts/download_models.py)")
    threads = cv2.getNumThreads()
    try:
        from ultralytics import YOLO

        cv2.setNumThreads(threads)  # the ultralytics import sets 0 (1 thread): slow YuNet tiles
        parts = YOLO(str(PARTS_MODEL))
        parts_plates = make_parts_plate_detector(parts, device=device)
        parts_plates(np.zeros((64, 64, 3), np.uint8))  # warm-up: a broken model fails here
        heads = make_person_head_detector(YOLO(str(COCO_MODEL)), device=device)
        heads(np.zeros((64, 64, 3), np.uint8))
    except PrivacyUnavailable:
        raise
    except Exception as exc:
        raise PrivacyUnavailable(f"parts/COCO model could not be loaded on {device}: {type(exc).__name__}: {exc}") from exc
    return Anonymizer(device=device, extra_detectors=[parts_plates, heads])


def unique_stem(p: Path, used: set[str]) -> str:
    """File stem for the outputs; adds the extension when two inputs share a stem (CAR1.jpg, CAR1.heic)."""
    stem = p.stem if p.stem not in used else f"{p.stem}_{p.suffix.lstrip('.').lower()}"
    used.add(stem)
    return stem


def dry_run(videos: list[Path], images: list[Path], vins: list[Path], out: Path) -> int:
    used: set[str] = set()
    for v in videos:
        try:
            info = probe_video(v)
            step = max(info.fps / FRAME_FPS, 1.0) if info.fps > 0 else 0.0
            n = int((info.frame_count - 1) // step) + 1 if step and info.frame_count > 0 else "?"
            print(f"video  {rel(v)}: {info.width}x{info.height} {info.fps:.2f} fps {info.frame_count} frames, "
                  f"rotation {info.rotation}, trc {info.color_trc}, hdr {info.hdr} -> ~{n} frames "
                  f"-> {rel(out / 'frames')}/{unique_stem(v, used)}_<t>s.jpg")
        except Exception as exc:
            print(f"video  {rel(v)}: cannot probe ({type(exc).__name__}: {exc})")
    used.clear()
    for p in images:
        print(f"image  {rel(p)} -> {rel(out / 'images' / (unique_stem(p, used) + '.jpg'))}")
    for p in vins:
        print(f"skip   {rel(p)} (VIN photos are never copied to processed/)")
    print(f"dry run: {len(videos)} videos, {len(images)} images, {len(vins)} VIN photos skipped; nothing written")
    return EXIT_OK


def record(out_path: Path, rep: PrivacyReport, **source: Any) -> dict[str, Any]:
    with Image.open(out_path) as im:
        width, height = im.size
    return {
        "output": rel(out_path), "sha256": sha256_file(out_path), "size": out_path.stat().st_size,
        "width": width, "height": height, **source,
        "n_plates": rep.n_plates, "n_faces": rep.n_faces, "passes": rep.passes,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be processed; load no models, write nothing")
    ap.add_argument("--device", default="auto", help="auto (cuda if available, else cpu) | cpu | cuda | cuda:N")
    ap.add_argument("--clean", action="store_true",
                    help="delete existing *.jpg in the output frames/ and images/ first")
    ap.add_argument("--raw", type=Path, default=RAW, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, default=PROCESSED, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    videos = list_files(args.raw / "video", VIDEO_EXT)
    images = list_files(args.raw / "images", IMAGE_EXT)
    vins = list_files(args.raw / "vin")
    if args.dry_run:
        return dry_run(videos, images, vins, args.out)

    device = resolve_device(args.device)
    t0 = time.perf_counter()
    try:
        anon = build_anonymizer(device)
    except (PrivacyUnavailable, ValueError) as exc:
        print(f"FAIL CLOSED: privacy detector unavailable ({exc}); nothing written", file=sys.stderr)
        return EXIT_PRIVACY_UNAVAILABLE
    print(f"detectors loaded on {device} in {time.perf_counter() - t0:.1f}s")

    frames_dir, images_dir = args.out / "frames", args.out / "images"
    if args.clean:
        for d in (frames_dir, images_dir):
            for old in list_files(d, {".jpg"}):
                old.unlink()
    outputs: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    written: set[Path] = set()
    used_stems: set[str] = set()

    for v in videos:
        stem = unique_stem(v, used_stems)
        src = {"source": rel(v), "source_sha256": sha256_file(v)}
        try:
            frames, info, errors = extract_frames(v, target_fps=FRAME_FPS)
        except Exception as exc:
            unreadable.append({**src, "error": f"{type(exc).__name__}: {exc}"})
            continue
        unreadable += [{**src, "error": e} for e in errors]
        for f in frames:
            meta = {**src, "frame_index": f.index, "t_s": round(f.t_s, 3)}
            safe, rep = anon.anonymize(f.image)
            if safe is None:
                dropped.append({**meta, "errors": rep.errors})
                continue
            path = save_jpeg(safe, frames_dir / f"{stem}_{f.t_s:07.2f}s.jpg", max_side=MAX_SIDE)
            written.add(path.resolve())
            outputs.append(record(path, rep, **meta))
        print(f"video {rel(v)}: {len(frames)} frames ({info.backend}, hdr={info.hdr}), "
              f"{sum(o['source'] == src['source'] for o in outputs)} written")

    used_stems.clear()
    for p in images:
        name = f"{unique_stem(p, used_stems)}.jpg"
        src = {"source": rel(p), "source_sha256": sha256_file(p)}
        try:
            img = load_image(p)
        except Exception as exc:
            unreadable.append({**src, "error": f"{type(exc).__name__}: {exc}"})
            continue
        safe, rep = anon.anonymize(img)
        if safe is None:
            dropped.append({**src, "errors": rep.errors})
            continue
        path = save_jpeg(safe, images_dir / name, max_side=MAX_SIDE)
        written.add(path.resolve())
        outputs.append(record(path, rep, **src))

    stale = [rel(p) for d in (frames_dir, images_dir) for p in list_files(d, {".jpg"}) if p.resolve() not in written]
    summary = {
        "videos": len(videos), "images": len(images), "vin_skipped": len(vins),
        "written": len(outputs), "dropped": len(dropped), "unreadable": len(unreadable), "stale": len(stale),
        "plates": sum(o["n_plates"] for o in outputs), "faces": sum(o["n_faces"] for o in outputs),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/blur_raw.py",
        "device": device,
        "settings": {"frame_fps": FRAME_FPS, "max_side": MAX_SIDE, "pad": anon.pad,
                     "plate_passes": [[anon.plate_imgsz, anon.plate_conf], [anon.plate_imgsz_hi, anon.plate_conf_hi]],
                     "face_score": anon.face_score},
        "detectors": {rel(m): sha256_file(m) for m in (anon.plate_model_path, anon.face_model_path, PARTS_MODEL)
                      if m is not None and m.is_file()},
        "skipped": [f"{rel(args.raw / 'vin')} (VIN photos are never copied to processed/)",
                    f"{rel(args.raw / 'originals')} (camera originals, decoding tests only)"],
        "summary": summary,
        "outputs": outputs,
        "dropped": dropped,
        "unreadable": unreadable,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(summary))
    if dropped:
        print(f"DROPPED {len(dropped)} image(s)/frame(s): privacy detection failed, nothing written for them")
    if stale:
        print(f"WARNING: {len(stale)} older file(s) in the output folders were not produced by this run "
              f"(re-run with --clean to remove them): {stale[:5]}")
    return EXIT_PARTIAL if dropped or unreadable or stale else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
