"""Parts 3D model of the car (main venv side): part labels per frame, then the fusion worker in .venv-3d.

1. ``segment_labels``: the car-parts model (same weights as the inspection) on every anonymised frame of a finished
   high-quality 3D dataset -> ``<dataset>/parts3d_labels/*.png`` (class id per pixel, 255 = none) + classes.json.
2. ``python -m avtopassport.recon3d.parts_cloud`` in .venv-3d: depth, multi-view fusion, car axes, part split
   -> ``<dataset>/parts_cloud.json`` (viewer payload) + ``parts_cloud_report.json``.

Method adapted from "Camry Parts Scan" by Team Trumen (see parts_cloud.py). Never raises; returns a result dict.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from avtopassport.paths import REPO_ROOT
from avtopassport.recon3d.build import DEFAULT_PYTHON, _env, _tail

PARTS_MODEL = REPO_ROOT / "models" / "parts" / "mitbersh_parts_segmentation.pt"
LABELS_DIR = "parts3d_labels"
OUT_NAME = "parts_cloud.json"


def segment_labels(images_dir: Path, labels_dir: Path, *, model_path: Path = PARTS_MODEL,
                   device: str = "cpu") -> int:
    """Per-pixel part class (highest-scoring mask wins) for every frame, as in the scan's seg_all.py."""
    import cv2  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    labels_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    (labels_dir / "classes.json").write_text(json.dumps({int(k): v for k, v in model.names.items()}))
    n = 0
    for f in sorted(images_dir.glob("*.jpg")):
        r = model.predict(str(f), imgsz=768, conf=0.3, iou=0.7, retina_masks=True, device=device, verbose=False)[0]
        H, W = r.orig_shape
        L = np.full((H, W), 255, np.uint8)
        S = np.zeros((H, W), np.float32)
        if r.masks is not None:
            for mk, c, s in zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                strict=False):
                upd = (mk > 0.5) & (s > S)
                L[upd] = int(c)
                S[upd] = s
        cv2.imwrite(str(labels_dir / f"{f.stem}.png"), L)
        n += 1
    return n


def build_parts3d(dataset: Path, *, device: str = "cpu", vehicle: str | None = None, length_m: float | None = None,
                  length_source: str | None = None, published: dict[str, float] | None = None,
                  python: Path = DEFAULT_PYTHON, timeout_s: int = 1800) -> dict[str, Any]:
    """{"ok", "error", "out", "report", "seconds"} for a dataset with images/ and sparse/0 (a recon3d_hq dir)."""
    dataset = Path(dataset).resolve()
    out = dataset / OUT_NAME
    res: dict[str, Any] = {"ok": False, "error": None, "out": None, "report": None, "seconds": None}
    t0 = time.perf_counter()
    try:
        if not (dataset / "sparse" / "0").is_dir() or not (dataset / "images").is_dir():
            raise RuntimeError("no structure-from-motion model to build the parts 3D model from")
        n = segment_labels(dataset / "images", dataset / LABELS_DIR, device=device)
        if n < 10:
            raise RuntimeError(f"only {n} frames to label")
        cmd = [str(python), "-m", "avtopassport.recon3d.parts_cloud", "--dataset", str(dataset),
               "--labels", str(dataset / LABELS_DIR), "--out", str(out), "--device", device]
        if vehicle:
            cmd += ["--vehicle", vehicle]
        if length_m:
            cmd += ["--length-m", str(length_m)]
        if length_source:
            cmd += ["--length-source", length_source]
        if published:
            cmd += ["--published", json.dumps(published)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=_env(device),
                              cwd=str(REPO_ROOT))
        (dataset / "parts_cloud.log").write_text(f"$ {' '.join(cmd)}\n--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr}")
        if proc.returncode != 0 or not out.is_file():
            last = (proc.stdout or "").strip().splitlines()[-1:] or [""]
            raise RuntimeError(f"parts 3D worker failed (exit {proc.returncode}): {last[0][:300] or _tail(proc.stderr, 300)}")
        res.update(ok=True, out=str(out), report=json.loads((dataset / "parts_cloud_report.json").read_text()))
    except subprocess.TimeoutExpired as exc:
        res["error"] = f"timed out after {exc.timeout} s"
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        res["error"] = f"{type(exc).__name__}: {exc}"
    res["seconds"] = round(time.perf_counter() - t0, 1)
    return res
