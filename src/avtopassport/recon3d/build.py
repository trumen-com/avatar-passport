"""Run the DA3 worker (separate .venv-3d, numpy<2) in a subprocess from the main app venv.

The main venv never imports depth_anything_3; it only starts `python -m avtopassport.recon3d.worker` with the 3D
venv's interpreter and reads the files the worker writes (see worker.py for the output layout).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from functools import lru_cache
from pathlib import Path

from avtopassport.paths import REPO_ROOT

DEFAULT_PYTHON = REPO_ROOT / ".venv-3d" / "bin" / "python"
WORKER_MODULE = "avtopassport.recon3d.worker"


@lru_cache(maxsize=8)
def available(python: Path = DEFAULT_PYTHON) -> tuple[bool, str]:
    """(ok, message): the 3D venv interpreter exists and can import depth_anything_3. Cached per interpreter."""
    python = Path(python)
    if not python.exists():
        return False, f"3D python not found: {python}"
    try:
        proc = subprocess.run([str(python), "-c", "import depth_anything_3"], capture_output=True, text=True,
                              timeout=180, env=_env("cpu"))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"cannot run {python}: {exc}"
    if proc.returncode != 0:
        return False, f"depth_anything_3 import failed: {_tail(proc.stderr)}"
    return True, f"depth_anything_3 importable with {python}"


def _env(device: str) -> dict[str, str]:
    """Subprocess env: repo src on PYTHONPATH (works without the editable install); hide GPUs for cpu runs."""
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _tail(text: str | None, n: int = 1500) -> str:
    """Last n characters of a process output, stripped."""
    return (text or "").strip()[-n:]


def build(frames_dir: Path, out_dir: Path, *, device: str = "cpu", model: str = "depth-anything/DA3-BASE",
          max_frames: int = 30, process_res: int | None = None, threads: int | None = None,
          masks_dir: Path | None = None, python: Path = DEFAULT_PYTHON, timeout_s: int = 3600) -> dict:
    """Run the worker; returns {"ok", "result", "error", "seconds_wall", "out_dir", "ply", "ply_web", "preview"}.

    process_res None lets the worker pick (756 on cuda, 504 on cpu). Never raises for worker failures.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(python), "-m", WORKER_MODULE, "--frames-dir", str(frames_dir), "--out-dir", str(out_dir),
           "--device", device, "--model", model, "--max-frames", str(max_frames)]
    if process_res:
        cmd += ["--process-res", str(process_res)]
    if threads:
        cmd += ["--threads", str(threads)]
    if masks_dir is not None:
        cmd += ["--masks-dir", str(masks_dir)]
    report: dict = {"ok": False, "result": None, "error": None, "seconds_wall": 0.0, "out_dir": str(out_dir),
                    "ply": None, "ply_web": None, "preview": None}
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=_env(device),
                              cwd=str(REPO_ROOT))
    except subprocess.TimeoutExpired:
        report["error"] = f"3D worker timed out after {timeout_s} s"
    except OSError as exc:
        report["error"] = f"cannot start 3D worker ({python}): {exc}"
    else:
        (out_dir / "worker.log").write_text(f"$ {' '.join(cmd)}\n--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr}")
        result_path = out_dir / "result.json"
        if proc.returncode != 0:
            report["error"] = f"3D worker exited with {proc.returncode}: {_tail(proc.stderr) or _tail(proc.stdout)}"
        elif not result_path.exists():
            report["error"] = "3D worker exited 0 but wrote no result.json"
        else:
            try:
                report["result"] = json.loads(result_path.read_text())
                report["ok"] = True
            except json.JSONDecodeError as exc:
                report["error"] = f"unreadable result.json: {exc}"
    report["seconds_wall"] = round(time.perf_counter() - t0, 3)
    if report["ok"]:
        for key, name in (("ply", "points.ply"), ("ply_web", "points_web.ply"), ("preview", "preview.jpg")):
            path = out_dir / name
            report[key] = str(path) if path.exists() else None
    return report
