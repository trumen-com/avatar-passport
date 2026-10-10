"""Generate notebooks/05_eval_local_testset.ipynb (Phase 2 evaluation) from the cell sources below.

The notebook is thin: every metric comes from scripts/eval.py (loaded as a module), so the notebook and the CLI
cannot disagree. Generated, never edited by hand: nbformat v4, no outputs, execution_count None, deterministic
cell ids; every code cell is syntax-checked and the notebook is validated with nbformat.

Usage:
    python scripts/build_notebook_05.py                 # write notebooks/05_eval_local_testset.ipynb
    python scripts/build_notebook_05.py --out X.ipynb   # write elsewhere (tests)
    python scripts/build_notebook_05.py --check         # exit 1 if the notebook on disk is out of date
"""

from __future__ import annotations

import argparse
import ast
import sys
import textwrap
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "notebooks" / "05_eval_local_testset.ipynb"

MD_TITLE = r"""
# AvtoPassport 05: evaluation on our local test set (Phase 2)

**Purpose.** CLAUDE.md Phase 2: mask mAP50 per damage model, part-level recall, false positives per clean car,
VIN exact-match rate and latency on our own blurred, labelled photos, sliced by day / night / wet and brand, with
per-class threshold calibration (baseline 0.25 vs calibrated) and a failure-case gallery. Every number comes from
`scripts/eval.py`; this notebook only calls it and shows the results. Method and definitions:
`docs/research/evaluation.md`; how to label: `docs/research/labelling_guide.md`.

**Inputs** (the notebook refuses to compute anything without them):
`data/local_testset/annotations.json` (COCO polygons), `data/local_testset/slices.csv`, the blurred photos in
`data/local_testset/images/<slice>/` (from `scripts/blur_testset.py`), and optionally the VIN photos +
`vin_ground_truth.csv` in `data/raw/testset/` (never committed, rates only).

**Privacy rule.** Only the blurred test-set images are read and shown. The VIN step prints rates, never VINs.
"""

MD_BOOTSTRAP = r"""
## 1. Bootstrap

Forces `YOLO_AUTOINSTALL=false` before anything imports ultralytics, finds the repository root (`pyproject.toml`;
on Colab `REPO_DIR`, cloned from `REPO_URL` or unpacked from `DRIVE_ZIP`), changes into it and puts `src/` first on
`sys.path`. On Colab it installs `requirements-colab.txt` and copies the local test set from `DRIVE_TESTSET_DIR`
(a Drive folder holding `annotations.json`, `slices.csv` and `images/`) and, if set, the VIN folder from
`DRIVE_VIN_DIR` (holding `vin/` and `vin_ground_truth.csv`).
"""

CODE_BOOTSTRAP = r'''
# 1. Bootstrap. YOLO_AUTOINSTALL is forced first: scripts/eval.py imports ultralytics.
import os

os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ.setdefault("YOLO_OFFLINE", "true")

import shutil
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

# ---- Colab settings (ignored locally) ----
REPO_URL = ""  # e.g. "https://github.com/<user>/avtopassport.git": cloned into REPO_DIR
REPO_DIR = "/content/avtopassport"
DRIVE_ZIP = ""  # e.g. "/content/drive/MyDrive/avtopassport.zip": a zip of the repository
DRIVE_TESTSET_DIR = ""  # e.g. "/content/drive/MyDrive/avtopassport_testset": annotations.json, slices.csv, images/
DRIVE_VIN_DIR = ""  # optional: holds vin/ and vin_ground_truth.csv (copied to data/raw/testset/, never committed)


def run_cmd(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("\n".join((proc.stdout + proc.stderr).strip().splitlines()[-20:]))
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def find_repo_root(start: Path) -> Path:
    for d in (start, *start.parents):
        if (d / "pyproject.toml").is_file():
            return d
    raise FileNotFoundError(f"no pyproject.toml above {start}: start Jupyter inside the repository")


if "ultralytics" in sys.modules:
    print("note: ultralytics was imported before this cell; restart the kernel so YOLO_AUTOINSTALL applies")

if IN_COLAB:
    if any(p.startswith("/content/drive") for p in (DRIVE_ZIP, DRIVE_TESTSET_DIR, DRIVE_VIN_DIR)) and not Path(
            "/content/drive/MyDrive").exists():
        from google.colab import drive

        drive.mount("/content/drive")
    repo = Path(REPO_DIR)
    if not (repo / "pyproject.toml").is_file():
        if REPO_URL:
            run_cmd(["git", "clone", "--depth", "1", REPO_URL, str(repo)])
        elif DRIVE_ZIP:
            unpacked = Path("/content/_avtopassport_unzip")
            shutil.rmtree(unpacked, ignore_errors=True)
            shutil.unpack_archive(DRIVE_ZIP, unpacked)
            roots = sorted((p.parent for p in unpacked.rglob("pyproject.toml")), key=lambda p: len(p.parts))
            if not roots:
                raise FileNotFoundError(f"no pyproject.toml inside {DRIVE_ZIP}")
            shutil.move(str(roots[0]), str(repo))
        else:
            raise FileNotFoundError(f"Colab: set REPO_URL or DRIVE_ZIP, or put the repository at {REPO_DIR}")
    REPO_ROOT = repo.resolve()
else:
    REPO_ROOT = find_repo_root(Path.cwd().resolve())

os.chdir(REPO_ROOT)
SRC = str(REPO_ROOT / "src")
if sys.path[0] != SRC:
    sys.path.insert(0, SRC)

if IN_COLAB:
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"])
    if DRIVE_TESTSET_DIR:
        shutil.copytree(DRIVE_TESTSET_DIR, REPO_ROOT / "data" / "local_testset", dirs_exist_ok=True)
    if DRIVE_VIN_DIR:
        shutil.copytree(DRIVE_VIN_DIR, REPO_ROOT / "data" / "raw" / "testset", dirs_exist_ok=True)

import avtopassport

if Path(avtopassport.__file__).resolve().parent != (REPO_ROOT / "src" / "avtopassport").resolve():
    raise RuntimeError(f"avtopassport was imported from {avtopassport.__file__}, not {SRC}: restart the kernel")
print("REPO_ROOT:", REPO_ROOT, "| Colab:", IN_COLAB, "| YOLO_AUTOINSTALL:", os.environ["YOLO_AUTOINSTALL"])
'''

MD_STATUS = r"""
## 2. Load `scripts/eval.py` and check the inputs

Prints what is present and what is missing. `DATA_READY` is False until at least one labelled image with its
blurred file exists; the later cells then do nothing.
"""

CODE_STATUS = r'''
# 2. Load scripts/eval.py as a module and check the inputs.
import importlib.util

_spec = importlib.util.spec_from_file_location("avto_eval", REPO_ROOT / "scripts" / "eval.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules["avto_eval"] = ev
_spec.loader.exec_module(ev)

dataset, status_lines = ev.data_status()
print("\n".join(status_lines))
DATA_READY = bool(ev.usable_images(dataset))
print("DATA_READY:", DATA_READY)
if not DATA_READY:
    print("Nothing to evaluate yet: label the blurred images (docs/research/labelling_guide.md) and export COCO 1.0.")
'''

MD_RUN = r"""
## 3. Run the evaluation

Same as `python scripts/eval.py --split <SPLIT> --device <DEVICE>`: writes `outputs/eval/<run>/metrics.json`, the
tables in `docs/research/evaluation.md` and the gallery in `docs/assets/failure_cases/`. `SPLIT = "kfold:5"` reports
calibrated metrics on held-out cars (calibrating and reporting on the same images is optimistic).
"""

CODE_RUN = r'''
# 3. Run scripts/eval.py (all metrics are computed there).
import json

DEVICE = "auto"  # auto | cpu | cuda
SPLIT = "kfold:5"  # none | kfold:K | holdout:F (by car)
RUN = None  # None: UTC time + device
EXTRA_ARGS: list[str] = []  # e.g. ["--latency-n", "0"] or ["--models", "damage_mitbersh"]

metrics = None
if DATA_READY:
    argv = ["--device", DEVICE, "--split", SPLIT, *(["--run", RUN] if RUN else []), *EXTRA_ARGS]
    rc = ev.main(argv)
    print("exit code:", rc)
    runs = sorted((ev.EVAL_OUT).glob("*/metrics.json"), key=lambda p: p.stat().st_mtime)
    if runs:
        metrics = json.loads(runs[-1].read_text(encoding="utf-8"))
        print("metrics:", runs[-1].relative_to(REPO_ROOT))
else:
    print("skipped: DATA_READY is False")
'''

MD_TABLES = r"""
## 4. Tables

The same Markdown blocks that `scripts/eval.py` writes between the markers in `docs/research/evaluation.md`.
"""

CODE_TABLES = r'''
# 4. Show the generated tables.
from IPython.display import Markdown, display

if metrics is not None:
    for name, body in ev.render_blocks(metrics).items():
        display(Markdown(f"### {name}\n\n{body}"))
else:
    print("no metrics yet")
'''

MD_GALLERY = r"""
## 5. Failure-case gallery

Blurred test-set images annotated by `scripts/eval.py`: red = missed ground truth, yellow = false positive,
green = true positive (calibrated thresholds).
"""

CODE_GALLERY = r'''
# 5. Show the failure-case gallery written by scripts/eval.py (blurred images only).
from IPython.display import Image as IPyImage

if metrics is not None:
    for model, cases in metrics.get("gallery", {}).items():
        print(model, f"({len(cases)} images)")
        for case in cases:
            print(case["file"], "| slice", case["slice"], "| missed", case["missed"], "| FP", case["fp"])
            display(IPyImage(filename=str(REPO_ROOT / case["file"]), width=640))
else:
    print("no gallery yet")
'''

CELLS: list[tuple[str, str, str | None, list[str]]] = [
    ("c00-title", MD_TITLE, None, []),
    ("c01-bootstrap", MD_BOOTSTRAP, CODE_BOOTSTRAP, ["bootstrap"]),
    ("c02-status", MD_STATUS, CODE_STATUS, ["status"]),
    ("c03-run", MD_RUN, CODE_RUN, ["run"]),
    ("c04-tables", MD_TABLES, CODE_TABLES, ["tables"]),
    ("c05-gallery", MD_GALLERY, CODE_GALLERY, ["gallery"]),
]


def _clean(src: str) -> str:
    return textwrap.dedent(src).strip("\n")


def build() -> nbformat.NotebookNode:
    """The notebook: a markdown cell before each code cell, no outputs, deterministic ids."""
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    }
    cells = []
    for cell_id, md_src, code_src, tags in CELLS:
        cells.append(nbformat.v4.new_markdown_cell(_clean(md_src), id=f"{cell_id}-md"))
        if code_src is not None:
            code = _clean(code_src)
            ast.parse(code, filename=cell_id)  # fail early on a syntax error
            cell = nbformat.v4.new_code_cell(code, id=cell_id, metadata={"tags": tags} if tags else {})
            cell.outputs, cell.execution_count = [], None
            cells.append(cell)
    nb.cells = cells
    nbformat.validate(nb)
    return nb


def render(nb: nbformat.NotebookNode) -> str:
    text = nbformat.writes(nb, version=4)
    return text if text.endswith("\n") else text + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="notebook path to write (or to check)")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if --out differs from the build")
    args = ap.parse_args(argv)
    text = render(build())
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.is_file() else None
        if current != text:
            print(f"{args.out} is out of date: run python scripts/build_notebook_05.py", file=sys.stderr)
            return 1
        print(f"{args.out} is up to date")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    n_code = sum(1 for _, _, code, _ in CELLS if code is not None)
    print(f"wrote {args.out} ({len(CELLS)} markdown + {n_code} code cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
