"""Anonymise the team's local test-set photos for labelling and evaluation (CLAUDE.md Phase 2). FAIL CLOSED.

  data/raw/testset/{day,night,wet}/*  -> load_image (full resolution, EXIF-rotated) -> Anonymizer ->
                                         data/local_testset/images/<slice>/<stem>.jpg  (long side <= 1600, no metadata)
  data/raw/testset/vin/*              -> SKIPPED: VIN photos are never copied (OCR input for scripts/eval.py only)
  data/raw/testset/vin_ground_truth.csv -> SKIPPED: real VINs stay in data/raw/ (gitignored)

Detectors: the same as scripts/blur_raw.py (its build_anonymizer: plate YOLO + YuNet faces + the parts model's
licence-plate class + the COCO person-head backstop). If any of them cannot be loaded nothing is written and the
exit code is 2. A photo whose detection fails is dropped (counted, never written).

data/local_testset/slices.csv gets one row per written image: file (relative to images/), slice (from the input
folder), brand, car, no_damage, notes. Existing brand / car / no_damage / notes values are kept; rows of files not
produced by this run are kept too (the team's data is never dropped), with a warning.
data/local_testset/blur_manifest.json records per output: sha256, size, plate / face counts (no source names or
hashes of the raw photos).

Only the blurred images in data/local_testset/images/ may be uploaded to CVAT / Roboflow
(docs/research/labelling_guide.md).

Usage (from the repo root):
    YOLO_AUTOINSTALL=false .venv/bin/python scripts/blur_testset.py --dry-run
    YOLO_AUTOINSTALL=false .venv/bin/python scripts/blur_testset.py [--device auto|cpu|cuda]

Exit codes: 0 every photo written; 1 something was dropped or unreadable; 2 a privacy detector is unavailable
(nothing written); 3 no input photos found.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
try:
    import avtopassport  # noqa: F401
except ImportError:  # not installed: use the source tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avtopassport.paths import DATA, RAW, REPO_ROOT  # noqa: E402

SLICES = ("day", "night", "wet")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
MAX_SIDE = 1600
SLICES_FIELDS = ("file", "slice", "brand", "car", "no_damage", "notes")
EXIT_OK, EXIT_PARTIAL, EXIT_PRIVACY_UNAVAILABLE, EXIT_NO_INPUT = 0, 1, 2, 3
DEFAULT_RAW = RAW / "testset"
DEFAULT_OUT = DATA / "local_testset"


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def list_inputs(raw: Path) -> list[tuple[Path, str]]:
    """(photo, slice) for every image in raw/<slice>/, sorted; other folders (vin/) are ignored."""
    out: list[tuple[Path, str]] = []
    for s in SLICES:
        d = raw / s
        if d.is_dir():
            out += [(p, s) for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXT]
    return out


def output_names(inputs: Sequence[tuple[Path, str]]) -> list[str]:
    """'<slice>/<stem>.jpg' per input; the extension is appended to the stem when two inputs of a slice share it."""
    used: set[str] = set()
    names = []
    for p, s in inputs:
        stem = p.stem
        if f"{s}/{stem}" in used:
            stem = f"{p.stem}_{p.suffix.lstrip('.').lower()}"
        used.add(f"{s}/{stem}")
        names.append(f"{s}/{stem}.jpg")
    return names


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]


def merge_slices(path: Path, produced: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    """slices.csv rows after this run: the slice of every produced file is set from its folder; brand, car,
    no_damage and notes already in the file are kept; existing rows of other files are kept. Sorted by file."""
    rows = {r.get("file", ""): r for r in read_rows(path) if r.get("file")}
    for file, slc in produced:
        old = rows.get(file, {})
        rows[file] = {**{k: old.get(k, "") for k in SLICES_FIELDS}, "file": file, "slice": slc}
    out = []
    for file in sorted(rows):
        r = rows[file]
        out.append({k: r.get(k, "") for k in SLICES_FIELDS})
    return out


def write_slices(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(SLICES_FIELDS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _blur_raw() -> ModuleType:
    """scripts/blur_raw.py as a module: its build_anonymizer / resolve_device are the single source of truth."""
    spec = importlib.util.spec_from_file_location("avto_blur_raw", Path(__file__).resolve().parent / "blur_raw.py")
    if spec is None or spec.loader is None:
        raise ImportError("scripts/blur_raw.py not found")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list what would be processed; load no models, write nothing")
    ap.add_argument("--device", default="auto", help="auto (cuda if available, else cpu) | cpu | cuda | cuda:N")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    inputs = list_inputs(args.raw)
    names = output_names(inputs)
    images_dir, slices_csv = args.out / "images", args.out / "slices.csv"
    if not inputs:
        print(f"no photos found: put the team's photos in {rel(args.raw)}/day/, {rel(args.raw)}/night/ and "
              f"{rel(args.raw)}/wet/ ({', '.join(sorted(IMAGE_EXT))}); nothing written")
        return EXIT_NO_INPUT
    skipped = [rel(args.raw / "vin"), rel(args.raw / "vin_ground_truth.csv")]
    if args.dry_run:
        for (p, s), name in zip(inputs, names):
            print(f"{s:5s}  {rel(p)} -> {rel(images_dir / name)}")
        per = {s: sum(1 for _, x in inputs if x == s) for s in SLICES}
        print(f"dry run: {len(inputs)} photos {per}; never copied: {skipped}; slices.csv: "
              f"{'merged into existing' if slices_csv.is_file() else 'new'}; nothing written")
        return EXIT_OK

    br = _blur_raw()
    device = br.resolve_device(args.device)
    t0 = time.perf_counter()
    try:
        anon = br.build_anonymizer(device)
    except (br.PrivacyUnavailable, ValueError) as exc:
        print(f"FAIL CLOSED: privacy detector unavailable ({exc}); nothing written", file=sys.stderr)
        return EXIT_PRIVACY_UNAVAILABLE
    print(f"detectors loaded on {device} in {time.perf_counter() - t0:.1f}s")

    from avtopassport.io import load_image, save_jpeg, sha256_file

    outputs: list[dict[str, Any]] = []
    produced: list[tuple[str, str]] = []
    dropped = unreadable = 0
    for (p, s), name in zip(inputs, names):
        try:
            img = load_image(p)
        except Exception as exc:
            unreadable += 1
            print(f"UNREADABLE {rel(p)}: {type(exc).__name__}")
            continue
        safe, rep = anon.anonymize(img)
        if safe is None:
            dropped += 1
            print(f"DROPPED {rel(p)}: privacy detection failed, nothing written")
            continue
        out = save_jpeg(safe, images_dir / name, max_side=MAX_SIDE)
        produced.append((name, s))
        outputs.append({"file": name, "slice": s, "sha256": sha256_file(out), "size": out.stat().st_size,
                        "n_plates": rep.n_plates, "n_faces": rep.n_faces, "passes": rep.passes})
        print(f"{name}: plates {rep.n_plates}, faces {rep.n_faces}")

    rows = merge_slices(slices_csv, produced)
    write_slices(slices_csv, rows)
    on_disk = {p.relative_to(images_dir).as_posix() for p in images_dir.rglob("*.jpg")} if images_dir.is_dir() else set()
    stale_rows = [r["file"] for r in rows if r["file"] not in on_disk]
    stale_files = sorted(on_disk - {n for n, _ in produced})
    summary = {"photos": len(inputs), "written": len(outputs), "dropped": dropped, "unreadable": unreadable,
               "plates": sum(o["n_plates"] for o in outputs), "faces": sum(o["n_faces"] for o in outputs),
               "seconds": round(time.perf_counter() - t0, 1)}
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "script": "scripts/blur_testset.py", "device": device, "max_side": MAX_SIDE,
                "never_copied": skipped, "summary": summary, "outputs": outputs}
    (args.out / "blur_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    print(f"wrote {rel(slices_csv)} ({len(rows)} rows): fill brand, car (same id for photos of one car) and "
          "no_damage (yes for cars without damage)")
    if stale_rows:
        print(f"WARNING: {len(stale_rows)} slices.csv row(s) have no image file (kept): {stale_rows[:5]}")
    if stale_files:
        print(f"WARNING: {len(stale_files)} image(s) in {rel(images_dir)} were not produced by this run: "
              f"{stale_files[:5]}")
    return EXIT_PARTIAL if dropped or unreadable else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
