"""Generate the measured tables of docs/research/recon3d_smoke_results.md from the 3D run reports.

Sources (all gitignored run outputs, written by our code):
- outputs/recon3d/<run>/results.json  (scripts/recon3d_smoke.py) and outputs/recon3d/<run>/result.json (recon3d.worker)
- outputs/recon3d/hq/sfm_report.json, outputs/recon3d/hq/brush_report_*.json (scripts/recon3d_hq.py)
Only the text between <!-- BEGIN:x --> / <!-- END:x --> markers is replaced; --check exits 1 when stale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "research" / "recon3d_smoke_results.md"
OUT = REPO / "outputs" / "recon3d"


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def da3_rows() -> list[dict]:
    rows = []
    for d in sorted(OUT.iterdir()) if OUT.is_dir() else []:
        for name in ("results.json", "result.json"):
            r = _load(d / name)
            if not r or "seconds" not in r:
                continue
            sec = r["seconds"]
            rows.append({
                "run": r.get("run", d.name), "device": r.get("device"), "model": str(r.get("model", "")).split("/")[-1],
                "frames": r.get("n_frames"), "process_res": r.get("process_res"),
                "load_s": sec.get("model_load"), "infer_s": sec.get("inference"),
                "post_s": sec.get("backproject_write", sec.get("postprocess")),
                "threads": r.get("torch_threads"), "points": r.get("n_points_written", r.get("n_points")),
                "peak_cuda_mb": r.get("peak_cuda_mem_mb"), "source": f"outputs/recon3d/{d.name}/{name}",
            })
    return rows


def table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_not generated yet_"
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "\n".join("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |" for r in rows)
    return head + body


def blocks() -> dict[str, str]:
    out = {"da3": table(da3_rows(), ["run", "device", "model", "frames", "process_res", "threads", "load_s",
                                      "infer_s", "post_s", "points", "peak_cuda_mb", "source"])}
    sfm = _load(OUT / "hq" / "sfm_report.json")
    if sfm:
        s = sfm.get("seconds", {})
        out["sfm"] = table([{
            "images": sfm.get("n_images"), "registered": sfm.get("n_registered"), "points3d": sfm.get("n_points3d"),
            "mean_reproj_px": round(sfm.get("mean_reprojection_error_px", float("nan")), 3),
            "features_s": s.get("features_s"), "matching_s": s.get("matching_s"), "mapping_s": s.get("mapping_s"),
            "camera": f"{sfm['camera']['model']} {sfm['camera']['width']}x{sfm['camera']['height']}",
            "source": "outputs/recon3d/hq/sfm_report.json"}],
            ["images", "registered", "points3d", "mean_reproj_px", "features_s", "matching_s", "mapping_s", "camera",
             "source"])
    else:
        out["sfm"] = "_not generated yet_"
    brush = []
    for p in sorted((OUT / "hq").glob("brush_report_*.json")) if (OUT / "hq").is_dir() else []:
        r = _load(p) or {}
        brush.append({"name": p.stem.removeprefix("brush_report_"), "exit": r.get("exit_code"), "steps": r.get("steps"),
                      "seconds": r.get("seconds"), "max_splats": r.get("max_splats"),
                      "ply_mb": ", ".join(map(str, r.get("ply_mb", []))),
                      "psnr": "; ".join(r.get("psnr_lines", [])[-2:]) or "not reported",
                      "source": f"outputs/recon3d/hq/{p.name}"})
    held = _load(OUT / "hq" / "heldout_psnr.json")
    if held:
        for b in brush:
            b["psnr"] = (f"held-out mean {held['mean_psnr_db']} dB (whole image), {held['mean_psnr_car_db']} dB "
                         f"(car mask), {held['n_heldout']} views; outputs/recon3d/hq/heldout_psnr.json")
    out["brush"] = table(brush, ["name", "exit", "steps", "seconds", "max_splats", "ply_mb", "psnr", "source"])
    return out


def render(text: str, new: dict[str, str]) -> str:
    for name, body in new.items():
        b, e = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
        if b not in text or e not in text:
            continue
        pre, rest = text.split(b, 1)
        _, post = rest.split(e, 1)
        text = f"{pre}{b}\n{body}\n{e}{post}"
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = DOC.read_text()
    new = render(text, blocks())
    if a.check:
        print("up to date" if new == text else "STALE")
        return 0 if new == text else 1
    DOC.write_text(new)
    print(f"updated {DOC.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
