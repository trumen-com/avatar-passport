"""Cut Paint Check gap photos out of the demo car's photos, for the "Use demo photos" button.

Same procedure as Paint Check's eval/extract_joints.py (Team Trumen), with our car-parts model giving the panel outlines
instead of hand-drawn polygons: for every pair of neighbouring painted panels, find their shared seam, rotate the photo
so the seam runs vertically, crop around it with Paint Check's joint panel `a` on the left, and keep the crop only when
the paint sample boxes on both sides lie (>= 92 %) inside the two panels. The longest seam per joint is kept.

Input: the blurred photos in data/processed/images (plates and faces already blurred). Output: web/paint/demo/*.jpg and
web/paint/demo/index.json ({joint, a, b, file, photo, seam_px, coverage}). Analysis itself runs live in the browser.
Usage: YOLO_AUTOINSTALL=false .venv/bin/python scripts/paintcheck_demo_crops.py [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "models" / "parts" / "mitbersh_parts_segmentation.pt"
PHOTOS = REPO / "data" / "processed" / "images"
OUT = REPO / "web" / "paint" / "demo"

# parts-model class -> Paint Check panel id (left/right: the car's own, as in web/paint/INTEGRATION.md)
TO_PAINT = {"front-bumper": "f_bumper", "back-bumper": "r_bumper", "hood": "hood", "trunk": "trunk", "roof": "roof",
            "left_fender": "lf_fender", "right_fender": "rf_fender", "left_front-door": "lf_door",
            "right_front-door": "rf_door", "left_back-door": "lr_door", "right_back-door": "rr_door",
            "left_quarter-panel": "lr_quarter", "right_quarter-panel": "rr_quarter"}
# Paint Check's JOINTS (web/paint/car.js): id -> (a = panel on the left in the photo, b = on the right)
JOINTS = {"j1": ("f_bumper", "lf_fender"), "j2": ("lf_fender", "hood"), "j3": ("lf_fender", "lf_door"),
          "j4": ("lf_door", "lr_door"), "j5": ("lr_door", "lr_quarter"), "j6": ("lr_quarter", "r_bumper"),
          "j7": ("lr_quarter", "trunk"), "j8": ("trunk", "rr_quarter"), "j9": ("r_bumper", "rr_quarter"),
          "j10": ("rr_quarter", "rr_door"), "j11": ("rr_door", "rf_door"), "j12": ("rf_door", "rf_fender"),
          "j13": ("hood", "rf_fender"), "j14": ("rf_fender", "f_bumper")}
BY_PAIR = {frozenset(v): k for k, v in JOINTS.items()}


def masks_of(model, path: Path, device: str) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    r = model.predict(str(path), imgsz=768, conf=0.3, iou=0.7, retina_masks=True, device=device, verbose=False)[0]
    H, W = img.shape[:2]
    panels: dict[str, np.ndarray] = {}
    other = np.zeros((H, W), np.uint8)  # lights, glass, wheels, mirrors, plate: never part of a paint sample
    if r.masks is not None:
        for mk, c in zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy(), strict=False):
            name = model.names[int(c)]
            m = (cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8)
            if name in TO_PAINT:
                pid = TO_PAINT[name]
                panels[pid] = panels.get(pid, np.zeros((H, W), np.uint8)) | m
            else:
                other |= m
    other = cv2.dilate(other, np.ones((7, 7), np.uint8))
    return img, {k: v & (1 - other) for k, v in panels.items()}, other


def seam_crops(img: np.ndarray, panels: dict[str, np.ndarray], photo: str) -> list[dict]:
    H, W = img.shape[:2]
    k = np.ones((5, 5), np.uint8)
    out = []
    ids = sorted(panels)
    for i, pa in enumerate(ids):
        for pb in ids[i + 1:]:
            jid = BY_PAIR.get(frozenset({pa, pb}))
            if not jid:
                continue
            ma, mb = panels[pa], panels[pb]
            seam = (cv2.dilate(ma, k) & cv2.dilate(mb, k)).astype(bool)
            ys, xs = np.nonzero(seam)
            if len(xs) < 40:
                continue
            pts = np.stack([xs, ys], 1).astype(np.float32)
            c = pts.mean(0)
            d = np.linalg.svd(pts - c, full_matrices=False)[2][0]
            proj = (pts - c) @ d
            L = float(np.percentile(proj, 95) - np.percentile(proj, 5))
            if L < 50:
                continue
            R = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), np.degrees(np.arctan2(d[1], d[0])) - 90, 1.0)
            rot = cv2.warpAffine(img, R, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            ra = cv2.warpAffine(ma * 255, R, (W, H), flags=cv2.INTER_NEAREST) > 0
            rb = cv2.warpAffine(mb * 255, R, (W, H), flags=cv2.INTER_NEAREST) > 0
            ch = int(L * 0.9)
            cw = int(min(ch * 0.9, L * 0.9))
            x0, y0 = int(c[0] - cw / 2), int(c[1] - ch / 2)
            if x0 < 0 or y0 < 0 or x0 + cw > W or y0 + ch > H or cw < 40:
                continue
            crop, A, B = rot[y0:y0 + ch, x0:x0 + cw], ra[y0:y0 + ch, x0:x0 + cw], rb[y0:y0 + ch, x0:x0 + cw]
            left, right = np.s_[:, : cw // 2], np.s_[:, cw // 2:]
            if A[left].mean() + B[right].mean() < B[left].mean() + A[right].mean():
                pa_l, pb_r, A, B = pb, pa, B, A
            else:
                pa_l, pb_r = pa, pb
            if (pa_l, pb_r) != JOINTS[jid]:  # Paint Check wants its panel `a` on the left: mirror the crop
                crop, A, B = crop[:, ::-1], B[:, ::-1], A[:, ::-1]
            ys0, ys1 = int(ch * 0.18), int(ch * 0.82)
            bl = np.s_[ys0:ys1, int(cw * 0.265): int(cw * 0.465)]
            br = np.s_[ys0:ys1, int(cw * 0.535): int(cw * 0.735)]
            cov = (float(A[bl].mean()), float(B[br].mean()))
            if min(cov) < 0.92:
                continue
            out.append({"joint": jid, "a": JOINTS[jid][0], "b": JOINTS[jid][1], "photo": photo, "seam_px": round(L),
                        "coverage": [round(v, 3) for v in cov], "crop": np.ascontiguousarray(crop)})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    from ultralytics import YOLO

    model = YOLO(str(MODEL))
    best: dict[str, dict] = {}
    for p in sorted(PHOTOS.glob("*.jpg")):
        img, panels, _ = masks_of(model, p, a.device)
        for c in seam_crops(img, panels, p.name):
            if c["joint"] not in best or c["seam_px"] > best[c["joint"]]["seam_px"]:
                best[c["joint"]] = c
        print(p.name, "panels:", sorted(panels), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.jpg"):
        old.unlink()
    index = []
    for jid in sorted(best, key=lambda j: int(j[1:])):
        c = best[jid]
        name = f"{jid}.jpg"
        crop = c.pop("crop")
        s = min(1.0, 720 / max(crop.shape[:2]))
        crop = cv2.resize(crop, (round(crop.shape[1] * s), round(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        index.append({**c, "file": name})
    (OUT / "index.json").write_text(json.dumps({"source": "data/processed/images (blurred demo photos)",
                                                "method": "scripts/paintcheck_demo_crops.py", "joints": index}, indent=1))
    print(f"{len(index)} of {len(JOINTS)} joints:", [(c['joint'], c['photo'], c['seam_px']) for c in index])
    return 0


if __name__ == "__main__":
    sys.exit(main())
