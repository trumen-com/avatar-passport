"""Cut real panel joints out of the DrBimmer car-parts dataset (998 photos, panel polygons, MIT licence).

For every pair of neighbouring painted panels in a photo, find their shared boundary, rotate the photo so the boundary
runs vertically, and save a crop with panel A on the left and panel B on the right. Keep a crop only when the paint
sample boxes on both sides lie inside the two panels' outlines.

Output: <out>/crops/*.jpg and <out>/joints.json (image, panels, boundary length in px, crop size).
Usage: python3 -I extract_joints.py <dataset dir> <out dir>
"""
import json, sys, os, glob
import numpy as np, cv2

src, out = sys.argv[1], sys.argv[2]
ann_dir = os.path.join(src, "Car damages dataset/File1/ann")   # the dataset's folders are swapped: parts live here
img_dir = os.path.join(src, "Car damages dataset/File1/img")
os.makedirs(os.path.join(out, "crops"), exist_ok=True)

PAINTED = {"Hood", "Fender", "Front-door", "Back-door", "Quarter-panel", "Trunk", "Front-bumper", "Back-bumper", "Roof"}
# neighbours that share a seam on a real car; map to the app's joint kinds
PAIRS = {
    frozenset({"Fender", "Front-door"}): "fender-door",
    frozenset({"Front-door", "Back-door"}): "door-door",
    frozenset({"Back-door", "Quarter-panel"}): "door-quarter",
    frozenset({"Hood", "Fender"}): "hood-fender",
    frozenset({"Fender", "Front-bumper"}): "fender-bumper",
    frozenset({"Quarter-panel", "Back-bumper"}): "quarter-bumper",
    frozenset({"Trunk", "Quarter-panel"}): "trunk-quarter",
    frozenset({"Hood", "Front-bumper"}): "hood-bumper",
    frozenset({"Trunk", "Back-bumper"}): "trunk-bumper",
}
PLASTIC = {"Front-bumper", "Back-bumper"}

rows = []
files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
for f in files:
    a = json.load(open(f))
    name = os.path.basename(f)[:-5]
    img = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_COLOR)
    if img is None:
        continue
    H, W = img.shape[:2]
    objs = [o for o in a["objects"] if o["classTitle"] in PAINTED and o.get("geometryType") == "polygon"]
    # everything that is not body paint: lights, glass, plates, grille, wheels, mirrors, trim
    other = np.zeros((H, W), np.uint8)
    for o in a["objects"]:
        if o["classTitle"] not in PAINTED and o.get("geometryType") == "polygon":
            cv2.fillPoly(other, [np.array(o["points"]["exterior"], np.int32)], 1)
    other = cv2.dilate(other, np.ones((7, 7), np.uint8))
    masks = []
    for o in objs:
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.array(o["points"]["exterior"], np.int32)], 1)
        for hole in o["points"].get("interior", []):
            cv2.fillPoly(m, [np.array(hole, np.int32)], 0)
        masks.append((o["classTitle"], m & (1 - other)))
    k = np.ones((5, 5), np.uint8)
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            ca, ma = masks[i]; cb, mb = masks[j]
            kind = PAIRS.get(frozenset({ca, cb}))
            if not kind:
                continue
            seam = (cv2.dilate(ma, k) & cv2.dilate(mb, k)).astype(bool)
            ys, xs = np.nonzero(seam)
            if len(xs) < 40:
                continue
            pts = np.stack([xs, ys], 1).astype(np.float32)
            c = pts.mean(0)
            u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
            d = vt[0]                                   # along the seam
            proj = (pts - c) @ d
            L = float(np.percentile(proj, 95) - np.percentile(proj, 5))
            if L < 50:
                continue
            # rotate so the seam is vertical
            ang = np.degrees(np.arctan2(d[1], d[0])) - 90
            R = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, 1.0)
            rot = cv2.warpAffine(img, R, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            ra = cv2.warpAffine(ma * 255, R, (W, H), flags=cv2.INTER_NEAREST) > 0
            rb = cv2.warpAffine(mb * 255, R, (W, H), flags=cv2.INTER_NEAREST) > 0
            ch = int(L * 0.9); cw = int(min(ch * 0.9, L * 0.9))
            x0, y0 = int(c[0] - cw / 2), int(c[1] - ch / 2)
            if x0 < 0 or y0 < 0 or x0 + cw > W or y0 + ch > H or cw < 40:
                continue
            crop, A, B = rot[y0:y0 + ch, x0:x0 + cw], ra[y0:y0 + ch, x0:x0 + cw], rb[y0:y0 + ch, x0:x0 + cw]
            left, right = np.s_[:, : cw // 2], np.s_[:, cw // 2 :]
            # which panel is on the left?
            if A[left].mean() + B[right].mean() < B[left].mean() + A[right].mean():
                ca, cb, A, B = cb, ca, B, A
            # the app's sample boxes: 3.5%..23.5% of the width away from the gap, 18%..82% of the height
            ys0, ys1 = int(ch * 0.18), int(ch * 0.82)
            bl = np.s_[ys0:ys1, int(cw * (0.5 - 0.235)) : int(cw * (0.5 - 0.035))]
            br = np.s_[ys0:ys1, int(cw * (0.5 + 0.035)) : int(cw * (0.5 + 0.235))]
            cov = (float(A[bl].mean()), float(B[br].mean()))
            if min(cov) < 0.92:
                continue
            # where the seam actually sits in the crop (annotation), for checking the gap finder
            seam_r = cv2.warpAffine(seam.astype(np.uint8) * 255, R, (W, H), flags=cv2.INTER_NEAREST)[y0:y0 + ch, x0:x0 + cw] > 0
            sx = np.nonzero(seam_r[ys0:ys1].any(0))[0]
            gap_ann = float(np.median(np.nonzero(seam_r[ys0:ys1])[1]) / cw) if sx.size else 0.5
            jid = f"{len(rows):04d}"
            cv2.imwrite(os.path.join(out, "crops", jid + ".jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"id": jid, "image": name, "a": ca, "b": cb, "kind": kind,
                         "plastic": bool({ca, cb} & PLASTIC), "seam_px": round(L), "w": cw, "h": ch,
                         "coverage": [round(v, 3) for v in cov], "gap_ann": round(gap_ann, 4)})
json.dump(rows, open(os.path.join(out, "joints.json"), "w"), indent=1)
from collections import Counter
print(len(files), "photos ->", len(rows), "joints from", len({r["image"] for r in rows}), "photos")
print(Counter(r["kind"] for r in rows))
print("seam length px: median", int(np.median([r["seam_px"] for r in rows])), "p10", int(np.percentile([r["seam_px"] for r in rows], 10)))
