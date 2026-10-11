"""Part-labelled 3D point cloud of the car: COLMAP poses + per-frame part labels + monocular depth, fused.

Method, parameters and most of the code are adapted from "Camry Parts Scan" by Team Trumen (WIUT Hackathon 2026),
``pipeline/tune.py`` and ``pipeline/parts.py`` (see DATA_SOURCES.md). Steps:

1. Depth per frame with Depth Anything V2 Small (Apache-2.0, revision pinned below), relative inverse depth.
2. Per frame, RANSAC fit ``1/z = a*d + b + c*u + e*v`` to the COLMAP points seen in that frame ("inv_plane").
3. Back-project car pixels (plate class dropped, mask edges eroded). Keep a point only when 2 of its 6 nearest views
   agree within 1.5 % depth and on the same part.
4. Fuse into 2 cm voxels seen in >= 3 views with a majority label; fill holes from the unchecked depth.
5. Car axes from the walkaround plane, a RANSAC ground plane and left/right symmetry. Paired parts are split by side.

Scale: COLMAP has no scale. Sizes in metres are reported only when the car's published length is given
(``--length-m``); otherwise a nominal length is used for display and ``meta.metric`` is false.

Runs in .venv-3d (pycolmap, scipy, transformers):
    python -m avtopassport.recon3d.parts_cloud --dataset <recon3d_hq dir> --labels <dir> --out <parts_cloud.json>
Inputs are the anonymised frames in ``<dataset>/images`` and the model in ``<dataset>/sparse/0``.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import numpy as np

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"  # Apache-2.0, checked 2026-10-11
NOMINAL_LENGTH_M = 4.6        # display scale when the real length is unknown (sizes are then not reported)
VIEWER_MAX_POINTS = 150_000
# Tuned parameters from the Camry Parts Scan (tune_final.json). Their voxel / fill_cell / ground margin were set in
# their COLMAP units (0.834 m each), so the metric equivalents are used here and converted with the measured unit.
PARAMS: dict[str, Any] = dict(ds=2, stride=2, erode=10, align="inv_plane", fit_tol=0.05, max_relerr=0.08, mv_k=6,
                              mv_tau=0.015, mv_min=2, voxel=0.0167, min_views=3, agree=0.5, fill=1, fill_views=3,
                              fill_cell=0.05, ground_margin=1.0, refine=1)
PRETTY = {"back-bumper": "Rear bumper", "back-windshield": "Rear windshield", "front-bumper": "Front bumper",
          "grille": "Front grille", "hood": "Hood", "back-door": "Rear door", "back-wheel": "Rear wheel",
          "back-window": "Rear door glass", "fender": "Front fender", "front-door": "Front door",
          "front-wheel": "Front wheel", "front-window": "Front door glass", "headlight": "Headlight",
          "mirror": "Side mirror", "quarter-panel": "Rear quarter panel", "rocker-panel": "Rocker panel",
          "tail-light": "Tail light", "roof": "Roof", "trunk": "Trunk lid", "windshield": "Windshield"}
# lateral offset (metres) beyond which a paired part counts as left or right
PAIRED = {"back-door": .3, "back-wheel": .3, "back-window": .3, "fender": .3, "front-door": .3, "front-wheel": .3,
          "front-window": .3, "headlight": .08, "mirror": .4, "quarter-panel": .3, "rocker-panel": .3,
          "tail-light": .08}


def base_name(cls: str) -> str:
    """'left_front-door' -> 'front-door' (sides come from the car's own axes, not the 2D label)."""
    for pre in ("left_", "right_"):
        if cls.startswith(pre):
            return cls[len(pre):]
    return cls


def load_labels(labels_dir: Path) -> tuple[dict[str, np.ndarray], list[str], np.ndarray]:
    """Label maps (uint8 class id, 255 = none) per image stem, the base part names, and a class -> base lookup."""
    from PIL import Image  # noqa: PLC0415

    classes = {int(k): v for k, v in json.loads((labels_dir / "classes.json").read_text()).items()}
    base = sorted({base_name(v) for v in classes.values()})
    lut = np.full(256, 255, np.uint8)
    for k, v in classes.items():
        lut[k] = base.index(base_name(v))
    maps = {p.stem: lut[np.array(Image.open(p))] for p in sorted(labels_dir.glob("*.png"))}
    return maps, base, lut


def estimate_depth(images: list[Path], device: str) -> dict[str, np.ndarray]:
    """Relative inverse depth (disparity-like) per image at the model's output resolution."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import pipeline  # noqa: PLC0415

    dev = 0 if device.startswith("cuda") and torch.cuda.is_available() else -1
    pipe = pipeline("depth-estimation", model=DEPTH_MODEL, revision=DEPTH_REVISION, device=dev)
    out = {}
    for p in images:
        d = pipe(Image.open(p).convert("RGB"))["predicted_depth"]
        out[p.stem] = d.squeeze().detach().cpu().numpy().astype(np.float32)
    return out


def _frame(up: np.ndarray, fwd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fwd = fwd - (fwd @ up) * up
    fwd = fwd / np.linalg.norm(fwd)
    return up, fwd, np.cross(up, fwd)


def fuse(rec: Any, images_dir: Path, labels: dict[str, np.ndarray], depth: dict[str, np.ndarray],  # noqa: C901, PLR0912, PLR0915
         base: list[str], P: dict[str, Any] = PARAMS) -> dict[str, Any]:
    """Fused, car-aligned cloud in COLMAP units: X (N,3) in (forward, left, up), C, L (base index), NV, AG + stats."""
    from PIL import Image  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    B = {n: i for i, n in enumerate(base)}
    plate = B.get("license-plate", 255)
    ds = P["ds"]
    views = []
    for iid, im in rec.images.items():
        n = Path(im.name).stem
        if n not in labels or n not in depth:
            continue
        Lf = labels[n]
        Hf, Wf = Lf.shape
        H, W = Hf // ds, Wf // ds
        L = Lf[::ds, ::ds][:H, :W]
        d = np.array(Image.fromarray(depth[n]).resize((W, H), Image.BILINEAR))
        img = np.array(Image.open(images_dir / im.name).convert("RGB").resize((W, H), Image.BILINEAR))
        cam = rec.cameras[im.camera_id]
        Rt = im.cam_from_world().matrix()
        uv, z = [], []
        for p2 in im.points2D:
            if p2.has_point3D():
                c = Rt[:, :3] @ rec.points3D[p2.point3D_id].xyz + Rt[:, 3]
                if c[2] > 0:
                    uv.append(p2.xy)
                    z.append(c[2])
        if len(z) < 50:
            continue
        uv, z = np.array(uv) / ds, np.array(z)
        ui = np.clip(uv[:, 0].astype(int), 0, W - 1)
        vi = np.clip(uv[:, 1].astype(int), 0, H - 1)
        dv = d[vi, ui]
        oncar = L[vi, ui] != 255
        w_idx = np.where(oncar)[0] if oncar.sum() > 30 else np.arange(len(z))
        A = np.stack([dv, np.ones_like(dv), ui / W - 0.5, vi / H - 0.5], 1)  # inv_plane
        y = 1 / z
        rng = np.random.default_rng(iid)
        best = None
        for _ in range(300):
            s = rng.choice(w_idx, A.shape[1], replace=False)
            try:
                coef = np.linalg.solve(A[s], y[s])
            except np.linalg.LinAlgError:
                continue
            zp = 1 / np.clip(A[w_idx] @ coef, 1e-6, None)
            inl = w_idx[np.abs(zp - z[w_idx]) / z[w_idx] < P["fit_tol"]]
            if best is None or len(inl) > len(best[1]):
                best = (coef, inl)
        if best is None or len(best[1]) < 20:
            continue
        coef = np.linalg.lstsq(A[best[1]], y[best[1]], rcond=None)[0]

        def depth_of(dd: np.ndarray, uu: np.ndarray, vv: np.ndarray, coef: np.ndarray = coef, W: int = W,
                     H: int = H) -> np.ndarray:
            return 1 / np.clip(coef[0] * dd + coef[1] + coef[2] * (uu / W - 0.5) + coef[3] * (vv / H - 0.5), 1e-6,
                               None)

        zr = depth_of(dv[w_idx], ui[w_idx], vi[w_idx])
        relerr = float(np.median(np.abs(zr - z[w_idx]) / z[w_idx]))
        if relerr > P["max_relerr"]:
            continue
        vv_, uu_ = np.mgrid[0:H, 0:W]
        views.append(dict(iid=iid, L=L, Z=depth_of(d, uu_, vv_).astype(np.float32), img=img, cam=cam, Rt=Rt, H=H,
                          W=W, C=im.projection_center(), relerr=relerr))
    if len(views) < 5:
        raise RuntimeError(f"only {len(views)} views could be aligned to the SfM points")
    centers = np.array([v["C"] for v in views])
    allX, allC, allL, allV, rX, rC, rL, rV = [], [], [], [], [], [], [], []
    for vi_, v in enumerate(views):
        H, W, L, Z = v["H"], v["W"], v["L"], v["Z"]
        st = P["stride"]
        vv, uu = np.mgrid[0:H:st, 0:W:st]
        lab = L[vv, uu]
        m = (lab != 255) & (lab != plate)  # the number plate never enters the model
        e = max(1, P["erode"] // ds)
        for dy, dx in ((e, 0), (-e, 0), (0, e), (0, -e)):
            m &= L[np.clip(vv + dy, 0, H - 1), np.clip(uu + dx, 0, W - 1)] == lab
        uu, vv, lab = uu[m], vv[m], lab[m]
        zz = Z[vv, uu]
        ok = (zz > 0) & np.isfinite(zz)
        uu, vv, lab, zz = uu[ok], vv[ok], lab[ok], zz[ok]
        xy = np.stack([(uu + 0.5) * ds, (vv + 0.5) * ds], 1)
        rays = np.array(v["cam"].cam_from_img(xy))
        Xc = np.concatenate([rays * zz[:, None], zz[:, None]], 1)
        Rt = v["Rt"]
        Xw = (Xc - Rt[:, 3]) @ Rt[:, :3]
        rX.append(Xw); rC.append(v["img"][vv, uu]); rL.append(lab); rV.append(np.full(len(lab), vi_, np.int16))  # noqa: E702
        dist = np.linalg.norm(centers - v["C"], axis=1)
        nb = [j for j in np.argsort(dist) if j != vi_][:P["mv_k"]]
        votes = np.zeros(len(Xw), np.int16)
        for j in nb:  # multi-view depth consistency against the nearest views
            w = views[j]
            Rj = w["Rt"]
            Xcj = Xw @ Rj[:, :3].T + Rj[:, 3]
            front = Xcj[:, 2] > 0.1
            pj = np.full((len(Xw), 2), -1.0)
            if front.any():
                pj[front] = np.array(w["cam"].img_from_cam(Xcj[front])) / ds
            ju, jv = pj[:, 0].astype(int), pj[:, 1].astype(int)
            inside = front & (ju >= 0) & (jv >= 0) & (ju < w["W"]) & (jv < w["H"])
            zj = np.full(len(Xw), np.nan)
            zj[inside] = w["Z"][jv[inside], ju[inside]]
            lj = np.full(len(Xw), 255)
            lj[inside] = w["L"][jv[inside], ju[inside]]
            with np.errstate(invalid="ignore"):
                votes += inside & (np.abs(zj - Xcj[:, 2]) / Xcj[:, 2] < P["mv_tau"]) & (lj == lab)
        keep = votes >= P["mv_min"]
        allX.append(Xw[keep]); allC.append(v["img"][vv[keep], uu[keep]]); allL.append(lab[keep])  # noqa: E702
        allV.append(np.full(int(keep.sum()), vi_, np.int16))

    def vox(X: np.ndarray, C: np.ndarray, L: np.ndarray, V: np.ndarray, min_views: int) -> tuple[np.ndarray, ...]:
        key = np.floor(X / voxel).astype(np.int64)
        key -= key.min(0)
        kk = (key[:, 0] * 1_000_003 + key[:, 1]) * 1_000_003 + key[:, 2]
        _, inv = np.unique(kk, return_inverse=True)
        inv = inv.ravel()
        nvox = int(inv.max()) + 1
        cnt = np.bincount(inv, minlength=nvox)
        lc = np.zeros((nvox, len(base)), np.int32)
        np.add.at(lc, (inv, L), 1)
        lab = lc.argmax(1)
        agree = lc.max(1) / cnt
        pv = np.unique(inv.astype(np.int64) * 1000 + V)
        nviews = np.bincount((pv // 1000).astype(np.int64), minlength=nvox)
        sel = lab[inv] == L
        s = np.maximum(np.bincount(inv[sel], minlength=nvox), 1)
        Xm = np.stack([np.bincount(inv[sel], X[sel, i], nvox) / s for i in range(3)], 1)
        Cm = np.stack([np.bincount(inv[sel], C[sel, i].astype(float), nvox) / s for i in range(3)], 1)
        keepv = (cnt >= 4) & (nviews >= min_views) & (agree >= P["agree"])
        return Xm[keepv], Cm[keepv].astype(np.uint8), lab[keepv], nviews[keepv], agree[keepv]

    # the voxel size is in metres in the original (metric scale); here COLMAP units, so scale it by the car length
    raw = np.concatenate(rX)
    unit = NOMINAL_LENGTH_M / max(float(np.ptp(np.percentile(raw, [1, 99], axis=0), axis=0).max()), 1e-6)
    voxel = P["voxel"] / unit
    X, C, L, NV, AG = vox(np.concatenate(allX), np.concatenate(allC), np.concatenate(allL), np.concatenate(allV),
                          P["min_views"])
    n_strict = len(X)
    if P.get("fill"):
        Xr, Cr, Lr, NVr, AGr = vox(raw, np.concatenate(rC), np.concatenate(rL), np.concatenate(rV), P["fill_views"])
        add = cKDTree(X).query(Xr, k=1)[0] > P["fill_cell"] / unit
        X, C, L = np.concatenate([X, Xr[add]]), np.concatenate([C, Cr[add]]), np.concatenate([L, Lr[add]])
        NV, AG = np.concatenate([NV, NVr[add]]), np.concatenate([AG, AGr[add]])

    # car frame: up = normal of the walkaround plane (flipped so wheels are below the roof), forward = front - back
    cams = np.array([im.projection_center() for im in rec.images.values()])
    up = np.linalg.svd(cams - cams.mean(0))[2][2]
    ids = lambda names: [B[n] for n in names if n in B]  # noqa: E731
    wh, rf = np.isin(L, ids(["front-wheel", "back-wheel"])), L == B.get("roof", -1)
    if wh.any() and rf.any() and (X[wh] @ up).mean() > (X[rf] @ up).mean():
        up = -up
    front = np.isin(L, ids(["hood", "grille", "front-bumper", "headlight", "windshield"]))
    back = np.isin(L, ids(["trunk", "back-bumper", "tail-light", "back-windshield"]))
    if not (front.any() and back.any()):
        raise RuntimeError("cannot orient the car: front or rear parts were not reconstructed")
    up, fwd, left = _frame(up, X[front].mean(0) - X[back].mean(0))
    ctr = X.mean(0)
    SP = np.array([p.xyz for p in rec.points3D.values()])

    def tolocal(Y: np.ndarray) -> np.ndarray:
        return np.stack([(Y - ctr) @ fwd, (Y - ctr) @ left, (Y - ctr) @ up], 1)

    for _ in range(2 if P.get("refine") else 0):
        loc0, sp0 = tolocal(X), tolocal(SP)
        ext = np.percentile(np.abs(loc0[:, :2]), 99, axis=0)
        zlow = np.percentile(loc0[:, 2], 2)
        gm = P["ground_margin"] / unit
        g = sp0[(np.abs(sp0[:, 0]) < ext[0] + gm) & (np.abs(sp0[:, 1]) < ext[1] + gm)
                & ~((np.abs(sp0[:, 0]) < ext[0]) & (np.abs(sp0[:, 1]) < ext[1]))
                & (np.abs(sp0[:, 2] - zlow) < 0.5 * ext[1])]
        if len(g) > 100:  # up from a RANSAC plane through sparse ground points around the car
            rng = np.random.default_rng(1)
            best_g = None
            tol = 0.01 * ext[0]
            for _ in range(500):
                a, b, c = g[rng.choice(len(g), 3, replace=False)]
                nrm = np.cross(b - a, c - a)
                if np.linalg.norm(nrm) < 1e-9:
                    continue
                nrm /= np.linalg.norm(nrm)
                inl = np.abs((g - a) @ nrm) < tol
                if best_g is None or inl.sum() > best_g.sum():
                    best_g = inl
            gi = g[best_g]
            nl = np.linalg.svd(gi - gi.mean(0))[2][2]
            if nl[2] < 0:
                nl = -nl
            up_new = nl[0] * fwd + nl[1] * left + nl[2] * up
            up, fwd, left = _frame(up_new / np.linalg.norm(up_new), fwd)
        loc0 = tolocal(X)  # yaw and lateral offset from the left/right symmetry of the side panels
        sidek = np.isin(L, ids(["front-door", "back-door", "fender", "quarter-panel", "rocker-panel", "front-wheel",
                                "back-wheel"]))
        xs, cs = [], []
        ex_ = np.percentile(np.abs(loc0[:, 0]), 99)
        for x0 in np.linspace(-0.6 * ex_, 0.6 * ex_, 13):
            m = sidek & (np.abs(loc0[:, 0] - x0) < 0.05 * ex_)
            yl, yr = loc0[m & (loc0[:, 1] > 0), 1], loc0[m & (loc0[:, 1] < 0), 1]
            if len(yl) > 30 and len(yr) > 30:
                xs.append(x0)
                cs.append((np.percentile(yl, 90) + np.percentile(yr, 10)) / 2)
        if len(xs) >= 4:
            b_, a_ = np.polyfit(xs, cs, 1)
            th = np.arctan(b_)
            up, fwd, left = _frame(up, np.cos(th) * fwd + np.sin(th) * left)
            ctr = ctr + a_ * left
    loc = tolocal(X)
    lo, hi = np.percentile(loc[:, 0], [0.5, 99.5])
    loc[:, 0] -= (lo + hi) / 2
    sp = tolocal(SP)
    sp[:, 0] -= (lo + hi) / 2
    return dict(X=loc, C=C, L=L, NV=NV, AG=AG, sparse=sp, length_units=float(hi - lo), n_views=len(views),
                n_strict=int(n_strict), align_err=float(np.median([v["relerr"] for v in views])))


def to_metres(f: dict[str, Any], base: list[str], length_m: float) -> dict[str, Any]:
    """Scale so the car's length is length_m, put the ground at z = 0, and measure sizes and noise."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    scale = length_m / f["length_units"]
    X, sp, L = f["X"] * scale, f["sparse"] * scale, f["L"]
    B = {n: i for i, n in enumerate(base)}
    ring = ((np.abs(sp[:, 0]) < 3.5) & (np.abs(sp[:, 1]) < 2.0) & ~((np.abs(sp[:, 0]) < 2.5) & (np.abs(sp[:, 1]) < 0.95))
            & (sp[:, 2] < np.percentile(X[:, 2], 5) + 0.3))
    X[:, 2] -= float(np.median(sp[ring, 2])) if ring.sum() > 50 else float(np.percentile(X[:, 2], 0.5))

    def cen(name: str) -> np.ndarray | None:
        m = L == B.get(name, -1)
        return X[m].mean(0) if m.sum() else None

    out: dict[str, Any] = {}
    fw, bw = cen("front-wheel"), cen("back-wheel")
    if fw is not None and bw is not None:
        out["wheelbase"] = round(float(fw[0] - bw[0]), 3)
    side = np.isin(L, [B[b] for b in ("front-door", "back-door", "fender", "quarter-panel") if b in B])
    if (side & (X[:, 1] > 0)).sum() > 50 and (side & (X[:, 1] < 0)).sum() > 50:
        out["width"] = round(float(np.percentile(X[side & (X[:, 1] > 0), 1], 95)
                                   - np.percentile(X[side & (X[:, 1] < 0), 1], 5)), 3)
    roof = L == B.get("roof", -1)
    if roof.sum() > 50:
        out["height"] = round(float(np.percentile(X[roof, 2], 99)), 3)
    th = []  # panel thickness: RMS distance to the local plane in 15 cm neighbourhoods (noise across flat panels)
    for nm in ("hood", "roof", "front-door", "back-door", "trunk"):
        Q = X[L == B.get(nm, -1)]
        if len(Q) < 200:
            continue
        tr = cKDTree(Q)
        for i in np.random.default_rng(0).choice(len(Q), min(150, len(Q)), replace=False):
            nb = tr.query_ball_point(Q[i], 0.075)
            if len(nb) >= 15:
                A = Q[nb] - Q[nb].mean(0)
                th.append(np.linalg.svd(A, compute_uv=False)[-1] / np.sqrt(len(nb)))
    if th:
        out["thickness_mm"] = round(float(np.median(th) * 1000), 1)
    return dict(f, X=X, scale=scale, measured=out)


def _largest_cluster(X: np.ndarray, g: np.ndarray, cell: float = 0.06) -> np.ndarray:
    from scipy import ndimage  # noqa: PLC0415

    idx = np.where(g)[0]
    k = np.floor((X[idx] - X[idx].min(0)) / cell).astype(int)
    grid = np.zeros(k.max(0) + 1, bool)
    grid[tuple(k.T)] = True
    lab, n = ndimage.label(ndimage.binary_dilation(grid, iterations=1))
    if n <= 1:
        return g
    pl = lab[tuple(k.T)]
    out = np.zeros_like(g)
    out[idx[pl == np.bincount(pl)[1:].argmax() + 1]] = True
    return out


def split_parts(X: np.ndarray, L: np.ndarray, NV: np.ndarray, base: list[str], *, vehicle: str | None,
                metric: bool) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Named parts (paired ones split into left / right by the car's axes) and a part id per point (-1 = dropped)."""
    parts: list[dict[str, Any]] = []
    pid = np.full(len(X), -1)
    for i, b in enumerate(base):
        m = L == i
        if b == "license-plate" or b not in PRETTY or m.sum() < 150:
            continue
        groups = [("", m)] if b not in PAIRED else [("Left", m & (X[:, 1] > PAIRED[b])),
                                                    ("Right", m & (X[:, 1] < -PAIRED[b]))]
        for side, g in groups:
            if g.sum() < 150:
                continue
            g = _largest_cluster(X, g)
            P = X[g]
            lo, hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)
            name = f"{side} {PRETTY[b].lower()}" if side else PRETTY[b]
            q = " ".join(x for x in (vehicle, side.lower(), PRETTY[b].lower()) if x)
            parts.append({"id": len(parts), "key": (side.lower() + "_" if side else "") + b, "name": name,
                          "side": side.lower() or "center", "points": int(g.sum()), "views": int(np.median(NV[g])),
                          "size_m": [round(float(v), 2) for v in hi - lo] if metric else None,
                          "center_m": [round(float(v), 2) for v in P.mean(0)] if metric else None,
                          "query": q,
                          "search": {"google": "https://www.google.com/search?q=" + quote_plus(q + " part"),
                                     "olx_uz": "https://www.olx.uz/list/q-" + quote_plus(q).replace("+", "-") + "/"}})
            pid[g] = parts[-1]["id"]
    return parts, pid


def viewer_payload(X: np.ndarray, C: np.ndarray, pid: np.ndarray, meta: dict[str, Any],
                   max_points: int = VIEWER_MAX_POINTS) -> dict[str, Any]:
    """Compact payload: int16 positions (x fwd, y left, z up) * S/32767, RGB, part id per point; base64."""
    keep = pid >= 0
    X, C, P = X[keep], C[keep], pid[keep]
    s = np.random.default_rng(0).permutation(len(X))[:max_points]
    Xs, Cs, Ps = X[s], C[s], P[s]
    S = float(np.abs(Xs).max())
    q = np.round(Xs / S * 32767).astype("<i2")
    return {"S": S, "n": int(len(s)), "p": base64.b64encode(q.tobytes()).decode(),
            "c": base64.b64encode(Cs.astype(np.uint8).tobytes()).decode(),
            "l": base64.b64encode(Ps.astype(np.uint8).tobytes()).decode(), "meta": meta}


def build(dataset: Path, labels_dir: Path, out: Path, *, device: str = "cuda", vehicle: str | None = None,
          length_m: float | None = None, length_source: str | None = None,
          published: dict[str, float] | None = None) -> dict[str, Any]:
    import pycolmap  # noqa: PLC0415

    t0 = time.perf_counter()
    secs: dict[str, float] = {}
    rec = pycolmap.Reconstruction(str(dataset / "sparse" / "0"))
    labels, base, _ = load_labels(labels_dir)
    images = [dataset / "images" / im.name for im in rec.images.values() if Path(im.name).stem in labels]
    t = time.perf_counter()
    depth = estimate_depth(images, device)
    secs["depth"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    f = fuse(rec, dataset / "images", labels, depth, base)
    secs["fuse"] = round(time.perf_counter() - t, 1)
    metric = bool(length_m)
    f = to_metres(f, base, float(length_m) if metric else NOMINAL_LENGTH_M)
    parts, pid = split_parts(f["X"], f["L"], f["NV"], base, vehicle=vehicle, metric=metric)
    measured = f["measured"] if metric else {}
    checks = [{"name": nm, "scan_m": measured[k], "published_m": float(published[k])}
              for k, nm in (("wheelbase", "Wheelbase"), ("width", "Body width (no mirrors)"), ("height", "Height"))
              if metric and published and k in published and k in measured]
    meta = {"vehicle": vehicle, "metric": metric, "units": "metres" if metric else "relative",
            "scale_from": (f"published length {length_m:.3f} m" + (f" ({length_source})" if length_source else ""))
            if metric else None,
            "source": f"{f['n_views']} anonymised frames, COLMAP poses",
            "checks": checks, "surface_thickness_mm": measured.get("thickness_mm"), "parts": parts,
            "method": "Camry Parts Scan method (Team Trumen): Depth Anything V2 Small aligned to COLMAP, "
                      "multi-view checked, voxel-fused, part labels from the parts model"}
    payload = viewer_payload(f["X"], f["C"], pid, meta)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    secs["total"] = round(time.perf_counter() - t0, 1)
    report = {"ok": True, "views": f["n_views"], "align_err": round(f["align_err"], 4), "points": int((pid >= 0).sum()),
              "strict_points": f["n_strict"], "viewer_points": payload["n"], "parts": len(parts), "metric": metric,
              "measured": measured, "checks": checks, "seconds": secs, "depth_model": DEPTH_MODEL,
              "depth_revision": DEPTH_REVISION, "params": PARAMS}
    (out.parent / "parts_cloud_report.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="recon3d_hq dir with images/ and sparse/0")
    ap.add_argument("--labels", type=Path, required=True, help="part label PNGs + classes.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vehicle", default=None)
    ap.add_argument("--length-m", type=float, default=None)
    ap.add_argument("--length-source", default=None)
    ap.add_argument("--published", default=None, help='JSON {"wheelbase":..,"width":..,"height":..} in metres')
    a = ap.parse_args(argv)
    try:
        rep = build(a.dataset, a.labels, a.out, device=a.device, vehicle=a.vehicle, length_m=a.length_m,
                    length_source=a.length_source, published=json.loads(a.published) if a.published else None)
    except Exception as exc:  # noqa: BLE001 - reported as JSON to the parent process
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps({k: rep[k] for k in ("ok", "views", "points", "parts", "metric", "seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
