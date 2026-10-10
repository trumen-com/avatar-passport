"""2D damage boxes -> 3D pins in the DA3 point-cloud frame (main venv: numpy + stdlib only).

Reads the worker outputs (cameras.json, depth.npz, points.ply). A box on an original frame is mapped to DA3's
processed resolution exactly as DA3's `upper_bound_resize` does it (longest side -> process_res, then each side
rounded to the nearest multiple of 14 by a plain resize, then a centre crop if the batch had mixed sizes), the
confident depth pixels inside it are back-projected (world = R^T (K^-1 [u+0.5, v+0.5, 1] z - t)) and the median is
the pin. Pins with the same (part, damage) within merge_radius are merged across frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PATCH = 14
_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
              "char": "i1", "int8": "i1", "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
              "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}


def da3_resized_wh(w: int, h: int, process_res: int) -> tuple[int, int]:
    """Size DA3 gives an image of size (w, h) with upper_bound_resize, before any batch centre crop."""
    if max(w, h) != process_res:
        s = process_res / float(max(w, h))
        w, h = max(1, int(round(w * s))), max(1, int(round(h * s)))

    def nearest(x: int) -> int:
        down = (x // PATCH) * PATCH
        up = down + PATCH
        return up if abs(up - x) <= abs(x - down) else down

    return max(1, nearest(w)), max(1, nearest(h))


def box_to_processed(box: list[float], frame_size: tuple[int, int], process_res: int,
                     processed_hw: tuple[int, int]) -> tuple[float, float, float, float]:
    """[x1, y1, x2, y2] in original pixels (frame_size = (w, h)) -> continuous coords in the processed image."""
    w, h = frame_size
    rw, rh = da3_resized_wh(w, h, process_res)
    ph, pw = processed_hw
    sx, sy = rw / float(w), rh / float(h)
    ox, oy = max(0, (rw - pw) // 2), max(0, (rh - ph) // 2)  # DA3 batch-unify centre crop (0 for same-size frames)
    x1, y1, x2, y2 = (float(v) for v in box)
    return x1 * sx - ox, y1 * sy - oy, x2 * sx - ox, y2 * sy - oy


def read_ply_xyz(path: Path) -> np.ndarray:
    """xyz float64 [N, 3] from a binary little-endian PLY (vertex element first)."""
    with Path(path).open("rb") as f:
        n, props, fmt, in_vertex = 0, [], "", False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"no end_header in {path}")
            tok = line.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    n = int(tok[2])
            elif tok[0] == "property" and in_vertex:
                props.append((tok[-1], _PLY_TYPES[tok[1]]))
            elif tok[0] == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"unsupported PLY format {fmt!r} in {path}")
        rec = np.fromfile(f, dtype=np.dtype(props), count=n)
    return np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)


def bbox_diagonal(pts: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> float:
    """Diagonal of the robust (lo-hi percentile) bounding box; DA3 clouds have a few far outliers."""
    if len(pts) == 0:
        return 0.0
    a, b = np.percentile(pts, lo, axis=0), np.percentile(pts, hi, axis=0)
    return float(np.linalg.norm(b - a))


def _backproject(depth: np.ndarray, K: np.ndarray, w2c: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """World points [M, 3] for mask pixels (same convention as the worker's cloud)."""
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float64)
    pix = np.stack([u + 0.5, v + 0.5, np.ones(len(u))], axis=0)
    cam = (np.linalg.inv(np.asarray(K, dtype=np.float64)) @ pix) * z
    w2c = np.asarray(w2c, dtype=np.float64)
    R, t = w2c[:3, :3], w2c[:3, 3:4]
    return (R.T @ (cam - t)).T


def _box_pixels(bx: tuple[float, float, float, float], hw: tuple[int, int]) -> tuple[slice, slice] | None:
    """Rows/cols whose pixel centres fall inside the processed box (at least the centre pixel); None if outside."""
    h, w = hw
    x1, y1, x2, y2 = min(bx[0], bx[2]), min(bx[1], bx[3]), max(bx[0], bx[2]), max(bx[1], bx[3])
    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return None
    u0, u1 = int(np.ceil(x1 - 0.5)), int(np.floor(x2 - 0.5))
    v0, v1 = int(np.ceil(y1 - 0.5)), int(np.floor(y2 - 0.5))
    if u1 < u0:
        u0 = u1 = int(np.floor((x1 + x2) / 2))
    if v1 < v0:
        v0 = v1 = int(np.floor((y1 + y2) / 2))
    u0, u1 = max(0, u0), min(w - 1, u1)
    v0, v1 = max(0, v0), min(h - 1, v1)
    if u1 < u0 or v1 < v0:
        return None
    return slice(v0, v1 + 1), slice(u0, u1 + 1)


def _merge(pins: list[dict], radius: float) -> list[dict]:
    """Single-linkage merge of pins with the same (part, damage) whose xyz are within radius."""
    parent = list(range(len(pins)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    xyz = np.array([p["xyz"] for p in pins], dtype=np.float64).reshape(-1, 3)
    for i in range(len(pins)):
        for j in range(i + 1, len(pins)):
            same = (pins[i]["part"], pins[i]["damage"]) == (pins[j]["part"], pins[j]["damage"])
            if same and np.linalg.norm(xyz[i] - xyz[j]) <= radius:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(pins)):
        groups.setdefault(find(i), []).append(i)
    merged = []
    for idx in groups.values():
        frames = sorted({f for i in idx for f in pins[i]["frames"]})
        merged.append({
            "xyz": np.median(xyz[idx], axis=0).round(6).tolist(),
            "damage": pins[idx[0]]["damage"], "part": pins[idx[0]]["part"],
            "score": max(float(pins[i]["score"]) for i in idx), "n_views": len(frames), "frames": frames,
        })
    merged.sort(key=lambda p: -p["score"])
    return merged


def project_pins(out_dir: Path, pins2d: list[dict], frame_size: tuple[int, int] | None, *,
                 conf_percentile: float = 40.0, merge_radius: float | None = None) -> list[dict]:
    """2D boxes -> merged 3D pins [{"xyz", "damage", "part", "score", "n_views", "frames"}], best score first.

    pins2d: {"frame": name or path (matched by basename, then stem), "box": [x1, y1, x2, y2] in original frame
    pixels, "damage", "part", "score"}. frame_size = original (w, h); None uses each camera's recorded orig_wh.
    Frames not in the reconstruction and boxes with no valid depth (off the car mask / sky) are skipped.
    merge_radius None = 5% of the robust bounding-box diagonal of points.ply.
    """
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "cameras.json").read_text())
    with np.load(out_dir / "depth.npz", allow_pickle=False) as z:
        depth = z["depth"].astype(np.float32)
        conf = z["conf"].astype(np.float32)
        valid = z["valid"].astype(bool) if "valid" in z.files else np.ones(depth.shape, dtype=bool)
        names = [str(x) for x in z["frames"]]
    valid &= np.isfinite(depth) & (depth > 0)
    hw = tuple(meta.get("processed_hw") or depth.shape[1:])
    process_res = int(meta["process_res"])
    cams = {c["frame"]: c for c in meta["cameras"]}
    thr = float(np.percentile(conf, conf_percentile))
    by_name = {n: i for i, n in enumerate(names)}
    by_stem = {Path(n).stem: i for i, n in enumerate(names)}

    singles: list[dict] = []
    for item in pins2d:
        key = Path(str(item.get("frame", ""))).name
        i = by_name.get(key, by_stem.get(Path(key).stem))
        if i is None or "box" not in item:
            continue
        cam = cams.get(names[i])
        if cam is None:
            continue
        size = frame_size or tuple(cam.get("orig_wh") or ())
        if len(size) != 2:
            continue
        sl = _box_pixels(box_to_processed(item["box"], size, process_res, hw), hw)  # type: ignore[arg-type]
        if sl is None:
            continue
        ok = valid[i][sl] & (conf[i][sl] >= thr)
        if not ok.any():  # nothing confident: fall back to any valid pixel (finite, not sky, on the car mask)
            ok = valid[i][sl]
            if not ok.any():  # box entirely off the car / sky: no pin
                continue
        mask = np.zeros(hw, dtype=bool)
        mask[sl] = ok
        pts = _backproject(depth[i], cam["intrinsics"], cam["w2c"], mask)
        singles.append({"xyz": np.median(pts, axis=0).tolist(), "damage": item.get("damage"),
                        "part": item.get("part"), "score": float(item.get("score") or 0.0), "frames": [names[i]]})
    if not singles:
        return []
    if merge_radius is None:
        ply = out_dir / "points.ply"
        if ply.exists():
            merge_radius = 0.05 * bbox_diagonal(read_ply_xyz(ply))
        else:
            centres = np.array([-np.asarray(c["w2c"])[:3, :3].T @ np.asarray(c["w2c"])[:3, 3]
                                for c in meta["cameras"]])
            merge_radius = 0.05 * bbox_diagonal(centres, 0.0, 100.0)
    return _merge(singles, float(merge_radius))
