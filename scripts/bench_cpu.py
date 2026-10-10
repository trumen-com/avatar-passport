"""CPU latency benchmark for one ONNX model. Run it as its own process so thread limits and taskset pinning apply.

Two measurement kinds, one CSV row each, appended to --out (header created if missing):
  raw-ort          onnxruntime session.run on a letterboxed float32 NCHW input -> infer-only ms
                   (wall_median_ms = letterbox + session.run)
  e2e-ultralytics  YOLO(onnx, task).predict(frame) -> preprocess + inference + postprocess from r.speed
                   (infer_median_ms = r.speed["inference"], wall_median_ms = wall time of predict())
Protocol per kind: one untimed setup call (session/model load + first run, reported as load_ms in the
protocol string), --warmup untimed runs, then up to --runs timed runs cycling the frames in name order.
The timed phase stops early once it has taken more than --cap-s seconds (cap_hit; --cap-s <= 0 = no cap).
p90 uses linear interpolation between closest ranks (numpy's default).

Threads: OMP/OPENBLAS/MKL_NUM_THREADS are exported right after argument parsing and BEFORE numpy, cv2,
onnxruntime, torch or ultralytics are imported (all are imported lazily, inside functions). ONNX Runtime
gets intra_op=T, inter_op=1 explicitly, including the session ultralytics creates internally (it passes
no SessionOptions, which would otherwise mean ORT's default pool size). cv2.setNumThreads(T) is applied
for raw-ort; ultralytics' import sets cv2.setNumThreads(0) (single-threaded) and its CPU select_device()
resets torch to min(8, os.cpu_count() - 1) threads, so torch.set_num_threads(T) is re-applied after the
setup call. The thread counts in force during the timed phase, the CPU affinity and whether the
affinity is really restricted (--pinned is verified) are recorded in the row.

Never crashes: a failure becomes a row with the `error` column filled and the exit code is 0
(exit 1 only if the CSV itself cannot be written).

Usage (the notebook adds `taskset -c 0,8` in front for the pinned 2-thread tier):
    python scripts/bench_cpu.py --onnx models/onnx/damage_mitbersh_896.onnx --model-key damage_mitbersh \\
        --imgsz 896 --task segment --frames-dir outputs/smoke/R2/latency_frames --threads 2 --warmup 3 \\
        --runs 20 --cap-s 120 --run R2 --pinned 1 --out outputs/smoke/R2/latency.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

# Ultralytics reads this at import time; it is never imported before this line (only lazily, below).
os.environ["YOLO_AUTOINSTALL"] = "false"

FIELDS: tuple[str, ...] = (
    "run", "tier", "model", "imgsz", "device", "threads", "pinned", "n_warmup", "n_runs",
    "median_ms", "p90_ms", "infer_median_ms", "wall_median_ms", "cap_hit", "loadavg_1m", "protocol", "error",
)
KINDS: tuple[str, ...] = ("raw-ort", "e2e-ultralytics")
THREAD_ENV: tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAD_VALUE = 114  # ultralytics letterbox grey
MAX_ERROR_CHARS = 400


@dataclass
class Measurement:
    """Timing samples of one kind. total = headline ms per run (raw-ort: session.run; e2e: pre+inf+post)."""

    kind: str
    total_ms: list[float] = field(default_factory=list)
    infer_ms: list[float] = field(default_factory=list)
    wall_ms: list[float] = field(default_factory=list)
    n_warmup: int = 0
    cap_hit: bool = False
    loadavg_1m: float | None = None
    notes: list[str] = field(default_factory=list)
    error: str = ""
    load_failed: bool = False   # model unusable in onnxruntime (load error or input shape != imgsz)


# --------------------------------------------------------------------------- environment

def export_thread_env(threads: int) -> list[str]:
    """Export the BLAS/OpenMP thread limits. Returns heavy modules that were already imported (should be [])."""
    for name in THREAD_ENV:
        os.environ[name] = str(threads)
    return sorted(m for m in ("numpy", "cv2", "onnxruntime", "torch", "ultralytics") if m in sys.modules)


def _cpu_ranges(cpus: Sequence[int]) -> str:
    out: list[str] = []
    start = prev = None
    for c in sorted(cpus):
        if start is None:
            start = prev = c
        elif c == prev + 1:
            prev = c
        else:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = c
    if start is not None:
        out.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(out)


def affinity_desc() -> str:
    """CPU affinity of this process and how many physical cores it spans (Linux sysfs), e.g. '0,8 (1 core)'."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return "unknown"
    cores: set[tuple[str, str]] = set()
    for c in cpus:
        topo = Path(f"/sys/devices/system/cpu/cpu{c}/topology")
        try:
            cores.add(((topo / "physical_package_id").read_text().strip(), (topo / "core_id").read_text().strip()))
        except OSError:
            return _cpu_ranges(cpus)
    return f"{_cpu_ranges(cpus)} ({len(cores)} physical core{'s' if len(cores) != 1 else ''})"


def affinity_restricted() -> bool | None:
    """True if this process may run on fewer CPUs than the machine has (e.g. under taskset); None if unknown."""
    try:
        return len(os.sched_getaffinity(0)) < (os.cpu_count() or 0)
    except (AttributeError, OSError):
        return None


def loadavg_1m() -> float | None:
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return None


def _one_line(exc: BaseException) -> str:
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return text if len(text) <= MAX_ERROR_CHARS else text[: MAX_ERROR_CHARS - 1] + "…"


# --------------------------------------------------------------------------- inputs

def load_frames(frames_dir: str | Path) -> list[Any]:
    """All images in frames_dir (sorted by name) as uint8 HxWx3 BGR. Raises if there are none."""
    import cv2

    d = Path(frames_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"frames dir not found: {d}")
    frames = []
    for p in sorted(x for x in d.iterdir() if x.suffix.lower() in FRAME_EXTS):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    if not frames:
        raise ValueError(f"no readable images in {d}")
    return frames


def letterbox(img: Any, size: int) -> Any:
    """Resize keeping aspect ratio (INTER_LINEAR) and pad to size x size with grey 114, centred (as ultralytics)."""
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)


def to_nchw(img: Any, dtype: Any) -> Any:
    """BGR uint8 HxWx3 -> RGB 1x3xHxW in [0, 1]."""
    import numpy as np

    x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)).astype(dtype) / dtype(255)
    return x[None]


# --------------------------------------------------------------------------- protocol

def percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default method)."""
    s = sorted(xs)
    if not s:
        raise ValueError("no samples")
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def run_protocol(
    step: Callable[[int], tuple[float, float, float]], *, n_frames: int, warmup: int, runs: int, cap_s: float,
    meas: Measurement,
) -> None:
    """Warm-ups (untimed), then up to `runs` timed calls of step(frame_index); stop once the timed phase > cap_s.
    The 1-minute load average is sampled right before the timed phase."""
    for i in range(warmup):
        step(i % n_frames)
        meas.n_warmup += 1
    meas.loadavg_1m = loadavg_1m()
    t_start = time.perf_counter()
    for j in range(runs):
        total, infer, wall = step(j % n_frames)
        meas.total_ms.append(total)
        meas.infer_ms.append(infer)
        meas.wall_ms.append(wall)
        if cap_s > 0 and time.perf_counter() - t_start > cap_s and j + 1 < runs:
            meas.cap_hit = True
            break


def make_session_options(ort: Any, threads: int) -> Any:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return so


def limit_torch_threads(torch: Any, threads: int) -> str:
    """torch intra-op = threads (inter-op = 1 if still settable); returns the counts now in force."""
    torch.set_num_threads(threads)
    with contextlib.suppress(RuntimeError):  # inter-op can only be set before torch's first parallel work
        torch.set_num_interop_threads(1)
    return f"torch_threads={torch.get_num_threads()} torch_interop={torch.get_num_interop_threads()}"


@contextlib.contextmanager
def ort_threads_injected(ort: Any, threads: int) -> Iterator[None]:
    """Sessions created without SessionOptions (as ultralytics does) get intra_op=threads, inter_op=1."""
    original = ort.InferenceSession

    def factory(path_or_bytes: Any, sess_options: Any = None, *args: Any, **kwargs: Any) -> Any:
        if sess_options is None:
            sess_options = make_session_options(ort, threads)
        return original(path_or_bytes, sess_options, *args, **kwargs)

    ort.InferenceSession = factory
    try:
        yield
    finally:
        ort.InferenceSession = original


def bench_raw_ort(
    onnx_path: str | Path, frames: Sequence[Any], *, imgsz: int, threads: int, warmup: int, runs: int, cap_s: float,
) -> Measurement:
    """raw-ort kind: session.run only, on letterboxed inputs built from `frames`. Never raises."""
    meas = Measurement("raw-ort")
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort

        t0 = time.perf_counter()
        try:
            sess = ort.InferenceSession(str(onnx_path), make_session_options(ort, threads),
                                        providers=["CPUExecutionProvider"])
        except Exception:
            meas.load_failed = True
            raise
        inp = sess.get_inputs()[0]
        dims = list(inp.shape)
        if len(dims) == 4 and all(isinstance(d, int) for d in dims[2:]) and tuple(dims[2:]) != (imgsz, imgsz):
            meas.load_failed = True
            raise ValueError(f"model input is {dims[2]}x{dims[3]} but --imgsz is {imgsz}")
        dtype = np.float16 if inp.type == "tensor(float16)" else np.float32

        def step(idx: int) -> tuple[float, float, float]:
            a = time.perf_counter()
            x = to_nchw(letterbox(frames[idx], imgsz), dtype)
            b = time.perf_counter()
            sess.run(None, {inp.name: x})
            c = time.perf_counter()
            return (c - b) * 1e3, (c - b) * 1e3, (c - a) * 1e3

        step(0)  # setup call: first run allocates buffers
        so = sess.get_session_options()
        meas.notes += [
            f"load_ms={(time.perf_counter() - t0) * 1e3:.1f} (session + first run, untimed)",
            f"input={'x'.join(str(d) for d in dims)} {inp.type}",
            f"ort={ort.__version__} intra_op={so.intra_op_num_threads} inter_op={so.inter_op_num_threads}",
            f"providers={','.join(sess.get_providers())}",
            f"cv2_threads={cv2.getNumThreads()}",
            "median/p90=session.run only; wall=letterbox+session.run",
        ]
        if not all(isinstance(d, int) for d in dims[2:]):
            meas.notes.append(f"dynamic input: timed at a square {imgsz}x{imgsz} letterbox "
                              "(ultralytics pads dynamic models only to the stride, so e2e may use a smaller input)")
        run_protocol(step, n_frames=len(frames), warmup=warmup, runs=runs, cap_s=cap_s, meas=meas)
    except Exception as exc:  # recorded in the row, never raised
        meas.error = _one_line(exc)
    return meas


def bench_e2e_ultralytics(
    onnx_path: str | Path, frames: Sequence[Any], *, imgsz: int, task: str, threads: int, warmup: int, runs: int,
    cap_s: float,
) -> Measurement:
    """e2e-ultralytics kind: YOLO(onnx).predict per frame, ms from r.speed plus wall time. Never raises."""
    meas = Measurement("e2e-ultralytics")
    try:
        try:
            import torch
        except ImportError:
            torch = None
            meas.notes.append("torch not importable")
        else:
            limit_torch_threads(torch, threads)
        import cv2
        import onnxruntime as ort
        import ultralytics
        from ultralytics import YOLO

        def step(idx: int) -> tuple[float, float, float]:
            a = time.perf_counter()
            res = model.predict(frames[idx], imgsz=imgsz, device="cpu", verbose=False, retina_masks=False)
            wall = (time.perf_counter() - a) * 1e3
            sp = res[0].speed
            return sp["preprocess"] + sp["inference"] + sp["postprocess"], sp["inference"], wall

        with ort_threads_injected(ort, threads):
            t0 = time.perf_counter()
            model = YOLO(str(onnx_path), task=task)
            step(0)  # setup call: predictor + session creation + ultralytics' own warm-up
            load_ms = (time.perf_counter() - t0) * 1e3
            # select_device() in the setup call reset torch to min(8, cpu_count - 1) threads: re-apply T
            torch_note = (f"torch={torch.__version__} {limit_torch_threads(torch, threads)} (re-applied after setup)"
                          if torch is not None else "torch not importable")
            session = getattr(getattr(getattr(getattr(model, "predictor", None), "model", None), "backend", None),
                              "session", None)
            intra = session.get_session_options().intra_op_num_threads if session is not None else None
            meas.notes += [
                f"load_ms={load_ms:.1f} (model + predictor setup + first predict, untimed)",
                f"ultralytics={ultralytics.__version__} task={task} retina_masks=False conf=ultralytics default",
                f"ort={ort.__version__} ort_intra_op={intra if intra is not None else 'unverified'} (injected)",
                torch_note,
                f"cv2_threads={cv2.getNumThreads()} (ultralytics sets cv2.setNumThreads(0) at import)",
                "median/p90=preprocess+inference+postprocess (r.speed); infer=r.speed inference; wall=predict()",
            ]
            run_protocol(step, n_frames=len(frames), warmup=warmup, runs=runs, cap_s=cap_s, meas=meas)
    except (Exception, SystemExit) as exc:  # recorded in the row, never raised
        meas.error = _one_line(exc)
    return meas


# --------------------------------------------------------------------------- CSV

def _f(x: float | None, nd: int = 3) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def make_row(
    meas: Measurement, *, run: str, model: str, imgsz: int, threads: int, pinned: bool, warmup: int, runs: int,
    cap_s: float, n_frames: int, extra: Sequence[str] = (),
) -> dict[str, str]:
    """One CSV row. Timing fields stay empty when the measurement failed (no partial numbers).
    `pinned` is recorded as True only if the CPU affinity really is restricted (the caller's claim is checked)."""
    ok = not meas.error and bool(meas.total_ms)
    restricted = affinity_restricted()
    pin_note = ([f"--pinned 1 but the CPU affinity is not restricted ({affinity_desc()}): recorded as pinned=False"]
                if pinned and restricted is False else [])
    protocol = "; ".join([
        meas.kind, f"warmup={warmup}", f"runs<={runs}", f"cap_s={cap_s:g} (timed phase)", f"frames={n_frames}",
        f"imgsz={imgsz}", f"threads={threads}", f"affinity={affinity_desc()}", *pin_note, *extra, *meas.notes,
    ])
    return {
        "run": run,
        "tier": f"cpu_ort{threads}",
        "model": model,
        "imgsz": str(imgsz),
        "device": "cpu",
        "threads": str(threads),
        "pinned": str(bool(pinned) and restricted is not False),
        "n_warmup": str(meas.n_warmup),
        "n_runs": str(len(meas.total_ms)),
        "median_ms": _f(statistics.median(meas.total_ms)) if ok else "",
        "p90_ms": _f(percentile(meas.total_ms, 0.9)) if ok else "",
        "infer_median_ms": _f(statistics.median(meas.infer_ms)) if ok else "",
        "wall_median_ms": _f(statistics.median(meas.wall_ms)) if ok else "",
        "cap_hit": str(meas.cap_hit),
        "loadavg_1m": _f(meas.loadavg_1m, 2),
        "protocol": protocol,
        "error": meas.error or ("" if ok else "no timed runs completed"),
    }


def append_rows(path: str | Path, rows: Sequence[dict[str, str]]) -> None:
    """Append rows, creating the header if the file is new. An existing header that lacks some of FIELDS
    (e.g. the notebook's rows without `error`) is upgraded in place: missing columns are added at the end and
    old rows keep their values."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header: list[str] | None = None
    if p.exists() and p.stat().st_size > 0:
        with p.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None) or None
    if header is None:
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(FIELDS), restval="", extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return
    missing = [c for c in FIELDS if c not in header]
    if missing:
        with p.open(newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
        header = header + missing
        tmp = p.with_name(p.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header, restval="", extrasaction="ignore")
            w.writeheader()
            w.writerows(old)
        tmp.replace(p)
    with p.open("rb") as f:
        f.seek(-1, os.SEEK_END)
        needs_newline = f.read(1) not in (b"\n", b"\r")
    with p.open("a", newline="", encoding="utf-8") as f:
        if needs_newline:
            f.write("\r\n")
        csv.DictWriter(f, fieldnames=header, restval="", extrasaction="ignore").writerows(rows)


# --------------------------------------------------------------------------- CLI

def _pos_int(minimum: int) -> Callable[[str], int]:
    def parse(s: str) -> int:
        v = int(s)
        if v < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}")
        return v
    return parse


def _flag(s: str) -> bool:
    v = s.strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected 0 or 1")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="ONNX model file")
    ap.add_argument("--model-key", required=True, help="model key from scripts/download_models.py")
    ap.add_argument("--imgsz", required=True, type=_pos_int(1))
    ap.add_argument("--task", required=True, choices=("segment", "detect"))
    ap.add_argument("--frames-dir", required=True, help="directory of already-anonymised frames (jpg/png)")
    ap.add_argument("--threads", required=True, type=_pos_int(1))
    ap.add_argument("--warmup", required=True, type=_pos_int(0))
    ap.add_argument("--runs", required=True, type=_pos_int(1))
    ap.add_argument("--cap-s", required=True, type=float, help="cap on the timed phase in seconds (<= 0: none)")
    ap.add_argument("--run", required=True, help="run id, e.g. R2")
    ap.add_argument("--pinned", required=True, type=_flag,
                    help="1 if the caller pinned this process (taskset); checked against the CPU affinity")
    ap.add_argument("--out", required=True, help="latency CSV to append to")
    ap.add_argument("--kinds", default=",".join(KINDS), help=f"comma-separated subset of {','.join(KINDS)}")
    args = ap.parse_args(argv)
    wanted = {k.strip() for k in args.kinds.split(",") if k.strip()}
    if not wanted or wanted - set(KINDS):
        ap.error(f"--kinds: expected a subset of {','.join(KINDS)}, got {args.kinds!r}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    late = export_thread_env(args.threads)  # before any numpy / cv2 / onnxruntime / torch / ultralytics import
    os.environ.setdefault("YOLO_OFFLINE", "1")  # no ultralytics update checks or network calls while timing
    kinds = [k for k in KINDS if k in {s.strip() for s in args.kinds.split(",")}]
    extra = [f"thread env exported late (already imported: {','.join(late)})"] if late else []
    results: list[Measurement] = []
    n_frames = 0
    try:
        import cv2

        cv2.setNumThreads(args.threads)
        if not Path(args.onnx).is_file():
            raise FileNotFoundError(f"onnx file not found: {args.onnx}")
        frames = load_frames(args.frames_dir)
        n_frames = len(frames)
    except Exception as exc:
        results = [Measurement(k, error=_one_line(exc)) for k in kinds]
    else:
        common = dict(imgsz=args.imgsz, threads=args.threads, warmup=args.warmup, runs=args.runs, cap_s=args.cap_s)
        raw: Measurement | None = None
        if "raw-ort" in kinds:
            raw = bench_raw_ort(args.onnx, frames, **common)
            results.append(raw)
        if "e2e-ultralytics" in kinds:
            if raw is not None and raw.load_failed:
                results.append(Measurement("e2e-ultralytics", error="skipped: onnxruntime could not load the model "
                                           "or its input shape does not match --imgsz (see the raw-ort row)"))
            else:
                results.append(bench_e2e_ultralytics(args.onnx, frames, task=args.task, **common))
    rows = [
        make_row(m, run=args.run, model=args.model_key, imgsz=args.imgsz, threads=args.threads, pinned=args.pinned,
                 warmup=args.warmup, runs=args.runs, cap_s=args.cap_s, n_frames=n_frames, extra=extra)
        for m in results
    ]
    for r in rows:
        status = f"ERROR {r['error']}" if r["error"] else (
            f"median {r['median_ms']} ms, p90 {r['p90_ms']} ms, n={r['n_runs']}, cap_hit={r['cap_hit']}")
        print(f"[bench_cpu] {r['run']} {r['tier']} {r['model']}@{r['imgsz']} {r['protocol'].split(';')[0]}: {status}")
    try:
        append_rows(args.out, rows)
    except (OSError, csv.Error, UnicodeError, ValueError) as exc:
        print(f"[bench_cpu] could not write {args.out}: {_one_line(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
