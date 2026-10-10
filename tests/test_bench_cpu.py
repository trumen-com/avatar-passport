"""scripts/bench_cpu.py: raw ONNX Runtime path, CSV contract, never-crash behaviour, import order.

Synthetic only: a tiny Conv model built with onnx.helper and generated frames; ultralytics is never run.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "bench_cpu.py"
HEAVY = {"numpy", "cv2", "onnxruntime", "torch", "ultralytics", "onnx"}
NB_COLS = ["run", "tier", "model", "imgsz", "device", "threads", "pinned", "n_warmup", "n_runs", "median_ms", "p90_ms",
           "infer_median_ms", "wall_median_ms", "cap_hit", "loadavg_1m", "protocol"]


def _load():
    spec = importlib.util.spec_from_file_location("bench_cpu", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


bench = _load()


def tiny_onnx(path: Path, size: int = 64) -> Path:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(np.random.default_rng(0).standard_normal((4, 3, 3, 3)).astype(np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["images", "w"], ["output0"], pads=[1, 1, 1, 1])], "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4, size, size])], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def frames_dir(path: Path, n: int = 3) -> Path:
    import cv2

    path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.zeros((120 + 10 * i, 90, 3), np.uint8)
        img[:, :, i % 3] = 200
        cv2.imwrite(str(path / f"f{i:02d}.png"), img)
    return path


def cli(tmp_path: Path, onnx_path: Path, *extra: str) -> tuple[subprocess.CompletedProcess, list[dict[str, str]]]:
    out = tmp_path / "lat.csv"
    cmd = [sys.executable, str(SCRIPT), "--onnx", str(onnx_path), "--model-key", "tiny", "--imgsz", "64",
           "--task", "detect", "--frames-dir", str(frames_dir(tmp_path / "frames")), "--threads", "1",
           "--warmup", "1", "--runs", "3", "--cap-s", "30", "--run", "T1", "--pinned", "0", "--out", str(out), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    rows = list(csv.DictReader(out.open(newline=""))) if out.exists() else []
    return proc, rows


def test_raw_ort_function_and_csv_row(tmp_path):
    model = tiny_onnx(tmp_path / "tiny.onnx")
    import cv2

    frames = [cv2.imread(str(p)) for p in sorted(frames_dir(tmp_path / "frames").iterdir())]
    m = bench.bench_raw_ort(model, frames, imgsz=64, threads=1, warmup=2, runs=4, cap_s=60)
    assert m.error == "" and not m.load_failed
    assert m.n_warmup == 2 and len(m.total_ms) == 4 and not m.cap_hit
    assert all(t > 0 for t in m.total_ms) and all(w >= i for w, i in zip(m.wall_ms, m.infer_ms))
    row = bench.make_row(m, run="T0", model="tiny", imgsz=64, threads=1, pinned=False, warmup=2, runs=4, cap_s=60,
                         n_frames=len(frames))
    out = tmp_path / "sub" / "lat.csv"
    bench.append_rows(out, [row])
    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(bench.FIELDS)
        (got,) = list(reader)
    assert got["run"] == "T0" and got["tier"] == "cpu_ort1" and got["device"] == "cpu" and got["model"] == "tiny"
    assert got["imgsz"] == "64" and got["threads"] == "1" and got["pinned"] == "False" and got["cap_hit"] == "False"
    assert got["n_warmup"] == "2" and got["n_runs"] == "4" and got["error"] == ""
    median, p90 = float(got["median_ms"]), float(got["p90_ms"])
    assert 0 < median <= p90 and float(got["wall_median_ms"]) >= float(got["infer_median_ms"]) > 0
    assert got["protocol"].startswith("raw-ort;") and "intra_op=1 inter_op=1" in got["protocol"]
    assert "input=1x3x64x64" in got["protocol"] and "cv2_threads=" in got["protocol"]
    assert "dynamic input" not in got["protocol"]


def test_pinned_is_recorded_only_if_the_affinity_is_restricted(monkeypatch):
    m = bench.Measurement("raw-ort", total_ms=[1.0], infer_ms=[1.0], wall_ms=[1.0])
    kw = dict(run="T", model="m", imgsz=64, threads=2, warmup=0, runs=1, cap_s=0, n_frames=1)
    for restricted, claimed, expected in ((True, True, "True"), (None, True, "True"), (False, True, "False"),
                                          (True, False, "False")):
        monkeypatch.setattr(bench, "affinity_restricted", lambda r=restricted: r)
        row = bench.make_row(m, pinned=claimed, **kw)
        assert row["pinned"] == expected
        assert ("affinity is not restricted" in row["protocol"]) == (claimed and restricted is False)


def test_torch_threads_are_reapplied_and_reported():
    class FakeTorch:                      # ultralytics' select_device() resets intra-op threads; inter-op is set once
        def __init__(self) -> None:
            self.intra, self.inter, self.inter_locked = 8, 16, False

        def set_num_threads(self, n: int) -> None:
            self.intra = n

        def get_num_threads(self) -> int:
            return self.intra

        def set_num_interop_threads(self, n: int) -> None:
            if self.inter_locked:
                raise RuntimeError("cannot set number of interop threads after parallel work has started")
            self.inter, self.inter_locked = n, True

        def get_num_interop_threads(self) -> int:
            return self.inter

    torch = FakeTorch()
    assert bench.limit_torch_threads(torch, 2) == "torch_threads=2 torch_interop=1"
    torch.set_num_threads(8)              # what select_device() does during the setup call
    assert bench.limit_torch_threads(torch, 2) == "torch_threads=2 torch_interop=1"


def test_cli_raw_ort_only(tmp_path):
    model = tiny_onnx(tmp_path / "tiny.onnx")
    proc, rows = cli(tmp_path, model, "--kinds", "raw-ort")
    assert proc.returncode == 0, proc.stderr
    assert len(rows) == 1 and rows[0]["error"] == "" and rows[0]["n_runs"] == "3"
    assert float(rows[0]["median_ms"]) > 0
    assert "exported late" not in rows[0]["protocol"]          # env set before numpy/cv2/onnxruntime imports
    assert "affinity=" in rows[0]["protocol"]


def test_cli_missing_model_writes_error_rows_and_exits_0(tmp_path):
    proc, rows = cli(tmp_path, tmp_path / "missing.onnx")
    assert proc.returncode == 0, proc.stderr
    assert [r["protocol"].split(";")[0] for r in rows] == ["raw-ort", "e2e-ultralytics"]
    for r in rows:
        assert "not found" in r["error"]
        assert r["median_ms"] == r["p90_ms"] == r["wall_median_ms"] == "" and r["n_runs"] == "0"
        assert r["run"] == "T1" and r["tier"] == "cpu_ort1" and r["model"] == "tiny"


def test_cli_invalid_model_file_is_recorded_and_e2e_skipped(tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"this is not an onnx model")
    proc, rows = cli(tmp_path, bad)
    assert proc.returncode == 0, proc.stderr
    assert len(rows) == 2 and rows[0]["error"] and rows[0]["median_ms"] == ""
    assert rows[1]["error"].startswith("skipped: onnxruntime could not load")


def test_cli_wrong_imgsz_for_static_model_skips_e2e(tmp_path):
    model = tiny_onnx(tmp_path / "tiny.onnx", size=32)
    proc, rows = cli(tmp_path, model)
    assert proc.returncode == 0 and "--imgsz is 64" in rows[0]["error"]
    assert rows[1]["error"].startswith("skipped:") and rows[1]["median_ms"] == ""


def test_cli_rejects_unknown_kind(tmp_path):
    proc, rows = cli(tmp_path, tmp_path / "missing.onnx", "--kinds", "raw-ort,bogus")
    assert proc.returncode == 2 and rows == []


def test_append_upgrades_notebook_header(tmp_path):
    out = tmp_path / "lat.csv"
    with out.open("w", newline="") as f:      # the notebook's in-process rows have no error column
        w = csv.writer(f)
        w.writerow(NB_COLS)
        w.writerow(["R2", "cpu_torch", "m", "896", "cpu", "16", "False", "3", "20", "1.5", "2.5", "1.0", "1.6",
                    "False", "0.1", "e2e-ultralytics; p"])
    row = {k: "" for k in bench.FIELDS} | {"run": "R2", "tier": "cpu_ort2", "error": "boom"}
    bench.append_rows(out, [row])
    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == NB_COLS + ["error"]
    assert rows[0]["median_ms"] == "1.5" and rows[0]["error"] == "" and rows[0]["protocol"] == "e2e-ultralytics; p"
    assert rows[1]["tier"] == "cpu_ort2" and rows[1]["error"] == "boom"


def test_append_after_file_without_trailing_newline(tmp_path):
    out = tmp_path / "lat.csv"
    out.write_text(",".join(bench.FIELDS) + "\n" + ",".join(["x"] * len(bench.FIELDS)))
    bench.append_rows(out, [{k: "y" for k in bench.FIELDS}])
    rows = list(csv.DictReader(out.open(newline="")))
    assert [r["run"] for r in rows] == ["x", "y"]


def test_protocol_cap_and_warmup():
    def slow(_: int) -> tuple[float, float, float]:
        time.sleep(0.02)
        return 20.0, 20.0, 20.0

    m = bench.Measurement("raw-ort")
    bench.run_protocol(slow, n_frames=3, warmup=1, runs=1000, cap_s=0.1, meas=m)
    assert m.cap_hit and m.n_warmup == 1 and 1 <= len(m.total_ms) < 1000
    m2 = bench.Measurement("raw-ort")
    bench.run_protocol(lambda i: (1.0, 1.0, 1.0), n_frames=2, warmup=0, runs=5, cap_s=0, meas=m2)
    assert not m2.cap_hit and len(m2.total_ms) == 5


def test_percentile_matches_numpy():
    xs = [5.0, 1.0, 9.0, 3.0, 7.0, 2.5]
    for q in (0.0, 0.5, 0.9, 1.0):
        assert bench.percentile(xs, q) == pytest.approx(float(np.percentile(xs, q * 100)))
    assert bench.percentile([4.2], 0.9) == 4.2


def test_letterbox_keeps_aspect_and_pads_grey():
    img = np.full((100, 50, 3), 10, np.uint8)
    lb = bench.letterbox(img, 64)
    assert lb.shape == (64, 64, 3)
    assert (lb[:, 0] == 114).all() and (lb[:, -1] == 114).all() and (lb[32, 32] == 10).all()
    x = bench.to_nchw(lb, np.float32)
    assert x.shape == (1, 3, 64, 64) and x.dtype == np.float32 and float(x.max()) <= 1.0


def test_heavy_modules_are_never_imported_at_module_level():
    tree = ast.parse(SCRIPT.read_text())
    top = []
    for node in tree.body:
        for sub in ast.walk(node) if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) else ():
            if isinstance(sub, ast.Import):
                top += [a.name.split(".")[0] for a in sub.names]
            elif isinstance(sub, ast.ImportFrom) and sub.module:
                top.append(sub.module.split(".")[0])
    assert not HEAVY & set(top), sorted(HEAVY & set(top))
