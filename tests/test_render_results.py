"""scripts/render_results.py: rewrites only the marker blocks, deterministically; --check detects stale tables."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "render_results.py"
SMOKE_BLOCKS = ("runs_env", "models", "detections", "privacy", "vin", "latency", "estimate", "artifacts")
SYN_VIN = "0APSYN12345678901"          # synthetic, VIN-shaped (row "0" is unassigned in ISO 3780)
SYN_VIN_MASKED = SYN_VIN[:11] + "******"
NB_COLS = ["run", "tier", "model", "imgsz", "device", "threads", "pinned", "n_warmup", "n_runs", "median_ms", "p90_ms",
           "infer_median_ms", "wall_median_ms", "cap_hit", "loadavg_1m", "protocol"]
BLOCK_RE = re.compile(r"(<!-- BEGIN:(\w+) -->).*?(<!-- END:\2 -->)", re.S)


def _load():
    spec = importlib.util.spec_from_file_location("render_results", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


rr = _load()


def outside(text: str) -> str:
    """The doc with every generated block body removed (what must never change)."""
    return BLOCK_RE.sub(r"\1\3", text)


def bodies(text: str) -> dict[str, str]:
    return {m.group(2): m.group(0) for m in BLOCK_RE.finditer(text)}


def write_csv(path: Path, cols: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def make_tree(root: Path) -> tuple[Path, Path]:
    out = root / "outputs" / "smoke"
    r1 = out / "R2"
    (r1 / "annotated" / "damage_mitbersh").mkdir(parents=True)
    results = {
        "env": {"python": "3.12.13", "cuda": "13.0", "cpu_count": 16, "loadavg": [1.2345, 0.5, 0.25]},
        "models": {
            "damage_mitbersh": {"status": "ok", "task": "segment", "imgsz": 896, "conf": 0.25,
                                "names": {"0": "dent", "1": "scratch", "10": "crack"}, "sha256": "ab" * 32},
            "damage_harpreet": {"status": "load_failed", "error": "RuntimeError: bad | file <x>"},
        },
        "counts": {"frames_kept": 60, "photos": 9},
        "privacy": {"inputs": 69, "ok": 68, "dropped": 1, "n_plates": 31, "n_faces": 2},
        "detections": {"damage_mitbersh": {"inputs": 69, "detections": 45, "per_class": {"scratch": 30, "dent": 15},
                                           "mean_conf": 0.51234, "max_mem_mb": 812.345}},
        "vin": {  # the notebook's layout: summary fields + one list per engine
            "note": "masked", "n_images": 1, "n_read": 1, "n_exact": 1, "ground_truth_file": True, "easyocr": "ok",
            "results": [{"file": "VIN_NUMBER1.jpg", "engine": "rapidocr", "status": "read",
                         "vin_masked": SYN_VIN_MASKED, "exact_match": True, "corrections": ["pos 15: B->8", "pos 3: O->0"],
                         "timings_ms": {"det": 120, "rec": 33.25},
                         "raw_lines": [["SECRETLINE 678901", 0.9]], "decode": {"vin": SYN_VIN, "make": "Synthetic"}}],
            "easyocr_results": [{"file": "VIN_NUMBER1.jpg", "engine": "easyocr", "status": "read",
                                 "vin_masked": SYN_VIN_MASKED, "exact_match": False, "char_errors": 2}],
        },
        "timing": {"extract": {"count": 1, "total_ms": 1234.5678, "median_ms": 1234.5678, "max_ms": 1234.5678}},
        "errors": [{"stage": "latency", "tier": "cpu_ort2", "model": "parts_mitbersh", "error": "RuntimeError: x"}],
    }
    (r1 / "results.json").write_text(json.dumps(results), encoding="utf-8")
    write_csv(r1 / "latency.csv", NB_COLS + ["error"], [
        ["R2", "cpu_ort2", "damage_mitbersh", 896, "cpu", 2, "True", 3, 20, 812.345, 830.0, 812.345, 815.5, "False",
         0.42, "raw-ort; warmup=3; runs<=20", ""],
        ["R2", "cpu_torch", "damage_mitbersh", 896, "cpu", 16, "False", 3, 20, 400.04, 410.0, 380.0, 405.0, "False",
         0.42, "e2e-ultralytics; torch", ""],
        ["R2", "cpu_ort2", "parts_mitbersh", 768, "cpu", 2, "True", 0, 0, "", "", "", "", "False", "",
         "e2e-ultralytics; x", "FileNotFoundError: onnx file not found"],
        ["R2", "cpu_ort16", "parts_mitbersh", 768, "cpu", 16, "False", 0, 0, "", "", "", "", "False", "",
         "onnx export", "RuntimeError: export failed"],
    ])
    (r1 / "estimate.csv").write_text(
        "tier,estimate_ms,steps_included,subtotal_ms,x2 safety factor (assumption)\n"
        "cpu_ort2,123456.789,frames x60,30015.000,60030.000\ncpu_ort2,1,photos x9,1234.568,2469.136\n",
        encoding="utf-8")
    (r1 / "annotated" / "damage_mitbersh" / "a.jpg").write_bytes(b"\xff" * 100)
    (r1 / ".ipynb_checkpoints").mkdir()                          # hidden: not a run output
    (r1 / ".ipynb_checkpoints" / "x.json").write_text("{}", encoding="utf-8")
    r10 = out / "R10"                     # partial run: notebook-style CSV only, broken results.json
    r10.mkdir()
    write_csv(r10 / "latency.csv", NB_COLS, [
        ["R10", "gpu", "damage_mitbersh", 896, "cuda", 16, "False", 3, 20, 12.3456, 14.0, 8.1, 13.2, "False", 0.5,
         "e2e-ultralytics; gpu"],
    ])
    (r10 / "results.json").write_text("{broken", encoding="utf-8")
    (out / "cache").mkdir()               # not a run: no run files
    docs = root / "docs" / "research"
    docs.mkdir(parents=True)
    blocks = "\n\n".join(f"## {n}\n<!-- BEGIN:{n} -->\n_not generated yet_\n<!-- END:{n} -->" for n in SMOKE_BLOCKS)
    (docs / "smoke_test_results.md").write_text(
        f"# Title | with pipe\n\nHand-written intro, 1.23456 stays as typed.\n\n{blocks}\n\n## Observations\nkeep me\n",
        encoding="utf-8")
    (docs / "model_candidates.md").write_text(
        "# Candidates\n\n| a | b |\n|---|---|\n\n<!-- BEGIN:measured_latency -->\n_not measured yet_\n"
        "<!-- END:measured_latency -->\n\ntail\n", encoding="utf-8")
    return out, docs


def run(out: Path, docs: Path, *extra: str) -> int:
    return rr.main(["--outputs-dir", str(out), "--docs", str(docs), *extra])


@pytest.fixture()
def tree(tmp_path: Path) -> tuple[Path, Path]:
    return make_tree(tmp_path)


def test_render_inserts_tables_only_between_markers(tree):
    out, docs = tree
    smoke, cand = docs / "smoke_test_results.md", docs / "model_candidates.md"
    before_s, before_c = smoke.read_text(), cand.read_text()
    assert run(out, docs) == 0
    after_s, after_c = smoke.read_text(), cand.read_text()
    assert outside(after_s) == outside(before_s)
    assert outside(after_c) == outside(before_c)
    new = bodies(after_s)
    assert set(new) == set(SMOKE_BLOCKS)
    for name, body in new.items():
        assert "_not generated yet_" not in body, name
        assert rr.GENERATED_NOTE in body, name
    assert "| --- |" in new["latency"] and "| --- |" in bodies(after_c)["measured_latency"]
    # numbers come from the files, one format per column
    assert "812.3" in new["latency"] and "812.345" not in new["latency"]
    assert "0.512" in new["detections"] and "812.3" in new["detections"]
    assert "123456.8" in new["estimate"]
    assert "1234.6" in new["latency"]                      # StepTimer summary
    assert "| 13.0 |" in new["runs_env"]                   # JSON strings are not re-parsed as numbers
    assert "1.23, 0.50, 0.25" in new["runs_env"]
    # provenance captions name the runs and the source files
    assert "_measured: R2, R10, source_" in new["latency"]
    assert "outputs/smoke/&lt;run&gt;" not in after_s and "/<run>/latency.csv`" in new["latency"]
    # the pinned 2-thread proxy is labelled
    assert rr.HF_PROXY_NOTE in bodies(after_c)["measured_latency"]
    # errors and unreadable files are reported, not fatal
    assert "onnx file not found" in new["latency"]
    assert "results.json: Expecting property name" in new["runs_env"]
    assert "load_failed" in new["models"] and "bad \\| file &lt;x>" in new["models"]
    assert "0 dent, 1 scratch, 10 crack" in new["models"]
    assert "latency: RuntimeError: x (model=parts_mitbersh, tier=cpu_ort2)" in new["runs_env"]
    assert ".ipynb_checkpoints" not in new["artifacts"] and "`annotated/`" in new["artifacts"]


def test_numbers_one_precision_per_column(tree):
    out, docs = tree
    assert run(out, docs) == 0
    new = bodies((docs / "smoke_test_results.md").read_text())
    est = [ln for ln in new["estimate"].splitlines() if ln.startswith("| R2 ")]
    # subtotal_ms and the doubled 'x2 safety factor' column are both ms: 1 decimal, rounded half up
    assert [ln.split("|")[5:7] for ln in est] == [[" 30015.0 ", " 60030.0 "], [" 1234.6 ", " 2469.1 "]]
    assert "| timings_ms.det | 120.0 |" in new["vin"] and "| timings_ms.rec | 33.3 |" in new["vin"]


def test_kind_only_from_a_known_protocol_kind(tree):
    out, docs = tree
    assert run(out, docs) == 0
    lat = bodies((docs / "smoke_test_results.md").read_text())["latency"]
    (row,) = [ln for ln in lat.splitlines() if "| cpu_ort16 |" in ln]
    assert row.split("|")[8].strip() == "" and "export failed" in row      # 'onnx export' is not a kind
    assert "| raw-ort |" in lat and "| e2e-ultralytics |" in lat


def test_vins_masked_and_raw_ocr_dropped(tree):
    out, docs = tree
    assert run(out, docs) == 0
    text = (docs / "smoke_test_results.md").read_text()
    vin = bodies(text)["vin"]
    assert SYN_VIN not in text
    assert f"`{SYN_VIN_MASKED}`" in vin
    assert "SECRETLINE" not in text and "678901" not in text
    # both engines are rendered, the primary one first, plus the run's VIN summary
    assert vin.index("VIN_NUMBER1.jpg (rapidocr)") < vin.index("VIN_NUMBER1.jpg (easyocr)")
    assert "| n_exact | 1 |" in vin and "| char_errors |  | 2 |" in vin
    # an OCR correction after position 11 would reveal a masked character; up to 11 it is shown
    assert "B->8" not in text and "pos 15: *->*" in vin and "pos 3: O->0" in vin


def test_runs_and_models_sorted_naturally(tree):
    out, docs = tree
    assert run(out, docs) == 0
    lat = bodies((docs / "smoke_test_results.md").read_text())["latency"]
    first_table = lat.split("\n\n")[0].splitlines()
    run_cells = [ln.split("|")[1].strip() for ln in first_table if ln.startswith("| R")]
    assert run_cells == ["R2", "R2", "R2", "R2", "R10"]
    models = [ln.split("|")[3].strip() for ln in first_table if ln.startswith("| R2 ")]
    assert models == sorted(models)


def test_check_clean_after_render_and_stale_after_tampering(tree, capsys):
    out, docs = tree
    smoke = docs / "smoke_test_results.md"
    assert run(out, docs, "--check") == 1             # fresh skeleton is stale
    assert smoke.read_text().count("_not generated yet_") == len(SMOKE_BLOCKS)  # --check never writes
    assert run(out, docs) == 0
    rendered = smoke.read_bytes()
    assert run(out, docs, "--check") == 0
    assert run(out, docs) == 0 and smoke.read_bytes() == rendered          # idempotent
    smoke.write_text(smoke.read_text().replace("812.3", "999.9", 1))      # tamper inside a block
    capsys.readouterr()
    assert run(out, docs, "--check") == 1
    assert "STALE" in capsys.readouterr().out
    assert run(out, docs) == 0 and smoke.read_bytes() == rendered          # restored
    smoke.write_text(smoke.read_text().replace("keep me", "edited by hand"))  # outside: not our business
    assert run(out, docs, "--check") == 0
    assert run(out, docs) == 0 and "edited by hand" in smoke.read_text()


def test_new_run_output_makes_check_fail(tree):
    out, docs = tree
    assert run(out, docs) == 0
    with (out / "R10" / "latency.csv").open("a", newline="") as f:
        csv.writer(f).writerow(["R10", "gpu", "parts_mitbersh", 768, "cuda", 16, "False", 3, 20, 5.0, 6.0, 4.0, 5.5,
                                "False", 0.5, "e2e-ultralytics; gpu"])
    assert run(out, docs, "--check") == 1


def test_deterministic_across_copies(tmp_path):
    a_out, a_docs = make_tree(tmp_path / "a")
    b_out, b_docs = make_tree(tmp_path / "b")
    assert run(a_out, a_docs) == 0 and run(b_out, b_docs) == 0
    norm = lambda p, root: p.read_text().replace(str(root), "ROOT")  # noqa: E731
    assert norm(a_docs / "smoke_test_results.md", tmp_path / "a") == norm(b_docs / "smoke_test_results.md", tmp_path / "b")


def test_missing_candidates_doc_and_marker_are_skipped(tree, capsys):
    out, docs = tree
    (docs / "model_candidates.md").unlink()
    assert run(out, docs) == 0
    assert "measured_latency skipped" in capsys.readouterr().err
    (docs / "model_candidates.md").write_text("# no marker yet\n")
    assert run(out, docs) == 0 and run(out, docs, "--check") == 0
    assert (docs / "model_candidates.md").read_text() == "# no marker yet\n"


def test_no_runs_leaves_docs_untouched(tmp_path):
    _, docs = make_tree(tmp_path)
    before = (docs / "smoke_test_results.md").read_bytes()
    empty = tmp_path / "empty"
    empty.mkdir()
    for argv in ([], ["--check"]):
        assert run(empty, docs, *argv) == 0
        assert run(tmp_path / "does-not-exist", docs, *argv) == 0
    assert (docs / "smoke_test_results.md").read_bytes() == before


def test_broken_markers_are_skipped_not_mangled(tree):
    out, docs = tree
    smoke = docs / "smoke_test_results.md"
    text = smoke.read_text().replace("<!-- END:privacy -->", "<!-- END:privacy -->\n<!-- END:privacy -->")
    smoke.write_text(text)
    assert run(out, docs) == 0
    new = smoke.read_text()
    assert new.count("<!-- END:privacy -->") == 2
    assert "_not generated yet_" in new.split("<!-- BEGIN:privacy -->")[1].split("<!-- END:privacy -->")[0]
    assert "_not generated yet_" not in bodies(new)["models"]
    assert run(out, docs, "--check") == 1               # G5 cannot pass while a block cannot be generated


def test_real_skeleton_has_every_marker_once_and_renders(tmp_path):
    out, _ = make_tree(tmp_path)
    docs = tmp_path / "real"
    docs.mkdir()
    shutil.copy(REPO / "docs" / "research" / "smoke_test_results.md", docs / "smoke_test_results.md")
    text = (docs / "smoke_test_results.md").read_text()
    for name in SMOKE_BLOCKS:
        assert not isinstance(rr.marker_span(text, name), str), name
        assert f"<!-- BEGIN:{name} -->\n_not generated yet_\n<!-- END:{name} -->" in text
    assert run(out, docs) == 0
    assert outside((docs / "smoke_test_results.md").read_text()) == outside(text)
    assert run(out, docs, "--check") == 0


def test_formatting_rules():
    assert rr.fmt_number("median_ms", 812.345) == "812.3"
    assert rr.fmt_number("median_ms", 5) == "5.0"
    assert rr.fmt_number("loadavg_1m", 0.4) == "0.40"
    assert rr.fmt_number("mean_conf", 0.5) == "0.500"
    assert rr.fmt_number("n_runs", 20.0) == "20"
    assert rr.fmt_number("median_ms", 500.25) == "500.3" and rr.fmt_number("x", 0.125) == "0.13"   # half up
    assert rr.fmt_number("timings_ms.det", 120) == "120.0"          # a nested key inherits its parent's unit
    assert rr.fmt_number("median_ms", 1e30) == f"{1e30:.1f}"        # beyond Decimal precision: plain format
    assert rr.parse_cell("pinned", "1") is True and rr.parse_cell("cap_hit", "False") is False
    assert rr.parse_cell("run", "2") == "2" and rr.parse_cell("median_ms", "nan") is None
    assert rr.mask_vin_like(f"vin {SYN_VIN} end") == f"vin {SYN_VIN_MASKED} end"
    assert rr.mask_vin_like("PASSENGERCARTRUCK") == "PASSENGERCARTRUCK"       # no digits: not VIN-like
    assert rr.mask_vin_like("sha 7c4a870b16bf642924d0b7473bb1a7df") == "sha 7c4a870b16bf642924d0b7473bb1a7df"
    assert rr.mask_vin_like("ocr 0apsyn1234567890oo") == "ocr 0apsyn1234567890oo"      # 18 chars, lowercase
    assert rr.mask_vin_like("ocr 0apsyn1234567890o") == "ocr 0apsyn12345******"       # 17 chars, any case
    assert rr.md_escape(f"key {SYN_VIN}") == f"key `{SYN_VIN_MASKED}`"                # headers and keys too
    assert rr.md_escape("a|b <tag> runs<=20") == "a\\|b &lt;tag> runs<=20"
