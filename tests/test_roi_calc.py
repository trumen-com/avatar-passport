"""scripts/roi_calc.py: formulas on known inputs, price anchors, and marker-only rewriting."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "roi_calc.py"


def _load():
    spec = importlib.util.spec_from_file_location("roi_calc", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rc = _load()

P = {
    "branches": 2, "inspections_per_branch_day": 10, "working_days_month": 25, "pilot_months": 3,
    "dispute_rate": 0.01, "dispute_avoided_share": 0.5, "damage_found_share": 0.1, "repair_conversion": 0.2,
    "tryon_conversion": 0.01, "gross_margin": 0.25, "paper_walkaround_min": 10, "app_handling_min": 3,
    "advisor_wait_share": 0.5, "open_hours_day": 10, "repair_ticket_uzs": 400_000, "appearance_ticket_uzs": 1_000_000,
}


def test_inspections_and_cost_per_inspection() -> None:
    assert rc.inspections_per_month(P) == 500                      # 2 x 10 x 25
    assert rc.hosting_usd_month(P) == pytest.approx(9.0)           # PRO only, CPU Basic $0/h
    assert rc.cost_per_inspection_usd(P) == pytest.approx(9.0 / 500)


def test_value_drivers_known_inputs() -> None:
    v = rc.value_drivers(P, deploy_s=120.0)
    assert v["disputes"] == pytest.approx(500 * 0.01 * 0.5 * 400_000 * 0.75)    # 750,000
    assert v["repair"] == pytest.approx(500 * 0.1 * 0.2 * 400_000 * 0.25)       # 1,000,000
    assert v["accessory"] == pytest.approx(500 * 0.01 * 1_000_000 * 0.25)       # 1,250,000
    assert v["wait_min"] == pytest.approx(1.0)                                  # 120 s / 60 x 0.5
    assert v["minutes_saved"] == pytest.approx(6.0)                             # 10 - 3 - 1
    assert v["advisor_hours"] == pytest.approx(50.0)                            # 500 x 6 / 60
    assert v["advisor"] == 0.0                                                  # not costed without a wage
    assert v["total"] == pytest.approx(3_000_000)
    assert rc.value_drivers(P, 120.0, advisor_cost_uzs_h=20_000)["advisor"] == pytest.approx(1_000_000)


def test_scenario_derived_metrics() -> None:
    s = rc.scenario(P, deploy_s=200.0)
    host_uzs = 9.0 * rc.FX_UZS_PER_USD
    assert s["hosting_uzs_month"] == pytest.approx(host_uzs)
    assert s["cost_per_inspection_uzs"] == pytest.approx(host_uzs / 500)
    assert s["net_uzs_month"] == pytest.approx(s["value_total"] - host_uzs)
    assert s["breakeven_repairs_month"] == pytest.approx(host_uzs / (400_000 * 0.25))
    assert s["compute_h_day"] == pytest.approx(2 * 10 * 200 / 3600)
    assert s["capacity_per_hour"] == pytest.approx(18.0)


def test_pilot_budget() -> None:
    b = rc.pilot_budget(P)
    assert b["pro_usd"] == pytest.approx(27.0)
    assert b["cpu_upgrade_always_on_usd"] == pytest.approx(0.03 * 720 * 3)
    assert b["t4_business_hours_usd"] == pytest.approx(0.40 * 10 * 25 * 3)
    assert b["t4_always_on_usd"] == pytest.approx(0.40 * 720 * 3)
    assert b["inspections_pilot"] == 1500


def _csv(path: Path, rows: list[tuple[str, str, str, int]]) -> Path:
    head = "service,service_category,part,price_uzs\n"
    path.write_text(head + "".join(f"{s},{c},{p},{v}\n" for s, c, p, v in rows), encoding="utf-8")
    return path


def test_price_anchors(tmp_path: Path) -> None:
    f = _csv(tmp_path / "p.csv", [
        ("paint", "paint", "any", 300), ("paint one panel", "paint", "panel", 500), ("paint one panel", "paint", "panel", 900),
        ("dent", "dent", "any", 200), ("replace", "panel_replacement", "panel", 1000),
        ("polishing", "polish", "any", 700), ("body polishing", "polish", "whole_body", 400),
        ("body polishing (dealer promo)", "promo", "whole_body", 2000), ("wheel alignment", "promo", "wheels", 50),
        ("inspection", "inspection", "whole_body", 9999), ("paint whole body", "paint", "whole_body", 9999),
    ])
    a = rc.price_anchors(f)
    assert a["category_median"] == {"dent": 200, "paint": 500, "panel_replacement": 1000}
    assert a["repair_ticket"] == {"low": 200, "base": 500, "high": 1000}     # median of 200,300,500,900,1000
    assert a["appearance_ticket"] == {"low": 400, "base": 700, "high": 2000}  # polish rows only, promo included
    assert a["appearance_n"] == 3


def test_replace_block_only_touches_markers() -> None:
    text = "head\n<!-- BEGIN:roi -->\nold\n<!-- END:roi -->\ntail <!-- BEGIN:x -->\nkeep\n<!-- END:x -->\n"
    out = rc.replace_block(text, "roi", "new\n")
    assert out == "head\n<!-- BEGIN:roi -->\nnew\n<!-- END:roi -->\ntail <!-- BEGIN:x -->\nkeep\n<!-- END:x -->\n"
    with pytest.raises(ValueError):
        rc.replace_block("no markers", "roi", "x")


def _fake_repo(tmp: Path) -> Path:
    (tmp / "outputs/e2e/run1").mkdir(parents=True)
    (tmp / "outputs/e2e/run1/result.json").write_text(json.dumps(
        {"timings_ms": {"total": 30000.0}, "extra": {"quality": {"kept": 10}, "config": {"device": "cuda"}}}))
    (tmp / "outputs/smoke/R2-local-cpu").mkdir(parents=True)
    (tmp / "outputs/smoke/R2-local-cpu/estimate.csv").write_text(
        "run,tier,step,included,count,median_ms,subtotal_ms,source,basis,x2 safety factor (assumption),note\n"
        "R2,cpu_ort2,TOTAL per inspection (included steps),True,,,100000.0,sum,derived,200000.0,n\n")
    (tmp / "data/reference").mkdir(parents=True)
    _csv(tmp / "data/reference/repair_prices.csv",
         [("paint", "paint", "any", 300), ("dent", "dent", "any", 200), ("polishing", "polish", "any", 700)])
    (tmp / "docs/pitch").mkdir(parents=True)
    (tmp / "docs/pitch/roi_model.md").write_text("intro text\n<!-- BEGIN:roi -->\nstale\n<!-- END:roi -->\noutro\n")
    blocks = "".join(f"## {n}\n<!-- BEGIN:{n} -->\n<!-- END:{n} -->\nhand {n}\n" for n in
                     ("measured_inspection_time", "measured_3d_time", "cost_per_inspection", "pilot_budget"))
    (tmp / "docs/pitch/cost_model.md").write_text("# Cost\n" + blocks)
    return tmp


def test_main_rewrites_only_between_markers_and_check(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    assert rc.main(["--repo", str(repo), "--check"]) == 1                     # stale before the first write
    assert rc.main(["--repo", str(repo)]) == 0
    roi = (repo / "docs/pitch/roi_model.md").read_text()
    assert roi.startswith("intro text\n<!-- BEGIN:roi -->\n<!-- generated by scripts/roi_calc.py")
    assert roi.endswith("<!-- END:roi -->\noutro\n")
    assert "stale" not in roi and "| `repair_ticket_uzs` | 200 | 250 | 300 |" in roi
    cost = (repo / "docs/pitch/cost_model.md").read_text()
    for n in ("measured_inspection_time", "measured_3d_time", "cost_per_inspection", "pilot_budget"):
        assert f"<!-- END:{n} -->\nhand {n}\n" in cost
    assert "| HF CPU Basic proxy: ONNX Runtime pinned to 2 threads | 100.0 |" in cost
    assert rc.main(["--repo", str(repo), "--check"]) == 0                     # idempotent


def test_missing_measurements_leave_docs_untouched(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    (repo / "outputs/e2e/run1/result.json").unlink()
    before = (repo / "docs/pitch/roi_model.md").read_text()
    assert rc.main(["--repo", str(repo)]) == 0
    assert (repo / "docs/pitch/roi_model.md").read_text() == before


def test_cli_override(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _fake_repo(tmp_path)
    assert rc.main(["--repo", str(repo), "--print", "--set", "dispute_rate=0.2"]) == 0
    assert "| `dispute_rate` | 20.0% | 20.0% | 20.0% |" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        rc.main(["--repo", str(repo), "--print", "--set", "nope=1"])
