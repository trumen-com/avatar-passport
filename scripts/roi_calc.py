"""Cost-per-inspection, pilot budget and ROI tables for docs/pitch/, generated from one parameters block.

Every number this script writes is one of:
  (a) measured: read from a file under outputs/ (path printed next to it);
  (b) sourced: a constant below with the file in docs/research or docs/decisions that cites the public source;
  (c) observed price: computed from data/reference/repair_prices.csv (median rule as in cost.estimator);
  (d) ASSUMPTION: a value in PARAMS with kind "assumption" -> replace with partner data;
  (e) derived: computed from (a)-(d) by the formulas in this file (also printed in docs/pitch/roi_model.md).
No partner data was available; every ROI figure is illustrative.

It rewrites ONLY the text between "<!-- BEGIN:<name> -->" and "<!-- END:<name> -->" in
  docs/pitch/roi_model.md   roi
  docs/pitch/cost_model.md  measured_inspection_time, measured_3d_time, cost_per_inspection, pilot_budget
Everything outside the markers is kept byte for byte. outputs/ is gitignored: if a required measured file is
missing (fresh clone), the docs are left untouched and a warning is printed.

Usage:
    python scripts/roi_calc.py                       # rewrite the generated blocks
    python scripts/roi_calc.py --check               # exit 1 if a rewrite would change a doc
    python scripts/roi_calc.py --print               # print the blocks, write nothing
    python scripts/roi_calc.py --set dispute_rate=0.01,0.02,0.03 --print    # low,base,high (or one value)
    python scripts/roi_calc.py --advisor-cost-uzs-per-hour <partner figure> --print
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
import sys
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SCENARIOS = ("low", "base", "high")
LAPTOP = "measured on the dev laptop (RTX 5050 / Ryzen 7 260)"
PROXY = "2-thread pinned proxy, optimistic lower bound"

# ---------------------------------------------------------------- sourced constants (kind "sourced")
FX_UZS_PER_USD = 11897.90          # CBU, 2026-10-10: docs/research/market_prices.md section 1
FX_SOURCE = "CBU rate 2026-10-10 (docs/research/market_prices.md s.1)"
HF_PRO_USD_MONTH = 9.0             # docs/decisions/003-deploy-hf-docker-local-first.md, huggingface.co/pricing
HF_CPU_BASIC_USD_HOUR = 0.0        # ADR 003, Spaces overview
HF_CPU_UPGRADE_USD_HOUR = 0.03     # ADR 003 / phase0_findings_deploy_market.md s.1
HF_T4_USD_HOUR = 0.40              # phase0_findings_deploy_market.md s.1 ("GPU tiers start at a T4 for $0.40/h")
HF_FREE_PRIVATE_STORAGE_GB = 100   # phase0_findings_deploy_market.md s.1 (stated for free accounts)
X2_SAFETY = 2.0                    # the smoke notebook's "x2 safety factor (assumption)" (estimate.csv)

# ---------------------------------------------------------------- measured inputs (paths relative to REPO)
E2E_GPU = "outputs/e2e/run1/result.json"
E2E_CPU2 = "outputs/e2e/app_cpu2/result.json"  # full app inspection on the 2-thread pinned proxy
SMOKE_GPU_EST = "outputs/smoke/R1-local-gpu/estimate.csv"
SMOKE_CPU_EST = "outputs/smoke/R2-local-cpu/estimate.csv"
R3D = {"gpu": "outputs/recon3d/R3D-gpu/results.json",
       "cpu16": "outputs/recon3d/R3D-cpu16/results.json",
       "cpu2": "outputs/recon3d/R3D-cpu2-pinned/results.json"}
R3D_MODULE_GPU = "outputs/recon3d/module-gpu/result.json"
SFM_HQ = "outputs/recon3d/hq/sfm_report.json"
BRUSH_GLOB = "outputs/recon3d/hq/brush_report_*.json"
STORAGE_DIRS = ("outputs/e2e/run1", "outputs/recon3d/module-gpu")
PRICES_CSV = "data/reference/repair_prices.csv"

ROI_DOC = "docs/pitch/roi_model.md"
COST_DOC = "docs/pitch/cost_model.md"
BLOCK_RE = re.compile(r"(<!-- BEGIN:(\w+) -->\n).*?(<!-- END:\2 -->)", re.S)

PARTNER = "ASSUMPTION - replace with partner data"


@dataclass(frozen=True)
class Param:
    name: str
    low: float | None
    base: float | None
    high: float | None
    unit: str
    kind: str          # "assumption" | "sourced" | "price" | "measured"
    source: str

    def value(self, scenario: str) -> float | None:
        return getattr(self, scenario)


# low = pessimistic scenario value, high = optimistic scenario value (ordered by effect on value, not size).
ASSUMPTIONS: tuple[Param, ...] = (
    Param("branches", 1, 1, 3, "service points in the pilot", "assumption",
          f"{PARTNER}. High = the 3 Royal Motors service desks named in docs/research/market_prices.md s.5"),
    Param("inspections_per_branch_day", 5, 10, 20, "inspections per branch per working day", "assumption",
          f"{PARTNER} (trade-in + service-intake volume)"),
    Param("working_days_month", 24, 24, 24, "days", "assumption", f"{PARTNER} (6-day week x 4 weeks)"),
    Param("pilot_months", 3, 3, 3, "months", "assumption", PARTNER),
    Param("dispute_rate", 0.005, 0.01, 0.02, "share of inspected cars with a 'that scratch wasn't there' dispute",
          "assumption", PARTNER),
    Param("dispute_avoided_share", 0.25, 0.5, 0.75, "share of those disputes settled by the timestamped passport",
          "assumption", PARTNER),
    Param("damage_found_share", 0.05, 0.10, 0.20,
          "share of inspected cars with real damage the advisor would have missed", "assumption",
          f"{PARTNER}; model recall not yet measured (Phase 2)"),
    Param("repair_conversion", 0.10, 0.20, 0.30, "share of those customers who book the repair at the dealer",
          "assumption", PARTNER),
    Param("tryon_conversion", 0.005, 0.01, 0.02,
          "share of inspections leading to an appearance/accessory sale attributed to the paint try-on",
          "assumption", PARTNER),
    Param("gross_margin", 0.20, 0.30, 0.40, "dealer gross margin on body/appearance work", "assumption", PARTNER),
    Param("paper_walkaround_min", 5, 10, 15, "advisor minutes per paper walkaround today", "assumption",
          f"{PARTNER} (time a paper walkaround at reception)"),
    Param("app_handling_min", 4, 3, 2, "advisor minutes to film 30-60 s + VIN photo + review the passport",
          "assumption", "ASSUMPTION; the 30-60 s video length is from CLAUDE.md s.1. Replace with a timed pilot"),
    Param("advisor_wait_share", 1.0, 0.5, 0.0,
          "share of the processing time the advisor waits idle", "assumption",
          "ASSUMPTION; 0 = advisor does other work while the passport builds"),
    Param("open_hours_day", 10, 10, 10, "hours per working day the hardware is billed (GPU option)", "assumption",
          PARTNER),
)
ASSUMPTION_NAMES = tuple(p.name for p in ASSUMPTIONS)


# ---------------------------------------------------------------- small helpers
def r_half(x: float, nd: int = 0) -> Decimal:
    q = Decimal(1).scaleb(-nd)
    return Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP)


def fmt(x: float | None, nd: int = 0) -> str:
    if x is None:
        return "n/a"
    d = r_half(float(x), nd)
    return f"{d:,.{nd}f}"


def pct(x: float) -> str:
    return f"{fmt(x * 100, 1)}%"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _total_row(path: Path, tier: str) -> tuple[float, float | None] | None:
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("tier") == tier and row.get("step", "").startswith("TOTAL"):
                    x2 = row.get("x2 safety factor (assumption)") or ""
                    return float(row["subtotal_ms"]), (float(x2) if x2 else None)
    except (OSError, ValueError, KeyError):
        return None
    return None


def _dir_bytes(path: Path) -> int | None:
    if not path.is_dir():
        return None
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ---------------------------------------------------------------- observed price anchors (kind "price")
PER_PART = {"any", "panel"}
REPAIR_CATS = ("paint", "dent", "scratch", "panel_replacement")  # polish rows are detailing (market_prices.md s.2)


def _median_uzs(values: list[int]) -> int:
    return int(r_half(statistics.median(values)))


def price_anchors(csv_path: Path) -> dict[str, Any]:
    """Repair ticket (low/base/high) and appearance ticket (low/base/high) from the observed CSV rows.

    repair ticket: per-part rows of the usable repair categories (the cost.estimator rule).
      low  = the lowest per-category median; base = the median of all those rows; high = the highest
      per-category median.
    appearance ticket (proxy for an accessory/appearance sale; no accessory price list is available):
      every polish row incl. whole-body and the ended dealer promo -> low = min, base = median, high = max.
    """
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    rep = [r for r in rows if r["service_category"] in REPAIR_CATS and r["part"] in PER_PART]
    by_cat: dict[str, list[int]] = {}
    for r in rep:
        by_cat.setdefault(r["service_category"], []).append(int(r["price_uzs"]))
    cat_median = {c: _median_uzs(v) for c, v in sorted(by_cat.items())}
    all_rep = [int(r["price_uzs"]) for r in rep]
    polish = [int(r["price_uzs"]) for r in rows if r["service_category"] in {"polish", "promo"}
              and "polish" in r["service"].lower()]
    lo_cat = min(cat_median, key=lambda c: cat_median[c])
    hi_cat = max(cat_median, key=lambda c: cat_median[c])
    return {
        "category_median": cat_median,
        "category_n": {c: len(v) for c, v in sorted(by_cat.items())},
        "repair_ticket": {"low": cat_median[lo_cat], "base": _median_uzs(all_rep), "high": cat_median[hi_cat]},
        "repair_basis": {"low": f"median of '{lo_cat}' rows", "base": f"median of all {len(all_rep)} per-part repair rows",
                         "high": f"median of '{hi_cat}' rows"},
        "appearance_ticket": {"low": min(polish), "base": _median_uzs(polish), "high": max(polish)},
        "appearance_n": len(polish),
    }


# ---------------------------------------------------------------- measured inputs (kind "measured")
@dataclass(frozen=True)
class Measured:
    gpu_e2e_s: float
    gpu_e2e_frames: int
    gpu_e2e_device: str
    smoke_gpu_s: float | None
    cpu_torch_s: float | None
    cpu_ort16_s: float | None
    cpu_ort2_s: float
    cpu_ort2_x2_s: float
    r3d: dict[str, dict[str, Any]]
    r3d_module_gpu: dict[str, Any] | None
    sfm: dict[str, Any] | None
    brush: list[tuple[str, dict[str, Any]]]
    storage_bytes: int | None
    app_cpu2_s: float | None = None
    app_cpu2_frames: int | None = None


def load_measured(repo: Path) -> tuple[Measured | None, list[str]]:
    missing: list[str] = []
    e2e = _read_json(repo / E2E_GPU)
    ort2 = _total_row(repo / SMOKE_CPU_EST, "cpu_ort2")
    if e2e is None:
        missing.append(E2E_GPU)
    if ort2 is None:
        missing.append(SMOKE_CPU_EST)
    if missing:
        return None, missing
    assert e2e is not None and ort2 is not None
    extra = e2e.get("extra", {})
    smoke_gpu = _total_row(repo / SMOKE_GPU_EST, "gpu")
    torch16 = _total_row(repo / SMOKE_CPU_EST, "cpu_torch")
    ort16 = _total_row(repo / SMOKE_CPU_EST, "cpu_ort16")
    r3d = {k: d for k, p in R3D.items() if (d := _read_json(repo / p)) is not None}
    brush = []
    for p in sorted(glob.glob(str(repo / BRUSH_GLOB))):
        d = _read_json(Path(p))
        if d is not None:
            brush.append((str(Path(p).relative_to(repo)), d))
    sizes = [_dir_bytes(repo / d) for d in STORAGE_DIRS]
    storage = sum(s for s in sizes if s is not None) if all(s is not None for s in sizes) else None
    m = Measured(
        gpu_e2e_s=float(e2e["timings_ms"]["total"]) / 1000,
        gpu_e2e_frames=int(extra.get("quality", {}).get("kept", 0)),
        gpu_e2e_device=str(extra.get("config", {}).get("device", "?")),
        smoke_gpu_s=smoke_gpu[0] / 1000 if smoke_gpu else None,
        cpu_torch_s=torch16[0] / 1000 if torch16 else None,
        cpu_ort16_s=ort16[0] / 1000 if ort16 else None,
        cpu_ort2_s=ort2[0] / 1000,
        cpu_ort2_x2_s=(ort2[1] if ort2[1] is not None else ort2[0] * X2_SAFETY) / 1000,
        r3d=r3d, r3d_module_gpu=_read_json(repo / R3D_MODULE_GPU), sfm=_read_json(repo / SFM_HQ),
        brush=brush, storage_bytes=storage,
        app_cpu2_s=(float(c2["timings_ms"]["total"]) / 1000) if (c2 := _read_json(repo / E2E_CPU2)) else None,
        app_cpu2_frames=len((c2 or {}).get("extra", {}).get("frames", [])) or None,
    )
    return m, []


# ---------------------------------------------------------------- formulas (pure; tested)
def inspections_per_month(p: dict[str, float]) -> float:
    return p["branches"] * p["inspections_per_branch_day"] * p["working_days_month"]


def hosting_usd_month(p: dict[str, float]) -> float:
    """PRO subscription + CPU Basic hardware ($0/h)."""
    return HF_PRO_USD_MONTH + HF_CPU_BASIC_USD_HOUR * 24 * 30


def cost_per_inspection_usd(p: dict[str, float]) -> float:
    return hosting_usd_month(p) / inspections_per_month(p)


def value_drivers(p: dict[str, float], deploy_s: float, advisor_cost_uzs_h: float | None = None) -> dict[str, float]:
    """Monthly value per driver in UZS (advisor time also in hours)."""
    n = inspections_per_month(p)
    disputes = n * p["dispute_rate"] * p["dispute_avoided_share"] * p["repair_ticket_uzs"] * (1 - p["gross_margin"])
    repair = n * p["damage_found_share"] * p["repair_conversion"] * p["repair_ticket_uzs"] * p["gross_margin"]
    accessory = n * p["tryon_conversion"] * p["appearance_ticket_uzs"] * p["gross_margin"]
    wait_min = deploy_s / 60 * p["advisor_wait_share"]
    minutes_saved = p["paper_walkaround_min"] - p["app_handling_min"] - wait_min
    hours = n * minutes_saved / 60
    advisor = hours * advisor_cost_uzs_h if advisor_cost_uzs_h is not None else 0.0
    return {"disputes": disputes, "repair": repair, "accessory": accessory, "advisor": advisor,
            "advisor_hours": hours, "minutes_saved": minutes_saved, "wait_min": wait_min,
            "total": disputes + repair + accessory + advisor}


def scenario(p: dict[str, float], deploy_s: float, advisor_cost_uzs_h: float | None = None) -> dict[str, float]:
    n = inspections_per_month(p)
    host_usd = hosting_usd_month(p)
    host_uzs = host_usd * FX_UZS_PER_USD
    v = value_drivers(p, deploy_s, advisor_cost_uzs_h)
    return {
        "inspections_month": n,
        "hosting_usd_month": host_usd,
        "hosting_uzs_month": host_uzs,
        "cost_per_inspection_usd": host_usd / n,
        "cost_per_inspection_uzs": host_uzs / n,
        **{f"value_{k}": x for k, x in v.items()},
        "net_uzs_month": v["total"] - host_uzs,
        "value_to_cost": v["total"] / host_uzs if host_uzs else float("inf"),
        "breakeven_repairs_month": host_uzs / (p["repair_ticket_uzs"] * p["gross_margin"]),
        "compute_h_day": p["branches"] * p["inspections_per_branch_day"] * deploy_s / 3600,
        "capacity_per_hour": 3600 / deploy_s,
    }


def pilot_budget(p: dict[str, float]) -> dict[str, float]:
    months = p["pilot_months"]
    biz_h = p["open_hours_day"] * p["working_days_month"]
    return {
        "pro_usd": HF_PRO_USD_MONTH * months,
        "cpu_upgrade_always_on_usd": HF_CPU_UPGRADE_USD_HOUR * 24 * 30 * months,
        "t4_business_hours_usd": HF_T4_USD_HOUR * biz_h * months,
        "t4_always_on_usd": HF_T4_USD_HOUR * 24 * 30 * months,
        "business_hours_month": biz_h,
        "inspections_pilot": inspections_per_month(p) * months,
    }


# ---------------------------------------------------------------- parameter assembly
def build_params(anchors: dict[str, Any], overrides: dict[str, tuple[float, float, float]] | None = None
                 ) -> list[Param]:
    params = list(ASSUMPTIONS)
    for name, vals in (overrides or {}).items():
        idx = next((i for i, q in enumerate(params) if q.name == name), None)
        if idx is None:
            raise SystemExit(f"--set: unknown parameter {name!r}; known: {', '.join(ASSUMPTION_NAMES)}")
        params[idx] = replace(params[idx], low=vals[0], base=vals[1], high=vals[2], source=params[idx].source + " (overridden on the CLI)")
    rt, at = anchors["repair_ticket"], anchors["appearance_ticket"]
    params += [
        Param("repair_ticket_uzs", rt["low"], rt["base"], rt["high"], "UZS per body repair (a 'from' price)", "price",
              f"{PRICES_CSV}: low {anchors['repair_basis']['low']}, base {anchors['repair_basis']['base']}, "
              f"high {anchors['repair_basis']['high']} (docs/research/market_prices.md s.2)"),
        Param("appearance_ticket_uzs", at["low"], at["base"], at["high"],
              "UZS per appearance/accessory sale (proxy)", "price",
              f"{PRICES_CSV}: min / median / max of the {anchors['appearance_n']} polishing rows (incl. whole-body "
              "and an ended dealer promo). PROXY - replace with the partner's accessory price list"),
    ]
    return params


def values_for(params: list[Param], sc: str) -> dict[str, float]:
    return {q.name: float(q.value(sc)) for q in params if q.value(sc) is not None}


# ---------------------------------------------------------------- rendering
def _table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


GEN = "<!-- generated by scripts/roi_calc.py; do not edit by hand -->"


def _pval(q: Param, x: float | None) -> str:
    if x is None:
        return "n/a"
    if "share" in q.name or q.name.endswith(("_rate", "_conversion", "margin")):
        return pct(x)
    return fmt(x, 0) if float(x).is_integer() else fmt(x, 2)


def render_roi(params: list[Param], anchors: dict[str, Any], m: Measured, adv: float | None) -> str:
    deploy_s = m.cpu_ort2_x2_s
    res = {sc: scenario(values_for(params, sc), deploy_s, adv) for sc in SCENARIOS}
    parts = [GEN, "", "**Parameters** (low = pessimistic, high = optimistic):", ""]
    parts.append(_table(["parameter", "low", "base", "high", "unit", "type", "source"],
                        [[f"`{q.name}`", _pval(q, q.low), _pval(q, q.base), _pval(q, q.high), q.unit,
                          q.kind.upper() if q.kind == "assumption" else q.kind, q.source] for q in params]))
    parts += ["", "Fixed inputs: "
              f"FX {fmt(FX_UZS_PER_USD, 2)} UZS/USD ({FX_SOURCE}); HF PRO ${fmt(HF_PRO_USD_MONTH, 2)}/month "
              "(ADR 003); CPU Basic $0/hour (ADR 003); processing time used for advisor waiting and capacity = "
              f"{fmt(deploy_s, 1)} s ({PROXY}, x{fmt(X2_SAFETY, 0)} safety factor, `{SMOKE_CPU_EST}`).",
              "", "**Observed price medians used above** (per-part rows only, as `cost.estimator` uses them):", ""]
    parts.append(_table(["category", "rows", "median UZS"],
                        [[c, str(anchors["category_n"][c]), fmt(v)] for c, v in anchors["category_median"].items()]))
    rows = [
        ("inspections per month", "inspections_month", 0),
        ("hosting cost, USD per month", "hosting_usd_month", 2),
        ("hosting cost, UZS per month", "hosting_uzs_month", 0),
        ("cost per inspection, USD", "cost_per_inspection_usd", 4),
        ("cost per inspection, UZS", "cost_per_inspection_uzs", 0),
        ("value: disputes avoided, UZS/month", "value_disputes", 0),
        ("value: repair upsell margin, UZS/month", "value_repair", 0),
        ("value: accessory upsell margin, UZS/month", "value_accessory", 0),
        ("advisor minutes saved per inspection", "value_minutes_saved", 1),
        ("advisor hours saved per month", "value_advisor_hours", 1),
        ("value: advisor time, UZS/month" + ("" if adv is not None else " (not costed: no wage data)"),
         "value_advisor", 0),
        ("total value, UZS/month", "value_total", 0),
        ("net (value - hosting), UZS/month", "net_uzs_month", 0),
        ("value / hosting cost (team time, integration, training not costed)", "value_to_cost", 1),
        ("break-even: upsold repairs per month to cover hosting", "breakeven_repairs_month", 2),
        ("compute hours needed per working day (all branches)", "compute_h_day", 2),
    ]
    parts += ["", "**Scenario results (monthly, illustrative):**", ""]
    parts.append(_table(["metric", "low", "base", "high"],
                        [[label] + [fmt(res[sc][key], nd) for sc in SCENARIOS] for label, key, nd in rows]))
    # one-way sensitivity: each assumption at its low/high, every other parameter at base
    base_p = values_for(params, "base")
    base_total = res["base"]["value_total"]
    sens = []
    for q in params:
        if q.kind not in {"assumption", "price"} or q.low == q.high:
            continue
        lo = scenario({**base_p, q.name: float(q.low)}, deploy_s, adv)["value_total"]
        hi = scenario({**base_p, q.name: float(q.high)}, deploy_s, adv)["value_total"]
        sens.append((abs(hi - lo), q.name, lo, hi))
    sens.sort(key=lambda t: (-t[0], t[1]))
    parts += ["", f"**One-way sensitivity** of total monthly value (base = {fmt(base_total)} UZS); one parameter "
              "at its low / high value, all others at base, sorted by swing:", ""]
    parts.append(_table(["parameter", "value at low, UZS/month", "value at high, UZS/month", "swing, UZS"],
                        [[f"`{n}`", fmt(lo), fmt(hi), fmt(sw)] for sw, n, lo, hi in sens if sw > 0]))
    return "\n".join(parts) + "\n"


def render_measured_time(m: Measured) -> str:
    rows = [
        ["GPU, full pipeline (app code path, end to end)", fmt(m.gpu_e2e_s, 1), f"{m.gpu_e2e_frames} frames, no photos",
         LAPTOP + f"; device={m.gpu_e2e_device}", f"`{E2E_GPU}` timings_ms.total", fmt(3600 / m.gpu_e2e_s, 1)],
    ]
    if m.smoke_gpu_s is not None:
        rows.append(["GPU, smoke-test per-inspection estimate", fmt(m.smoke_gpu_s, 1), "72 frames + 9 photos + 3 VIN photos",
                     LAPTOP, f"`{SMOKE_GPU_EST}` TOTAL (tier gpu)", fmt(3600 / m.smoke_gpu_s, 1)])
    if m.cpu_torch_s is not None:
        rows.append(["CPU, 16 threads, PyTorch", fmt(m.cpu_torch_s, 1), "24 frames + 9 photos + 3 VIN photos", LAPTOP,
                     f"`{SMOKE_CPU_EST}` TOTAL (tier cpu_torch)", fmt(3600 / m.cpu_torch_s, 1)])
    if m.cpu_ort16_s is not None:
        rows.append(["CPU, 16 threads, ONNX Runtime", fmt(m.cpu_ort16_s, 1), "24 frames + 9 photos + 3 VIN photos", LAPTOP,
                     f"`{SMOKE_CPU_EST}` TOTAL (tier cpu_ort16)", fmt(3600 / m.cpu_ort16_s, 1)])
    rows.append(["HF CPU Basic proxy: ONNX Runtime pinned to 2 threads", fmt(m.cpu_ort2_s, 1),
                 "24 frames + 9 photos + 3 VIN photos", PROXY, f"`{SMOKE_CPU_EST}` TOTAL (tier cpu_ort2)",
                 fmt(3600 / m.cpu_ort2_s, 1)])
    rows.append([f"HF CPU Basic planning figure: proxy x{fmt(X2_SAFETY, 0)} safety factor (assumption)",
                 fmt(m.cpu_ort2_x2_s, 1), "as above", PROXY + " x2", f"`{SMOKE_CPU_EST}` x2 column",
                 fmt(3600 / m.cpu_ort2_x2_s, 1)])
    if m.app_cpu2_s is not None:
        rows.append(["HF CPU Basic proxy: full app inspection (server pinned to 2 threads, CUDA hidden)",
                     fmt(m.app_cpu2_s, 1), f"{m.app_cpu2_frames} frames + 1 VIN photo", PROXY,
                     f"`{E2E_CPU2}` timings_ms.total", fmt(3600 / m.app_cpu2_s, 1)])
    rows.append(["HF CPU Basic, on the Space itself", "not measured yet", "", "Phase 4", "", ""])
    return "\n".join([GEN, "", _table(["configuration", "seconds per inspection", "input", "label", "source",
                                       "inspections per hour (3600 / s, one at a time)"], rows)]) + "\n"


def render_3d_time(m: Measured) -> str:
    rows = []
    labels = {"gpu": ("DA3-BASE, GPU", LAPTOP), "cpu16": ("DA3-BASE, CPU (16-CPU laptop)", LAPTOP),
              "cpu2": ("DA3-BASE, CPU pinned to 2 threads", PROXY)}
    for key in ("gpu", "cpu16", "cpu2"):
        d = m.r3d.get(key)
        if not d:
            rows.append([labels[key][0], "missing", "", "", "", "", f"`{R3D[key]}` missing"])
            continue
        s = d.get("seconds", {})
        tot = sum(float(v) for v in s.values())
        rows.append([labels[key][0], str(d.get("n_frames")), fmt(s.get("model_load"), 1), fmt(s.get("inference"), 1),
                     fmt(tot, 1), labels[key][1] + f"; torch threads {d.get('torch_threads')}", f"`{R3D[key]}`"])
    if m.r3d_module_gpu:
        d = m.r3d_module_gpu
        s = d.get("seconds", {})
        rows.append(["DA3-BASE, GPU, app module (pins, previews, web PLY)", str(d.get("n_frames")),
                     fmt(s.get("model_load"), 1), fmt(s.get("inference"), 1), fmt(s.get("total"), 1), LAPTOP,
                     f"`{R3D_MODULE_GPU}` seconds.total"])
    if m.sfm:
        s = m.sfm.get("seconds", {})
        rows.append([f"pycolmap {m.sfm.get('pycolmap')} SfM (input for Brush splats)", str(m.sfm.get("n_images")), "",
                     "", fmt(sum(float(v) for v in s.values()), 1), LAPTOP + "; features + matching + mapping",
                     f"`{SFM_HQ}`"])
    if m.brush:
        for path, d in m.brush:
            rows.append([f"Brush splat training, GPU, {d.get('steps')} steps (exit {d.get('exit_code')})", "", "", "",
                         fmt(d.get("seconds"), 1), LAPTOP, f"`{path}`"])
    else:
        rows.append(["Brush splat training, GPU", "", "", "", "not measured yet", "",
                     f"no `{BRUSH_GLOB}` file yet"])
    return "\n".join([GEN, "", _table(["3D step", "frames", "model load s", "inference s", "total s", "label",
                                       "source"], rows)]) + "\n"


def render_cost(params: list[Param], m: Measured) -> str:
    res = {sc: scenario(values_for(params, sc), m.cpu_ort2_x2_s) for sc in SCENARIOS}
    rows = [
        ["inspections per month = branches x inspections/branch/day x working days"]
        + [fmt(res[sc]["inspections_month"]) for sc in SCENARIOS],
        ["fixed hosting, USD/month = HF PRO $9 + CPU Basic $0/h"] + [fmt(res[sc]["hosting_usd_month"], 2) for sc in SCENARIOS],
        ["cost per inspection, USD = fixed hosting / inspections per month"]
        + [fmt(res[sc]["cost_per_inspection_usd"], 4) for sc in SCENARIOS],
        [f"cost per inspection, UZS = USD x {fmt(FX_UZS_PER_USD, 2)}"]
        + [fmt(res[sc]["cost_per_inspection_uzs"]) for sc in SCENARIOS],
        ["marginal cost per inspection (no paid API; CPU Basic $0/h)"] + ["0"] * 3,
        [f"compute hours per working day at {fmt(m.cpu_ort2_x2_s, 1)} s per inspection (proxy x2)"]
        + [fmt(res[sc]["compute_h_day"], 2) for sc in SCENARIOS],
    ]
    if m.storage_bytes is not None:
        mb = m.storage_bytes / 1e6
        rows.append([f"storage per inspection, MB (sizes of `{STORAGE_DIRS[0]}` + `{STORAGE_DIRS[1]}`)"]
                    + [fmt(mb, 1)] * 3)
        rows.append(["storage per month, GB = MB x inspections per month / 1000"]
                    + [fmt(mb * res[sc]["inspections_month"] / 1000, 1) for sc in SCENARIOS])
    return "\n".join([GEN, "", _table(["item (formula)", "low", "base", "high"], rows)]) + "\n"


def render_pilot(params: list[Param], m: Measured) -> str:
    b = {sc: pilot_budget(values_for(params, sc)) for sc in SCENARIOS}
    rows = [
        ["pilot length, months (ASSUMPTION)"] + [fmt(values_for(params, sc)["pilot_months"]) for sc in SCENARIOS],
        ["inspections in the pilot"] + [fmt(b[sc]["inspections_pilot"]) for sc in SCENARIOS],
        ["HF PRO, USD = $9 x months (required)"] + [fmt(b[sc]["pro_usd"], 2) for sc in SCENARIOS],
        ["HF PRO, UZS"] + [fmt(b[sc]["pro_usd"] * FX_UZS_PER_USD) for sc in SCENARIOS],
        ["option A: CPU Upgrade 8 vCPU, USD = $0.03/h x 720 h (24 h x 30 days, ASSUMPTION) x months, always on"]
        + [fmt(b[sc]["cpu_upgrade_always_on_usd"], 2) for sc in SCENARIOS],
        ["option B: T4 GPU, USD = $0.40/h x open hours/day x working days x months"]
        + [fmt(b[sc]["t4_business_hours_usd"], 2) for sc in SCENARIOS],
        ["option B': T4 GPU always on, USD = $0.40/h x 720 h (24 h x 30 days) x months"]
        + [fmt(b[sc]["t4_always_on_usd"], 2) for sc in SCENARIOS],
        ["cash total, base plan (PRO only), USD"] + [fmt(b[sc]["pro_usd"], 2) for sc in SCENARIOS],
        ["cash total with option B, USD"] + [fmt(b[sc]["pro_usd"] + b[sc]["t4_business_hours_usd"], 2) for sc in SCENARIOS],
    ]
    if m.storage_bytes is not None:
        gb = m.storage_bytes / 1e9
        rows.append([f"storage over the pilot, GB (vs {HF_FREE_PRIVATE_STORAGE_GB} GB private storage reported for "
                     "free accounts)"] + [fmt(gb * b[sc]["inspections_pilot"], 1) for sc in SCENARIOS])
    return "\n".join([GEN, "", _table(["item", "low", "base", "high"], rows)]) + "\n"


# ---------------------------------------------------------------- marker handling
def replace_block(text: str, name: str, body: str) -> str:
    """Replace the text between BEGIN:name and END:name; everything else is kept byte for byte."""
    begin, end = f"<!-- BEGIN:{name} -->\n", f"<!-- END:{name} -->"
    i = text.find(begin)
    j = text.find(end, i + len(begin)) if i >= 0 else -1
    if i < 0 or j < 0 or text.count(begin) != 1:
        raise ValueError(f"marker pair for block {name!r} missing or duplicated")
    return text[: i + len(begin)] + body + text[j:]


def render_all(repo: Path, overrides: dict[str, tuple[float, float, float]] | None = None,
               advisor_cost: float | None = None) -> tuple[dict[str, dict[str, str]], list[str]]:
    m, missing = load_measured(repo)
    if m is None:
        return {}, missing
    anchors = price_anchors(repo / PRICES_CSV)
    params = build_params(anchors, overrides)
    return {
        ROI_DOC: {"roi": render_roi(params, anchors, m, advisor_cost)},
        COST_DOC: {"measured_inspection_time": render_measured_time(m), "measured_3d_time": render_3d_time(m),
                   "cost_per_inspection": render_cost(params, m), "pilot_budget": render_pilot(params, m)},
    }, []


def _parse_set(items: list[str]) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for it in items:
        name, _, raw = it.partition("=")
        vals = [float(v) for v in raw.split(",") if v.strip()]
        if len(vals) == 1:
            vals = vals * 3
        if len(vals) != 3:
            raise SystemExit(f"--set {it!r}: give one value or low,base,high")
        out[name.strip()] = (vals[0], vals[1], vals[2])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--print", dest="do_print", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=LOW,BASE,HIGH")
    ap.add_argument("--advisor-cost-uzs-per-hour", type=float, default=None)
    a = ap.parse_args(argv)
    blocks, missing = render_all(a.repo, _parse_set(a.set), a.advisor_cost_uzs_per_hour)
    if not blocks:
        print(f"roi_calc: measured inputs missing ({', '.join(missing)}); docs left untouched", file=sys.stderr)
        return 0
    stale = []
    for rel, named in blocks.items():
        if a.do_print:
            for name, body in named.items():
                print(f"===== {rel} :: {name}\n{body}")
            continue
        path = a.repo / rel
        old = path.read_text(encoding="utf-8")
        new = old
        for name, body in named.items():
            new = replace_block(new, name, body)
        if new != old:
            stale.append(rel)
            if not a.check:
                path.write_text(new, encoding="utf-8")
    if a.check and stale:
        print(f"roi_calc: stale generated blocks in {', '.join(stale)}; run python scripts/roi_calc.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
