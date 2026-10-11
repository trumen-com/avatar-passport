"""Fit the scoring (brightness tolerance angleK, texture and gloss weights) on sweep output.
Operating point: the threshold that catches 90% of simulated 3 ΔE repaints. We report how often original paint is
flagged at that threshold (false flags), and how many 2 ΔE colour and texture-only repaints are caught."""
import json, sys, itertools
import numpy as np
rows = json.load(open(sys.argv[1]))
TARGET = 0.90
def score(v, K, wt, wg, plastic):
    dE, dL, dT, dG, rel = v
    s = dE / (1 + dL / K) + wt * dT + wg * dG
    return s / (1.6 if plastic else 1.0)
def evaluate(cfg, K, wt, wg, rs, thr=None):
    use = [r for r in rs if r[cfg]["neg"][4]]
    S = {k: np.array([score(r[cfg][k], K, wt, wg, r["plastic"]) for r in use]) for k in ["neg", "c2", "c3", "t3"]}
    if thr is None: thr = float(np.quantile(S["c3"], 1 - TARGET))
    return {"thr": thr, "ff": float(np.mean(S["neg"] >= thr)), "c2": float(np.mean(S["c2"] >= thr)),
            "c3": float(np.mean(S["c3"] >= thr)), "t3": float(np.mean(S["t3"] >= thr)), "usable": len(use) / len(rs)}
if __name__ == "__main__":
    cfgs = [k for k in rows[0] if k not in ("id", "image", "kind", "plastic")]
    res = []
    for cfg, K, wt, wg in itertools.product(cfgs, [1e9, 30, 15, 8, 5, 3], [0, 0.3, 0.6, 1.2], [0, 0.4, 0.8]):
        e = evaluate(cfg, K, wt, wg, rows); res.append((e["ff"], cfg, K, wt, wg, e))
    res.sort(key=lambda x: x[0])
    print("current engine (base, no tolerance, wTex 1.2, wGloss 0.8):", {k: round(v, 3) for k, v in evaluate("base", 1e9, 1.2, 0.8, rows).items()})
    for cfg in cfgs:
        best = min((r for r in res if r[1] == cfg), key=lambda x: x[0])
        print(f"{cfg:12s} best ff {best[0]:.3f}  K={best[2]:g} wTex={best[3]} wGloss={best[4]}  " + " ".join(f"{k}={v:.3f}" for k, v in best[5].items()))
    print("\ntop 8 overall:")
    for r in res[:8]: print(f"  ff {r[0]:.3f} {r[1]:12s} K={r[2]:g} wTex={r[3]} wGloss={r[4]}  c2={r[5]['c2']:.3f} t3={r[5]['t3']:.3f} thr={r[5]['thr']:.2f} usable={r[5]['usable']:.2f}")
