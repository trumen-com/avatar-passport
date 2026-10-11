"""Compare panel-level rules on panels.mjs output, tuned on one half of the photos and reported on the other.
R1 'all joints differ': every joint's score >= t (the shipped rule).
R2 'consistent direction': project each joint's colour difference (this panel minus neighbour) on the panel's mean
   difference direction; the panel's strength is the smallest projection. A repaint is off the same way at every joint."""
import json, sys
import numpy as np
P = json.load(open(sys.argv[1]))
K = float(sys.argv[2]) if len(sys.argv) > 2 else 30
imgs = sorted({p["image"] for p in P}); rng = np.random.default_rng(1); rng.shuffle(imgs); half = set(imgs[: len(imgs) // 2])

def joint_score(f, plastic):
    return f["dE"] / (1 + f["dL"] / K) / (1.6 if plastic else 1)
def r1(fs, js):  # min score over joints
    return min(joint_score(f, j["plastic"]) for f, j in zip(fs, js))
def r2(fs, js):
    v = np.array([np.array(f["dab"]) * j["sgn"] / (1 + f["dL"] / K) / (1.6 if j["plastic"] else 1) for f, j in zip(fs, js)])
    m = v.mean(0); n = np.linalg.norm(m)
    if n == 0: return 0.0
    return float((v @ (m / n)).min())
RULES = {"R1 all joints differ": r1, "R2 consistent direction": r2}

def panel_values(P, rule, which):
    out = []
    for p in P:
        js = [j for j in p["joints"] if j["neg"]["rel"]]
        if len(js) < 2: continue
        if which == "neg": fs = [j["neg"] for j in js]; ok = all(f["rel"] for f in fs)
        else:
            fs = [next(x for x in j["pos"] if x["mag"] == which) for j in js]; ok = all(f["rel"] for f in fs)
        if ok: out.append(rule(fs, js))
    return np.array(out)

tuneP = [p for p in P if p["image"] in half]; testP = [p for p in P if p["image"] not in half]
for name, rule in RULES.items():
    for target_mag in (2, 3):
        thr = float(np.quantile(panel_values(tuneP, rule, target_mag), 0.10))   # catch 90% of simulated repaints on the tuning half
        neg = panel_values(testP, rule, "neg")
        caught = {m: float(np.mean(panel_values(testP, rule, m) >= thr)) for m in (1.5, 2, 3)}
        print(f"{name:26s} tuned to catch 90% of {target_mag} ΔE: thr {thr:.2f} | TEST: original panels called repainted {np.mean(neg >= thr):.1%} (n={len(neg)}) | caught " + " ".join(f"{m}ΔE {v:.0%}" for m, v in caught.items()))
