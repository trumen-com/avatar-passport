"""Summarise real_joints.mjs output: false flags on real original-paint joints, gap finding, detection of simulated repaints.
Photos are split in two halves (by image) so thresholds are set on one half and reported on the other."""
import json, sys, collections
import numpy as np
rows = json.load(open(sys.argv[1]))
FLAG = float(sys.argv[2]) if len(sys.argv) > 2 else 1.6
CHECK = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
fac = lambda r: 1.6 if r["plastic"] else 1.0
imgs = sorted({r["image"] for r in rows}); rng = np.random.default_rng(0); rng.shuffle(imgs)
half = set(imgs[: len(imgs) // 2])
tune = [r for r in rows if r["image"] in half]; test = [r for r in rows if r["image"] not in half]

def summary(rs, label):
    rel = [r for r in rs if r["neg"]["reliable"]]
    s = np.array([r["neg"]["score"] / fac(r) for r in rel])
    print(f"\n== {label}: {len(rs)} joints, {len(rel)} usable ({1-len(rel)/len(rs):.0%} retake)")
    print(f"  original paint flagged 'different' (>= {FLAG}): {np.mean(s >= FLAG):.1%}   'check' or worse (>= {CHECK}): {np.mean(s >= CHECK):.1%}")
    print("  score percentiles p50/p80/p90/p95:", np.percentile(s, [50, 80, 90, 95]).round(2))
    for kind in sorted({r["kind"] for r in rel}):
        k = np.array([r["neg"]["score"] / fac(r) for r in rel if r["kind"] == kind])
        print(f"    {kind:15s} n={len(k):4d} flagged {np.mean(k >= FLAG):5.1%}  p90 {np.percentile(k, 90):.2f}")
    return rel

rel = summary(rows, "ALL")
w = collections.Counter(x for r in rows for x in r["neg"]["warnings"]); print("  warnings:", dict(w))
for f in ["dE", "dTexture", "dGloss", "dL"]:
    v = np.array([r["neg"][f] for r in rel]); print(f"  {f:9s} p50 {np.median(v):.2f}  p90 {np.percentile(v, 90):.2f}")
# gap finder
err = np.array([abs(r["auto"]["gapX"] - r["gap_ann"]) for r in rows]); found = np.array([r["auto"]["found"] for r in rows])
print(f"\n== gap finder: within 3% of width {np.mean(err < 0.03):.1%}, within 6% {np.mean(err < 0.06):.1%}; says 'found' {found.mean():.1%}, of which correct (<3%) {np.mean(err[found] < 0.03):.1%}")
# simulated repaints
print("\n== simulated repaints (real photos; right panel colour shifted), caught at >= FLAG:")
for key in [(1, 0), (2, 0), (3, 0), (5, 0), (2, 3), (0, 3)]:
    p = [x for r in rows for x in r["pos"] if (x["mag"], x["tex"]) == key and x["reliable"] and r["neg"]["reliable"]]
    pr = [x for r in rows for x in r["pos"] if (x["mag"], x["tex"]) == key and x["reliable"] and r["neg"]["reliable"] for _ in [0]]
    sc = np.array([x["score"] for x in p])
    print(f"  colour shift {key[0]} ΔE, texture noise {key[1]}: caught {np.mean(sc >= FLAG):.1%}  (median dE {np.median([x['dE'] for x in p]):.2f})")
