// Evaluate Paint Check against paint-gauge ground truth.
// Usage: node eval/run.mjs <data dir> [--gauge 200] [--out results.json]
// Data dir: either the app's .zip export unpacked (labels.json + photos/), or a pull from the artifact store
// (cars/*.json, shots/*.json, photos/<asset id>.jpg).
// Ground truth: a panel is repainted when its gauge reading is above --gauge µm (default 200).
// A joint is positive when exactly one of its two panels is repainted; joints with both repainted, or a missing reading, are left out.
import { readFileSync, readdirSync, existsSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import jpeg from "jpeg-js";
import { analyzeJoint, judgePanels, DEFAULTS } from "../engine.js";
import { JOINTS, jointFactor } from "../car.js";

const args = process.argv.slice(2);
const dir = args[0];
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const GAUGE = +opt("--gauge", 200);
const OUT = opt("--out", null);

function load() {
  if (existsSync(join(dir, "labels.json"))) {
    const L = JSON.parse(readFileSync(join(dir, "labels.json"), "utf8"));
    return { cars: L.cars, shots: L.shots.map((s) => ({ ...s, file: join(dir, s.photo) })) };
  }
  const rd = (sub) => existsSync(join(dir, sub)) ? readdirSync(join(dir, sub)).filter((f) => f.endsWith(".json")).map((f) => { const d = JSON.parse(readFileSync(join(dir, sub, f), "utf8")); return d.data || d; }) : [];
  const cars = rd("cars"), shots = rd("shots");
  return { cars, shots: shots.map((s) => ({ ...s, file: join(dir, "photos", s.photo + ".jpg") })) };
}

const { cars, shots } = load();
const carById = Object.fromEntries(cars.map((c) => [c.id, c]));
const rows = [];
for (const s of shots) {
  const c = carById[s.car]; const j = JOINTS.find((x) => x.id === s.joint);
  if (!c || !j || !existsSync(s.file)) { console.warn("skip", s.car, s.joint, !c ? "no car" : !existsSync(s.file) ? "no photo" : "joint"); continue; }
  const img = jpeg.decode(readFileSync(s.file), { useTArray: true, formatAsRGBA: true });
  const r = analyzeJoint({ data: img.data, width: img.width, height: img.height }, { orientation: s.orientation || "vertical", gapX: s.gapX ?? null, factor: jointFactor(j) });
  const ga = c.gauge?.[j.a], gb = c.gauge?.[j.b];
  const ra = ga != null ? ga > GAUGE : null, rb = gb != null ? gb > GAUGE : null;
  const truth = ra == null || rb == null ? null : ra && rb ? "both" : ra || rb ? 1 : 0;
  rows.push({ car: s.car, joint: s.joint, a: j.a, b: j.b, ga, gb, truth, score: r.score / jointFactor(j), dE: r.dE, dTexture: r.dTexture, dGloss: r.dGloss, dL: r.dL, reliable: r.reliable, warnings: r.warnings, result: r, factor: jointFactor(j) });
}

// ---------------------------------------------------------------- joint level
const lab = rows.filter((r) => r.truth === 0 || r.truth === 1);
const usable = lab.filter((r) => r.reliable);
function auc(xs) {
  const pos = xs.filter((r) => r.truth === 1).map((r) => r.score), neg = xs.filter((r) => r.truth === 0).map((r) => r.score);
  if (!pos.length || !neg.length) return null;
  let s = 0; for (const p of pos) for (const n of neg) s += p > n ? 1 : p === n ? 0.5 : 0;
  return s / (pos.length * neg.length);
}
const rate = (xs, thr) => {
  const pos = xs.filter((r) => r.truth === 1), neg = xs.filter((r) => r.truth === 0);
  return { caught: pos.filter((r) => r.score >= thr).length, positives: pos.length, falseFlags: neg.filter((r) => r.score >= thr).length, negatives: neg.length };
};
// bootstrap over cars, so photos of one car stay together
function bootstrapAUC(xs, n = 1000) {
  const ids = [...new Set(xs.map((r) => r.car))]; const by = Object.fromEntries(ids.map((id) => [id, xs.filter((r) => r.car === id)]));
  const v = []; let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < n; i++) { const smp = []; for (let k = 0; k < ids.length; k++) smp.push(...by[ids[Math.floor(rnd() * ids.length)]]); const a = auc(smp); if (a != null) v.push(a); }
  v.sort((a, b) => a - b); return v.length > 20 ? [v[Math.floor(v.length * 0.025)], v[Math.floor(v.length * 0.975)]] : null;
}
// threshold chosen on the other cars only (leave one car out), so the reported rate is not tuned on itself
function looThreshold(xs) {
  const ids = [...new Set(xs.map((r) => r.car))]; let caught = 0, pos = 0, ff = 0, neg = 0;
  for (const id of ids) {
    const train = xs.filter((r) => r.car !== id), test = xs.filter((r) => r.car === id);
    const thr = bestThreshold(train);
    const k = rate(test, thr); caught += k.caught; pos += k.positives; ff += k.falseFlags; neg += k.negatives;
  }
  return { caught, positives: pos, falseFlags: ff, negatives: neg };
}
function bestThreshold(xs) {
  const cands = [...new Set(xs.map((r) => r.score))].sort((a, b) => a - b);
  let best = DEFAULTS.flagAt, bj = -Infinity;
  for (const t of cands) { const k = rate(xs, t); if (!k.positives || !k.negatives) continue; const J = k.caught / k.positives - k.falseFlags / k.negatives; if (J > bj) { bj = J; best = t; } }
  return best;
}
const feat = (name) => { const s = usable.map((r) => ({ ...r, score: r[name] })); return auc(s); };

// ---------------------------------------------------------------- panel level
const panelRes = { caught: 0, repainted: 0, falseCalls: 0, original: 0, perCar: [] };
for (const c of cars) {
  const js = rows.filter((r) => r.car === c.id && r.reliable).map((r) => ({ a: r.a, b: r.b, factor: r.factor, result: r.result }));
  const st = judgePanels(js);
  for (const [p, v] of Object.entries(st)) {
    const g = c.gauge?.[p]; if (g == null) continue;
    const isR = g > GAUGE, called = v.status === "repainted";
    if (isR) { panelRes.repainted++; if (called || v.status === "suspect") panelRes.caught++; } else { panelRes.original++; if (called) panelRes.falseCalls++; }
  }
  panelRes.perCar.push({ car: c.id, make: c.make, model: c.model, colour: c.colour, called: Object.entries(st).filter(([, v]) => v.status === "repainted").map(([k]) => k), gaugeRepainted: Object.entries(c.gauge || {}).filter(([, g]) => g > GAUGE).map(([k]) => k) });
}

const out = {
  gaugeThreshold: GAUGE, cars: cars.length, photos: rows.length, labelledJoints: lab.length, usableJoints: usable.length,
  retakeRate: lab.length ? +(1 - usable.length / lab.length).toFixed(3) : null,
  auc: auc(usable), aucCI: bootstrapAUC(usable),
  featureAUC: { colour: feat("dE"), texture: feat("dTexture"), gloss: feat("dGloss"), brightness: feat("dL") },
  atShippedThreshold: { threshold: DEFAULTS.flagAt, ...rate(usable, DEFAULTS.flagAt) },
  bestThresholdAllData: bestThreshold(usable),
  leaveOneCarOut: looThreshold(usable),
  panels: panelRes,
  joints: rows.map(({ result, ...r }) => r),
};
console.log(JSON.stringify({ ...out, joints: undefined, panels: { ...panelRes, perCar: undefined } }, null, 1));
if (OUT) writeFileSync(OUT, JSON.stringify(out, null, 1));
