// Final check of the shipped settings on the held-out photos (never used for tuning).
// Joint level: original paint vs simulated repaints of 2 and 3 ΔE on one side.
// Panel level: panels with two or more joints, original vs the whole panel shifted (same shift at every joint).
// Usage: node eval/final_eval.mjs <joints dir> <test ids.json> <out.json>
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import jpeg from "jpeg-js";
import { analyzeJoint, judgePanels, rgbToLab, labToRgb, DEFAULTS } from "../engine.js";
import { BUMPER_FACTOR, LID_FACTOR } from "../car.js";

const [dir, idsPath, outPath] = process.argv.slice(2);
const ids = new Set(JSON.parse(readFileSync(idsPath, "utf8")));
const joints = JSON.parse(readFileSync(join(dir, "joints.json"), "utf8")).filter((j) => ids.has(j.id));
const factorOf = (j) => (/Hood|Trunk/.test(j.a + j.b) ? LID_FACTOR : j.plastic ? BUMPER_FACTOR : 1);
let seed = 23; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
function shiftSide(img, gapX, right, da, db) {
  const { width: W, height: H, data } = img; const out = new Uint8ClampedArray(data); const g = Math.round(gapX * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (right ? x < g + 3 : x > g - 3) continue;
    const i = (y * W + x) * 4; const L = rgbToLab(data[i], data[i + 1], data[i + 2]); const k = Math.max(0, 1 - Math.max(0, L[0] - 85) / 15);
    const rgb = labToRgb([L[0], L[1] + da * k, L[2] + db * k]); out[i] = rgb[0]; out[i + 1] = rgb[1]; out[i + 2] = rgb[2];
  }
  return { data: out, width: W, height: H };
}
const load = (id) => { const r = jpeg.decode(readFileSync(join(dir, "crops", id + ".jpg")), { useTArray: true, formatAsRGBA: true }); return { data: r.data, width: r.width, height: r.height }; };
const SIDE = new Set(["fender-door", "door-door", "door-quarter"]);

// ---- joint level
const J = { all: { n: 0, ff: 0, c2: 0, c3: 0 }, side: { n: 0, ff: 0, c2: 0, c3: 0 }, corner: { n: 0, ff: 0, c2: 0, c3: 0 }, total: joints.length, retake: 0, gapAuto: 0 };
const imgCache = new Map();
for (const j of joints) {
  const img = load(j.id); imgCache.set(j.id, img);
  const f = factorOf(j);
  const neg = analyzeJoint(img, { gapX: j.gap_ann, factor: f });
  const auto = analyzeJoint(img, { guide: 0.5, factor: f });
  if (Math.abs(auto.gap.x - j.gap_ann) < 0.03) J.gapAuto++;
  if (!neg.reliable) { J.retake++; continue; }
  const a2 = rnd() * 6.283, a3 = rnd() * 6.283;
  const p2 = analyzeJoint(shiftSide(img, j.gap_ann, true, 2 * Math.cos(a2), 2 * Math.sin(a2)), { gapX: j.gap_ann, factor: f });
  const p3 = analyzeJoint(shiftSide(img, j.gap_ann, true, 3 * Math.cos(a3), 3 * Math.sin(a3)), { gapX: j.gap_ann, factor: f });
  for (const g of ["all", SIDE.has(j.kind) ? "side" : "corner"]) {
    J[g].n++; J[g].ff += neg.verdict === "differs"; J[g].c2 += p2.verdict === "differs"; J[g].c3 += p3.verdict === "differs";
  }
}
for (const g of ["all", "side", "corner"]) for (const k of ["ff", "c2", "c3"]) J[g][k] = +(J[g][k] / J[g].n).toFixed(3);
J.retake = +(J.retake / J.total).toFixed(3); J.gapAuto = +(J.gapAuto / J.total).toFixed(3);

// ---- panel level
const byPanel = new Map();
for (const j of joints) for (const p of [j.a, j.b]) { const k = j.image + "|" + p; if (!byPanel.has(k)) byPanel.set(k, []); byPanel.get(k).push(j); }
const P = { n: 0, falseRepainted: 0, falseAnyFlag: 0, caught: { 1.5: 0, 2: 0, 3: 0 }, caughtOrSuspect: { 1.5: 0, 2: 0, 3: 0 } };
for (const [key, js] of byPanel) {
  if (js.length < 2) continue;
  const panel = key.split("|")[1];
  const run = (shift) => judgePanels(js.map((j) => {
    const right = j.b === panel, img = imgCache.get(j.id);
    const im = shift ? shiftSide(img, j.gap_ann, right, shift[0], shift[1]) : img;
    return { a: j.a, b: j.b, factor: factorOf(j), result: analyzeJoint(im, { gapX: j.gap_ann, factor: factorOf(j) }) };
  }))[panel];
  const base = run(null);
  if (!base || base.joints < 2) continue;
  P.n++; P.falseRepainted += base.status === "repainted"; P.falseAnyFlag += base.status === "repainted" || base.status === "suspect";
  for (const m of [1.5, 2, 3]) {
    const a = rnd() * 6.283, r = run([m * Math.cos(a), m * Math.sin(a)]);
    P.caught[m] += r?.status === "repainted"; P.caughtOrSuspect[m] += r?.status === "repainted" || r?.status === "suspect";
  }
}
for (const k of ["falseRepainted", "falseAnyFlag"]) P[k] = +(P[k] / P.n).toFixed(3);
for (const m of [1.5, 2, 3]) { P.caught[m] = +(P.caught[m] / P.n).toFixed(3); P.caughtOrSuspect[m] = +(P.caughtOrSuspect[m] / P.n).toFixed(3); }
const res = { settings: DEFAULTS, factors: { bumper: BUMPER_FACTOR, lid: LID_FACTOR }, photos: new Set(joints.map((j) => j.image)).size, joints: J, panels: P };
writeFileSync(outPath, JSON.stringify(res, null, 1));
console.log(JSON.stringify({ joints: J, panels: P }, null, 1));
