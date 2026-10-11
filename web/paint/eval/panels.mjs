// Panel-level test on real photos. For every panel with two or more measurable joints in one photo:
//  - as photographed (original paint): would the panel be called repainted?  -> false calls
//  - with the whole panel's colour shifted by a known amount (the same shift at all its joints)  -> detection
// Writes per-joint features; panel rules are compared in panel_rules.py. Usage: node eval/panels.mjs <joints dir> <out.json> [config json]
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import jpeg from "jpeg-js";
import { analyzeJoint, rgbToLab, labToRgb, DEFAULTS } from "../engine.js";
const [dir, outPath, cfgJson] = process.argv.slice(2);
const opt = { ...DEFAULTS, ...(cfgJson ? JSON.parse(cfgJson) : {}) };
const joints = JSON.parse(readFileSync(join(dir, "joints.json"), "utf8"));
const byPanel = new Map();
for (const j of joints) for (const p of [j.a, j.b]) { const k = j.image + "|" + p; if (!byPanel.has(k)) byPanel.set(k, []); byPanel.get(k).push(j); }
const panels = [...byPanel.entries()].filter(([, js]) => js.length >= 2);
let seed = 5; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const MAGS = [1.5, 2, 3];
function shiftSide(img, gapX, right, da, db) {
  const { width: W, height: H, data } = img; const out = new Uint8ClampedArray(data); const g = Math.round(gapX * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (right ? x < g + 3 : x > g - 3) continue;
    const i = (y * W + x) * 4; const L = rgbToLab(data[i], data[i + 1], data[i + 2]); const k = Math.max(0, 1 - Math.max(0, L[0] - 85) / 15);
    const rgb = labToRgb([L[0], L[1] + da * k, L[2] + db * k]); out[i] = rgb[0]; out[i + 1] = rgb[1]; out[i + 2] = rgb[2];
  }
  return { data: out, width: W, height: H };
}
const cache = new Map();
const load = (id) => { if (!cache.has(id)) { const r = jpeg.decode(readFileSync(join(dir, "crops", id + ".jpg")), { useTArray: true, formatAsRGBA: true }); cache.set(id, { data: r.data, width: r.width, height: r.height }); } return cache.get(id); };
const F = (r) => ({ dE: r.dE, dL: r.dL, dT: r.dTexture, dG: r.dGloss, dab: r.dab, rel: r.reliable });
const out = [];
for (const [key, js] of panels) {
  const [image, panel] = key.split("|");
  const rec = { image, panel, plasticPanel: /bumper/i.test(panel), joints: [] };
  const shifts = MAGS.map((m) => { const a = rnd() * 6.283; return [m, m * Math.cos(a), m * Math.sin(a)]; });
  for (const j of js) {
    const img = load(j.id);
    const right = j.b === panel;     // is the panel on the right side of this crop?
    // sign: features are measured as right minus left; flip so they always read "this panel minus its neighbour"
    const sgn = right ? 1 : -1;
    const neg = F(analyzeJoint(img, { gapX: j.gap_ann }, opt));
    const pos = shifts.map(([m, da, db]) => ({ mag: m, ...F(analyzeJoint(shiftSide(img, j.gap_ann, right, da, db), { gapX: j.gap_ann }, opt)) }));
    rec.joints.push({ id: j.id, kind: j.kind, plastic: j.plastic, sgn, neg, pos });
  }
  out.push(rec);
}
writeFileSync(outPath, JSON.stringify(out));
console.log("panels with >= 2 joints:", out.length, "joints measured:", out.reduce((s, p) => s + p.joints.length, 0));
