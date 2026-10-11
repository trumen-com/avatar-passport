// Features for each sampling setting on a set of real joints: negatives (original paint) and simulated repaints.
// The scoring weights are then fitted offline (fit_score.py). Usage: node eval/sweep.mjs <joints dir> <ids.json> <out.json>
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import jpeg from "jpeg-js";
import { analyzeJoint, rgbToLab, labToRgb, DEFAULTS } from "../engine.js";

const [dir, idsPath, outPath] = process.argv.slice(2);
const all = JSON.parse(readFileSync(join(dir, "joints.json"), "utf8"));
const ids = new Set(JSON.parse(readFileSync(idsPath, "utf8")));
const joints = all.filter((j) => ids.has(j.id));
const CONFIGS = {
  base:        {},
  diffLow:     { diffuse: [0.05, 0.4] },
  diffLower:   { diffuse: [0.05, 0.3] },
  narrow:      { sampleWidth: 0.1 },
  narrowLow:   { sampleWidth: 0.1, diffuse: [0.05, 0.4] },
  narrowClose: { sampleWidth: 0.1, margin: 0.025, diffuse: [0.05, 0.4] },
  narrow5:     { sampleWidth: 0.1, diffuse: [0.05, 0.4], strips: 5 },
  oneStrip:    { sampleWidth: 0.1, diffuse: [0.05, 0.4], strips: 1 },
};
let seed = 11; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const gauss = () => { let u = 0; for (let i = 0; i < 6; i++) u += rnd(); return u - 3; };
function repaint(img, gapX, mag, ang, texSigma) {
  const { width: W, height: H, data } = img; const out = new Uint8ClampedArray(data);
  const x0 = Math.round(gapX * W) + 3, da = mag * Math.cos(ang), db = mag * Math.sin(ang);
  for (let y = 0; y < H; y++) for (let x = x0; x < W; x++) {
    const i = (y * W + x) * 4; const L = rgbToLab(data[i], data[i + 1], data[i + 2]);
    const k = Math.max(0, 1 - Math.max(0, L[0] - 85) / 15);
    const rgb = labToRgb([L[0], L[1] + da * k, L[2] + db * k]);
    for (let c = 0; c < 3; c++) out[i + c] = rgb[c] + (texSigma ? texSigma * gauss() : 0);
  }
  return { data: out, width: W, height: H };
}
const f = (r) => [r.dE, r.dL, r.dTexture, r.dGloss, r.reliable ? 1 : 0];
const rows = [];
for (const j of joints) {
  const raw = jpeg.decode(readFileSync(join(dir, "crops", j.id + ".jpg")), { useTArray: true, formatAsRGBA: true });
  const img = { data: raw.data, width: raw.width, height: raw.height };
  const variants = { neg: img, c2: repaint(img, j.gap_ann, 2, rnd() * 6.283, 0), c3: repaint(img, j.gap_ann, 3, rnd() * 6.283, 0), t3: repaint(img, j.gap_ann, 0, 0, 3) };
  const row = { id: j.id, image: j.image, kind: j.kind, plastic: j.plastic };
  for (const [cn, c] of Object.entries(CONFIGS)) {
    const opt = { ...DEFAULTS, ...c };
    row[cn] = Object.fromEntries(Object.entries(variants).map(([vn, im]) => [vn, f(analyzeJoint(im, { gapX: j.gap_ann }, opt))]));
  }
  rows.push(row);
}
writeFileSync(outPath, JSON.stringify(rows));
console.log("done", rows.length);
