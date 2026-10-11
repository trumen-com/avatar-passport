// Run the engine on real joints cut from the car-parts dataset (see extract_joints.py).
// For each crop: (1) gap from the annotation, original paint on both sides (negative);
// (2) automatic gap finding, compared with the annotated gap;
// (3) simulated repaints on the right-hand panel: colour shifted by a known amount, optionally with rougher texture.
// Writes one row per crop with all features, for analysis in analyze_real.py.
// Usage: node eval/real_joints.mjs <joints dir> <out.json>
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import jpeg from "jpeg-js";
import { analyzeJoint, rgbToLab, labToRgb } from "../engine.js";

const [dir, outPath] = process.argv.slice(2);
const joints = JSON.parse(readFileSync(join(dir, "joints.json"), "utf8"));
const MAGS = [1, 2, 3, 5];   // colour shift, ΔE76 units in a/b (chroma and hue), lightness kept
let seed = 11; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const gauss = () => { let u = 0; for (let i = 0; i < 6; i++) u += rnd(); return u - 3; };

function repaint(img, gapX, mag, ang, texSigma) {
  const { width: W, height: H, data } = img;
  const out = new Uint8ClampedArray(data);
  const x0 = Math.round(gapX * W) + 3;   // leave the gap shadow itself alone
  const da = mag * Math.cos(ang), db = mag * Math.sin(ang);
  for (let y = 0; y < H; y++) for (let x = x0; x < W; x++) {
    const i = (y * W + x) * 4;
    const L = rgbToLab(data[i], data[i + 1], data[i + 2]);
    // a repaint shifts the paint colour; reflections of the sky stay sky-coloured, so scale the shift down in highlights
    const k = Math.max(0, 1 - Math.max(0, L[0] - 85) / 15);
    const rgb = labToRgb([L[0], L[1] + da * k, L[2] + db * k]);
    for (let c = 0; c < 3; c++) out[i + c] = rgb[c] + (texSigma ? texSigma * gauss() : 0);
  }
  return { data: out, width: W, height: H };
}
const pick = (r) => ({ score: r.score, dE: r.dE, dTexture: r.dTexture, dGloss: r.dGloss, glossCompared: r.glossCompared, dL: r.dL, reliable: r.reliable, warnings: r.warnings });

const rows = [];
const t0 = Date.now();
for (const j of joints) {
  const raw = jpeg.decode(readFileSync(join(dir, "crops", j.id + ".jpg")), { useTArray: true, formatAsRGBA: true });
  const img = { data: raw.data, width: raw.width, height: raw.height };
  const neg = analyzeJoint(img, { gapX: j.gap_ann });
  const auto = analyzeJoint(img, { guide: 0.5 });
  const pos = [];
  for (const m of MAGS) {
    const ang = rnd() * 2 * Math.PI;
    pos.push({ mag: m, tex: 0, ...pick(analyzeJoint(repaint(img, j.gap_ann, m, ang, 0), { gapX: j.gap_ann })) });
  }
  pos.push({ mag: 2, tex: 3, ...pick(analyzeJoint(repaint(img, j.gap_ann, 2, rnd() * 2 * Math.PI, 3), { gapX: j.gap_ann })) });
  pos.push({ mag: 0, tex: 3, ...pick(analyzeJoint(repaint(img, j.gap_ann, 0, 0, 3), { gapX: j.gap_ann })) });
  rows.push({ ...j, neg: { ...pick(neg), sides: neg.sides.map((s) => ({ lab: s.lab, rawLab: s.rawLab, texture: s.texture, spread: s.spread, n: s.n })) },
    auto: { gapX: auto.gap.x, found: auto.gap.found, strength: auto.gap.strength, ...pick(auto) }, pos });
  if (rows.length % 250 === 0) console.log(rows.length, "joints", ((Date.now() - t0) / 1000).toFixed(0) + "s");
}
writeFileSync(outPath, JSON.stringify(rows));
console.log("done", rows.length, ((Date.now() - t0) / 1000).toFixed(0) + "s");
