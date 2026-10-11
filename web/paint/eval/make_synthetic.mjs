// Builds a small synthetic dataset in the app's export format, only to test the evaluation pipeline end to end.
// These numbers are NOT results: real results come from the team's photos with gauge readings.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import jpeg from "jpeg-js";
import { JOINTS, PANELS } from "../car.js";
const out = process.argv[2]; mkdirSync(join(out, "photos"), { recursive: true });
let seed = 3; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const g = () => { let u = 0; for (let i = 0; i < 6; i++) u += rnd(); return u - 3; };
function photo(c1, c2, n1, n2, W = 480, H = 640) {
  const d = Buffer.alloc(W * H * 4), gx = Math.round(W * (0.38 + 0.24 * rnd())), sx = rnd();
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const L = x < gx, c = L ? c1 : c2, nz = L ? n1 : n2; let k = 0.8 + 0.3 * (y / H) - 0.15 * (x / W) * (rnd() < 0.5 ? 1 : 1);
    if (Math.abs(x - gx) < 3) k *= 0.25;
    const sky = Math.exp(-(((x - W * sx) / 40) ** 2 + ((y - H * 0.35) / 200) ** 2)); const i = (y * W + x) * 4;
    for (let ch = 0; ch < 3; ch++) d[i + ch] = Math.max(0, Math.min(255, c[ch] * k + 180 * sky + nz * g())); d[i + 3] = 255;
  }
  return jpeg.encode({ data: d, width: W, height: H }, 92).data;
}
const colours = [[166, 169, 172], [236, 237, 236], [40, 42, 46], [120, 20, 24], [30, 60, 120]];
const cars = [], shots = [];
for (let ci = 0; ci < 8; ci++) {
  const base = colours[ci % colours.length]; const id = "syn" + ci;
  const rep = new Set(Object.keys(PANELS).filter(() => rnd() < 0.15));
  const shift = [g() * 3, g() * 3, g() * 3].map((v) => Math.round(v));   // repaint colour error, ~ΔE 1–4
  const gauge = {}; for (const p of Object.keys(PANELS)) gauge[p] = rep.has(p) ? 220 + Math.round(rnd() * 200) : 90 + Math.round(rnd() * 50);
  cars.push({ id, make: "Synthetic", model: "#" + ci, colour: "c" + (ci % 5), gauge });
  for (const j of JOINTS) {
    const ca = rep.has(j.a) ? base.map((v, k) => v + shift[k]) : base, cb = rep.has(j.b) ? base.map((v, k) => v + shift[k]) : base;
    const key = `${id}_${j.id}`; writeFileSync(join(out, "photos", key + ".jpg"), photo(ca, cb, rep.has(j.a) ? 5 : 3, rep.has(j.b) ? 5 : 3));
    shots.push({ key, car: id, joint: j.id, a: j.a, b: j.b, orientation: "vertical", gapX: null, photo: `photos/${key}.jpg` });
  }
}
writeFileSync(join(out, "labels.json"), JSON.stringify({ format: "trumen-paintcheck-1", synthetic: true, cars, shots }));
console.log(cars.length, "cars", shots.length, "photos");
