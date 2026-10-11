// Self-test: ΔE2000 against the Sharma et al. reference pairs, and the joint analysis on synthetic panels.
import { deltaE2000, analyzeJoint, rgbToLab, judgePanels, DEFAULTS } from "../engine.js";

const ref = [
  [[50, 2.6772, -79.7751], [50, 0, -82.7485], 2.0425],
  [[50, 3.1571, -77.2803], [50, 0, -82.7485], 2.8615],
  [[50, 0, 0], [50, -1, 2], 2.3669],
  [[60.2574, -34.0099, 36.2677], [60.4626, -34.1751, 39.4387], 1.2644],
  [[22.7233, 20.0904, -46.694], [23.0331, 14.973, -42.5619], 2.0373],
];
let ok = true;
for (const [a, b, e] of ref) { const d = deltaE2000(a, b); if (Math.abs(d - e) > 1e-3) { ok = false; console.log("FAIL", a, b, d, e); } }
console.log("deltaE2000 reference pairs:", ok ? "ok" : "FAILED");

// synthetic panel joint: two paints, a dark gap, a lighting gradient, noise, a sky reflection
function make({ W = 1200, H = 900, c1, c2, gapAt = 0.5, noise1 = 3, noise2 = 3, seed = 1 }) {
  let s = seed; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const g = () => { let u = 0; for (let i = 0; i < 6; i++) u += rnd(); return (u - 3); };
  const data = new Uint8ClampedArray(W * H * 4); const gx = Math.round(W * gapAt);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const left = x < gx, c = left ? c1 : c2, nz = left ? noise1 : noise2;
    let k = 0.85 + 0.3 * (y / H) - 0.1 * (x / W);           // smooth light falloff across the car
    if (Math.abs(x - gx) < 4) k *= 0.25;                     // gap shadow
    const sky = Math.exp(-(((x - W * 0.3) / 60) ** 2 + ((y - H * 0.3) / 200) ** 2)); // reflection streak
    const i = (y * W + x) * 4;
    for (let ch = 0; ch < 3; ch++) data[i + ch] = Math.min(255, c[ch] * k + 200 * sky + nz * g());
    data[i + 3] = 255;
  }
  return { data, width: W, height: H };
}
const silver = [168, 170, 172];
const cases = [
  ["same paint", silver, silver, 3, 3],
  ["slightly bluer repaint", silver, [164, 169, 178], 3, 3],
  ["same colour, rougher repaint", silver, silver, 3, 7],
  ["white vs white", [235, 236, 236], [235, 236, 236], 2, 2],
  ["white vs warmer white", [235, 236, 236], [238, 234, 226], 2, 2],
];
for (const [name, a, b, n1, n2] of cases) {
  const r = analyzeJoint(make({ c1: a, c2: b, gapAt: 0.53, noise1: n1, noise2: n2 }), { guide: 0.5 });
  console.log(name.padEnd(30), r.verdict.padEnd(8), "score", r.score, "dE", r.dE, "dTex", r.dTexture, "gap", r.gap.x.toFixed(3), r.gap.found, r.warnings.join(","));
}
const t = make({ c1: silver, c2: [164, 169, 178], gapAt: 0.5 });
const tr = { data: new Uint8ClampedArray(t.data.length), width: t.height, height: t.width };
for (let y = 0; y < t.height; y++) for (let x = 0; x < t.width; x++) { const i = (y * t.width + x) * 4, o = (x * t.height + y) * 4; for (let c = 0; c < 4; c++) tr.data[o + c] = t.data[i + c]; }
const rh = analyzeJoint(tr, { orientation: "horizontal", guide: 0.5 });
console.log("horizontal gap".padEnd(30), rh.verdict, rh.score, rh.gap.x.toFixed(3));
const t0 = performance.now(); for (let i = 0; i < 5; i++) analyzeJoint(t, {}); console.log("ms per joint", ((performance.now() - t0) / 5).toFixed(0));
const P = (a, b, s) => ({ a, b, result: { reliable: true, score: s } });
console.log(judgePanels([P("lf_fender", "lf_door", 2.4), P("lf_door", "lr_door", 2.1), P("lr_door", "lr_quarter", 0.3)]));
