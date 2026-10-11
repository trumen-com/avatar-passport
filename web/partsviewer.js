// Parts 3D viewer: a part-labelled point cloud you can take apart, recolour by part, auto-rotate and tap.
// Ported from the "Camry Parts Scan" viewer by Team Trumen (WIUT Hackathon 2026), plain WebGL 1, no libraries.
// Data: /api/inspections/<id>/parts3d.json (see src/avtopassport/recon3d/parts_cloud.py for the format).

import { partSearch } from "./i18n.js";

function dec(b) {
  const s = atob(b);
  const u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return u;
}

function hsl(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = (n) => {
    const k = (n + h / 30) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

const GROUPS = [
  ["Front", ["front-bumper", "grille", "hood", "headlight", "windshield"]],
  ["Left side", null],
  ["Right side", null],
  ["Rear", ["back-bumper", "trunk", "tail-light", "back-windshield"]],
  ["Top", ["roof"]],
];

function groupOf(p) {
  const b = p.key.replace(/^(left|right)_/, "");
  if (p.side === "left") return "Left side";
  if (p.side === "right") return "Right side";
  for (const [g, ks] of GROUPS) if (ks && ks.includes(b)) return g;
  return "Top";
}

// "take apart" offsets in metres (car frame: x forward, y left, z up), rules per part type
function offsetOf(p, centre) {
  const [x, y, z] = centre;
  const b = p.key.replace(/^(left|right)_/, "");
  const sy = p.side === "left" ? 1 : p.side === "right" ? -1 : Math.sign(y) || 1;
  let o = [x * 0.55, y * 2.2, (z - 0.65) * 1.3];
  if (b === "hood") o = [0.6, 0, 0.9];
  if (b === "roof") o = [0, 0, 1.0];
  if (b === "trunk") o = [-0.5, 0, 0.8];
  if (b === "windshield") o = [0.35, 0, 0.75];
  if (b === "back-windshield") o = [-0.35, 0, 0.7];
  if (b === "front-bumper") o = [0.9, 0, -0.05];
  if (b === "back-bumper") o = [-0.9, 0, -0.05];
  if (b === "grille") o = [1.25, 0, 0.05];
  if (b.endsWith("wheel")) o = [x * 0.15, sy * 1.3, -0.05];
  if (b === "mirror") o = [0.1, sy * 1.5, 0.25];
  if (b.endsWith("window")) o = [0, sy * 1.0, 0.45];
  if (b === "rocker-panel") o = [0, sy * 1.6, -0.35];
  return o;
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// Damage colours by severity (same order as the 360° legend); undamaged parts are drawn pale in Damage mode.
const SEV_RGB = { severe: [0.83, 0.2, 0.12], moderate: [0.87, 0.42, 0.05], minor: [0.79, 0.6, 0.02] };
const SEV_RANK = { severe: 3, moderate: 2, minor: 1 };

// The finding's part name ("left_back-door") is matched to the 3D part key; sides come from different sources
// (2D label vs the car's axes), so an exact key wins and the plain part type is the fallback.
function damageFor(p, damage) {
  if (!damage) return null;
  if (damage[p.key]) return damage[p.key];
  const b = p.key.replace(/^(left|right)_/, "");
  return p.side === "center" ? damage[b] || null : null;
}

// opts: { damage: {partKey: {severity, items: [text]}}, mode: 0 photo | 1 part | 2 damage, compact, onSelect(part|null) }
export function createPartsViewer(container, D, opts = {}) {
  const M = D.meta || {};
  const PARTS = Array.isArray(M.parts) ? M.parts : [];
  if (!PARTS.length || !D.n) throw new Error("the parts model has no parts");
  const POS = new Int16Array(dec(D.p).buffer);
  const COL = dec(D.c);
  const LAB = dec(D.l);
  const N = D.n;
  const S = D.S;
  const metric = Boolean(M.metric);

  // part centres in metres from the points themselves (the payload has them only when metric)
  const sums = PARTS.map(() => [0, 0, 0, 0]);
  for (let i = 0; i < N; i++) {
    const s = sums[LAB[i]];
    if (!s) continue;
    s[0] += POS[3 * i]; s[1] += POS[3 * i + 1]; s[2] += POS[3 * i + 2]; s[3] += 1;
  }
  let zmin = Infinity;
  for (let i = 0; i < N; i++) zmin = Math.min(zmin, (POS[3 * i + 2] / 32767) * S);
  const CENTRE = sums.map((s) => (s[3] ? [s[0] / s[3] / 32767 * S, s[1] / s[3] / 32767 * S, s[2] / s[3] / 32767 * S - zmin] : [0, 0, 0]));
  const PAL = PARTS.map((p, i) => hsl((i * 137.508) % 360, 0.62, 0.55));
  const OFF = PARTS.map((p, i) => offsetOf(p, CENTRE[i]));
  const lift = -zmin; // the scan puts the ground at z = 0; keep the view target at mid-height
  const DMG = PARTS.map((p) => {
    const d = damageFor(p, opts.damage);
    return d && SEV_RGB[d.severity] ? [...SEV_RGB[d.severity], 1] : [0, 0, 0, 0];
  });
  const hasDamage = DMG.some((d) => d[3] > 0);
  let sel = -1;
  let mode = opts.mode ?? 0;
  let ex = 0;
  let yaw = -0.75;
  let pitch = 0.32;
  let dist = 5.0;
  let spin = true;

  container.classList.add("pv");
  container.classList.toggle("pv-compact", Boolean(opts.compact));
  container.innerHTML = `
    <div class="pv-grid">
      <div class="pv-main">
        <div class="pv-stage"><canvas></canvas><div class="pv-hint">Drag to turn · pinch or scroll to zoom · tap a part</div></div>
        <div class="pv-controls">
          <label class="pv-ex">Take apart <input type="range" min="0" max="1" step="0.01" value="0"></label>
          <span class="pv-seg" role="group" aria-label="Colour"><button type="button" data-mode="0" aria-pressed="${mode === 0}">Photo</button><button type="button" data-mode="1" aria-pressed="${mode === 1}">Parts</button>${opts.damage ? `<button type="button" data-mode="2" aria-pressed="${mode === 2}">Damage</button>` : ""}</span>
          <button type="button" class="pv-spin" aria-pressed="true">Auto-rotate</button>
          <button type="button" class="pv-reset">Reset</button>
        </div>
      </div>
      <aside class="pv-side"${opts.compact ? " hidden" : ""}>
        <div class="pv-detail"></div>
        <div class="pv-list"></div>
      </aside>
    </div>`;
  const cv = container.querySelector("canvas");
  const st = container.querySelector(".pv-stage");
  const list = container.querySelector(".pv-list");
  const detail = container.querySelector(".pv-detail");
  const exEl = container.querySelector(".pv-ex input");
  const spinB = container.querySelector(".pv-spin");

  // ---------- list + detail
  const hex = (c) => "#" + c.map((v) => Math.round(v * 255).toString(16).padStart(2, "0")).join("");
  let html = "";
  for (const [g] of GROUPS) {
    const ps = PARTS.filter((p) => groupOf(p) === g);
    if (!ps.length) continue;
    html += `<h4>${g}</h4>`;
    for (const p of ps) {
      const d = damageFor(p, opts.damage);
      const right = d ? `<span class="badge sev-${esc(d.severity)}">${esc(d.severity)}</span>` : metric && p.size_m ? `${p.size_m[0].toFixed(2)} m` : "";
      html += `<button type="button" data-id="${p.id}" aria-pressed="false"><span class="pv-nm"><span class="pv-sw" style="background:${hex(PAL[p.id])}"></span><span>${esc(p.name)}</span></span><span class="pv-ct">${right}</span></button>`;
    }
  }
  list.innerHTML = html;
  list.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-id]");
    if (b) select(+b.dataset.id === sel ? -1 : +b.dataset.id);
  });

  function renderDetail() {
    if (sel < 0) {
      const nd = PARTS.filter((p) => damageFor(p, opts.damage)).length;
      detail.innerHTML = `<strong>${PARTS.length} parts${opts.damage ? ` · ${nd} with damage` : ""}</strong><p class="small muted">Tap a part in 3D or in the list.${metric ? "" : " Sizes need the car's published length, so they are not shown."}</p>`;
      return;
    }
    const p = PARTS[sel];
    const rows = [];
    if (metric && p.size_m) rows.push(`<dt>Size (L × W × H)</dt><dd>${p.size_m.map((v) => v.toFixed(2)).join(" × ")} m</dd>`);
    rows.push(`<dt>Points</dt><dd>${p.points.toLocaleString()}</dd><dt>Seen in</dt><dd>${p.views}+ views</dd>`);
    const d = damageFor(p, opts.damage);
    if (d) rows.unshift(`<dt>Damage</dt><dd>${(d.items || [d.severity]).map(esc).join("<br>")}</dd>`);
    const s = partSearch(p.key, opts.car, p.name);
    const links = `<a href="${esc(s.olx)}" target="_blank" rel="noopener" title="${esc(s.query)}">OLX.uz</a><a href="${esc(s.google)}" target="_blank" rel="noopener">Google</a>`;
    detail.innerHTML = `<strong>${esc(p.name)}</strong><dl class="kv">${rows.join("")}</dl>${links ? `<div class="pv-links"><span class="small muted">Find the part:</span>${links}</div>` : ""}`;
  }

  function select(i) {
    sel = i;
    if (opts.onSelect) opts.onSelect(i >= 0 ? { ...PARTS[i], damage: damageFor(PARTS[i], opts.damage) } : null);
    for (const b of list.querySelectorAll("button")) b.setAttribute("aria-pressed", String(+b.dataset.id === i));
    renderDetail();
    draw();
  }
  renderDetail();

  // ---------- WebGL
  const gl = cv.getContext("webgl", { antialias: true, preserveDrawingBuffer: true });
  if (!gl) st.insertAdjacentHTML("beforeend", '<div class="pv-nogl">WebGL is off in this browser, so the 3D view cannot draw. The parts list still works.</div>');
  const uni = {};
  if (gl) {
    const NP = PARTS.length;
    const vs = `attribute vec3 p;attribute vec3 c;attribute float l;uniform mat4 mvp;uniform vec3 off[${NP}];uniform vec3 pal[${NP}];uniform vec4 dmg[${NP}];uniform float ex,sel,mode,ps,S,lift;varying vec3 vc;
      void main(){int i=int(l+0.5);vec3 o=vec3(0.);vec3 pc=vec3(0.);vec4 dg=vec4(0.);for(int k=0;k<${NP};k++){if(k==i){o=off[k];pc=pal[k];dg=dmg[k];}}
      vec3 w=p*S+o*ex;w.z+=lift;gl_Position=mvp*vec4(w.y*-1.0,w.z,w.x,1.0);vec3 col=mode>1.5?(dg.w>0.5?dg.rgb:mix(c,vec3(0.88),0.7)):(mode>0.5?pc:c);float on=(sel<0.0||abs(l-sel)<0.5)?1.0:0.0;
      vc=mix(col*0.35+vec3(0.5)*0.25,col,on);gl_PointSize=ps*(on>0.5&&sel>=0.0?1.5:1.0)/gl_Position.w;}`;
    const fs = "precision mediump float;varying vec3 vc;void main(){vec2 d=gl_PointCoord-0.5;if(dot(d,d)>0.25)discard;gl_FragColor=vec4(vc,1.0);}";
    const sh = (t, s) => {
      const o = gl.createShader(t);
      gl.shaderSource(o, s);
      gl.compileShader(o);
      if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(o));
      return o;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    gl.useProgram(prog);
    const buf = (data, name, size, type, norm) => {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      const a = gl.getAttribLocation(prog, name);
      gl.enableVertexAttribArray(a);
      gl.vertexAttribPointer(a, size, type, norm, 0, 0);
    };
    buf(POS, "p", 3, gl.SHORT, true);
    buf(COL, "c", 3, gl.UNSIGNED_BYTE, true);
    buf(new Float32Array(LAB), "l", 1, gl.FLOAT, false);
    for (const n of ["mvp", "off", "pal", "dmg", "ex", "sel", "mode", "ps", "S", "lift"]) uni[n] = gl.getUniformLocation(prog, n);
    gl.uniform4fv(uni.dmg, new Float32Array(DMG.flat()));
    gl.uniform3fv(uni.off, new Float32Array(OFF.flat()));
    gl.uniform3fv(uni.pal, new Float32Array(PAL.flat()));
    gl.uniform1f(uni.S, S);
    gl.uniform1f(uni.lift, lift);
    gl.enable(gl.DEPTH_TEST);
  }

  function mats() {
    const W = cv.width;
    const H = cv.height;
    const f = 1 / Math.tan(0.42);
    const a = W / H;
    const n = 0.1;
    const fa = 60;
    const P = [f / a, 0, 0, 0, 0, f, 0, 0, 0, 0, (fa + n) / (n - fa), -1, 0, 0, (2 * fa * n) / (n - fa), 0];
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const dd = dist * (1 + 0.45 * ex) * (a < 1 ? 1.35 : 1); // narrow (phone) stages need a step back
    const eye = [dd * cp * sy, 0.55 + dd * sp, dd * cp * cy];
    const tg = [0, 0.55, 0];
    let z = [eye[0] - tg[0], eye[1] - tg[1], eye[2] - tg[2]];
    const zl = Math.hypot(...z);
    z = z.map((v) => v / zl);
    let x = [z[2], 0, -z[0]];
    const xl = Math.hypot(...x);
    x = x.map((v) => v / xl);
    const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
    const V = [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]), -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]), -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]), 1];
    const R = new Array(16).fill(0);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++) R[j * 4 + i] += P[k * 4 + i] * V[j * 4 + k];
    return R;
  }

  function clearColour() {
    const css = getComputedStyle(container).getPropertyValue("--pv-stage").trim() || "#eef1f5";
    const c = parseInt(css.replace("#", ""), 16);
    return [((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255];
  }

  function draw() {
    if (!gl || !cv.width) return;
    const [r, g, b] = clearColour();
    gl.viewport(0, 0, cv.width, cv.height);
    gl.clearColor(r, g, b, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.uniformMatrix4fv(uni.mvp, false, new Float32Array(mats()));
    gl.uniform1f(uni.ex, ex);
    gl.uniform1f(uni.sel, sel);
    gl.uniform1f(uni.mode, mode);
    gl.uniform1f(uni.ps, cv.height * 0.014); // larger than the original (fewer points per part here)
    gl.drawArrays(gl.POINTS, 0, N);
  }

  function size() {
    const r = st.getBoundingClientRect();
    if (!r.width) return;
    const d = Math.min(2, devicePixelRatio || 1);
    cv.width = Math.round(r.width * d);
    cv.height = Math.round(r.height * d);
    draw();
  }
  const ro = new ResizeObserver(size);
  ro.observe(st);

  // ---------- interaction
  const ptrs = new Map();
  let moved = 0;
  const setSpin = (v) => {
    spin = v;
    spinB.setAttribute("aria-pressed", String(v));
  };
  st.addEventListener("pointerdown", (e) => {
    st.setPointerCapture(e.pointerId);
    ptrs.set(e.pointerId, [e.clientX, e.clientY]);
    moved = 0;
    setSpin(false);
  });
  st.addEventListener("pointermove", (e) => {
    if (!ptrs.has(e.pointerId)) return;
    const p = ptrs.get(e.pointerId);
    if (ptrs.size === 1) {
      const dx = e.clientX - p[0], dy = e.clientY - p[1];
      moved += Math.abs(dx) + Math.abs(dy);
      yaw -= dx * 0.008;
      pitch = Math.max(-0.15, Math.min(1.45, pitch + dy * 0.006));
      ptrs.set(e.pointerId, [e.clientX, e.clientY]);
      draw();
    } else {
      const [a, b] = [...ptrs.values()];
      const d0 = Math.hypot(a[0] - b[0], a[1] - b[1]);
      ptrs.set(e.pointerId, [e.clientX, e.clientY]);
      const [c, d] = [...ptrs.values()];
      const d1 = Math.hypot(c[0] - d[0], c[1] - d[1]);
      if (d0 > 0 && d1 > 0) dist = Math.max(3, Math.min(16, (dist * d0) / d1));
      moved += 99;
      draw();
    }
  });
  st.addEventListener("pointerup", (e) => {
    ptrs.delete(e.pointerId);
    if (moved < 6) pick(e);
  });
  st.addEventListener("pointercancel", (e) => ptrs.delete(e.pointerId));
  st.addEventListener("wheel", (e) => {
    e.preventDefault();
    dist = Math.max(3, Math.min(16, dist * Math.exp(e.deltaY * 0.0012)));
    draw();
  }, { passive: false });

  function pick(e) {
    if (!gl) return;
    const r = cv.getBoundingClientRect();
    const mx = ((e.clientX - r.left) / r.width) * 2 - 1;
    const my = 1 - ((e.clientY - r.top) / r.height) * 2;
    const m = mats();
    let best = -1, bz = 1e9;
    const tol = (14 / r.width) * 2;
    for (let i = 0; i < N; i++) {
      const l = LAB[i];
      const o = OFF[l];
      const px = (POS[3 * i] / 32767) * S + o[0] * ex;
      const py = (POS[3 * i + 1] / 32767) * S + o[1] * ex;
      const pz = (POS[3 * i + 2] / 32767) * S + o[2] * ex + lift;
      const X = -py, Y = pz, Z = px;
      const cw = m[3] * X + m[7] * Y + m[11] * Z + m[15];
      if (cw <= 0) continue;
      const cx = (m[0] * X + m[4] * Y + m[8] * Z + m[12]) / cw;
      const cy = (m[1] * X + m[5] * Y + m[9] * Z + m[13]) / cw;
      if (Math.hypot(cx - mx, cy - my) < tol && cw < bz - 0.15) {
        bz = cw;
        best = l;
      }
    }
    select(best);
  }

  exEl.addEventListener("input", () => {
    ex = +exEl.value;
    draw();
  });
  for (const b of container.querySelectorAll(".pv-seg button")) {
    b.addEventListener("click", () => {
      mode = +b.dataset.mode;
      for (const o of container.querySelectorAll(".pv-seg button")) o.setAttribute("aria-pressed", String(o === b));
      draw();
    });
  }
  spinB.addEventListener("click", () => setSpin(!spin));
  container.querySelector(".pv-reset").addEventListener("click", () => {
    yaw = -0.75; pitch = 0.32; dist = 5.0; ex = 0; exEl.value = "0";
    select(-1);
  });

  let disposed = false;
  (function loop() {
    if (disposed) return;
    if (spin && cv.clientWidth) {
      yaw += 0.004;
      draw();
    }
    requestAnimationFrame(loop);
  })();

  return {
    dispose() {
      disposed = true;
      ro.disconnect();
      container.replaceChildren();
    },
    redraw: draw,
    hasDamage,
    select,
  };
}
