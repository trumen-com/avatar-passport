// Paint check (trade-in intake): photograph each gap between two panels; the paint on both sides of the gap is
// compared in the same photo, under the same light, and panels that differ are marked "check with a gauge".
// Engine and car layout: web/paint/engine.js and car.js from Trumen Paint Check by Team Trumen, unchanged.
// Photos are analysed in this browser and never uploaded.

import { analyzeJoint, judgePanels } from "./paint/engine.js";
import { GLASS, JOINT_XY, JOINTS, PANELS, SHAPES, jointFactor } from "./paint/car.js";

// Paint Check panel id -> parts 3D model key (the table in web/paint/INTEGRATION.md; left/right = the car's own)
export const PAINT_TO_SCAN = {
  f_bumper: "front-bumper", hood: "hood", lf_fender: "left_fender", rf_fender: "right_fender",
  lf_door: "left_front-door", rf_door: "right_front-door", lr_door: "left_back-door", rr_door: "right_back-door",
  lr_quarter: "left_quarter-panel", rr_quarter: "right_quarter-panel", trunk: "trunk", r_bumper: "back-bumper",
  roof: "roof",
};

const VERDICT = { match: ["Same paint", "ok"], check: ["Small difference", "warn"], differs: ["Different paint", "err"] };
export const STATUS = {
  original: ["Original paint", "ok", null],
  check: ["Check with a gauge", "warn", "minor"],
  suspect: ["Gauge first", "warn", "moderate"],
  repainted: ["Likely repainted", "err", "severe"],
};
const WARN = {
  gap_not_found: "Gap not found", gap_near_edge: "Gap too close to the edge", too_few_pixels: "Too little paint in view",
  mixed_surface: "Trim or sticker in the way", glare: "Too much glare", too_dark: "Too dark", blurry: "Blurry",
};
const FILL = { original: "#3fa66a", check: "#e3b21b", suspect: "#ec8a1f", repainted: "#d4321f" };
const NS = "http://www.w3.org/2000/svg";

async function loadImage(file, maxSide = 720) {
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  const s = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(bmp.width * s));
  c.height = Math.max(1, Math.round(bmp.height * s));
  const g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close?.();
  return { img: g.getImageData(0, 0, c.width, c.height), thumb: c.toDataURL("image/jpeg", 0.7) };
}

const svg = (tag, attrs) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};

// container: an empty element. onPanels(statuses) is called after every change: {panelId: {status, ...}}.
export function createPaintCheck(container, { onPanels } = {}) {
  const shots = {}; // jointId -> {orientation, result, thumb, img}
  container.innerHTML = `
    <div class="pc-demo" hidden>
      <span class="pc-demo-text small muted"></span>
      <button type="button" class="btn btn-outline btn-sm pc-demo-btn">Use demo photos</button>
    </div>
    <div class="pc-grid">
      <div class="pc-diagram">
        <div class="pc-svg"></div>
        <div class="legend pc-legend"><span class="lg pc-ok">Original paint</span><span class="lg sev-minor">Check with a gauge</span><span class="lg sev-severe">Likely repainted</span></div>
      </div>
      <div class="pc-side">
        <div class="pc-result"></div>
        <ol class="pc-joints"></ol>
      </div>
    </div>`;
  const svgBox = container.querySelector(".pc-svg");
  const list = container.querySelector(".pc-joints");
  const resultBox = container.querySelector(".pc-result");

  const root = svg("svg", { viewBox: "0 0 220 440", class: "pc-car", role: "img", "aria-label": "Car seen from above, front at the top" });
  const panelEls = {};
  for (const [id, d] of Object.entries(SHAPES)) {
    const p = svg("path", { d, class: "pc-panel" });
    const t = svg("title", {});
    t.textContent = PANELS[id].name;
    p.append(t);
    panelEls[id] = p;
    root.append(p);
  }
  for (const d of GLASS) root.append(svg("path", { d, class: "pc-glass" }));
  const dotEls = {};
  JOINTS.forEach((j, i) => {
    const [x, y] = JOINT_XY[j.id];
    const g = svg("g", { class: "pc-dot", "data-joint": j.id, tabindex: "0" });
    g.append(svg("circle", { cx: x, cy: y, r: 9 }));
    const tx = svg("text", { x, y: y + 3.5, "text-anchor": "middle" });
    tx.textContent = String(i + 1);
    g.append(tx);
    g.addEventListener("click", () => list.querySelector(`[data-joint="${j.id}"] input[type=file]`)?.click());
    dotEls[j.id] = g;
    root.append(g);
  });
  svgBox.append(root);

  for (const [i, j] of JOINTS.entries()) {
    const li = document.createElement("li");
    li.dataset.joint = j.id;
    li.innerHTML = `
      <span class="pc-num">${i + 1}</span>
      <span class="pc-where">${j.where}</span>
      <span class="pc-state"></span>
      <span class="pc-actions">
        <button type="button" class="pc-orient" title="Which way the gap runs in the photo" aria-label="Gap direction">↕</button>
        <label class="btn btn-sm pc-take"><input class="vh" type="file" accept="image/*" capture="environment">Photo</label>
      </span>
      <img class="pc-thumb" alt="" hidden>`;
    const input = li.querySelector("input");
    input.addEventListener("change", async () => {
      const f = input.files && input.files[0];
      input.value = "";
      if (!f) return;
      setState(li, "…", "");
      try {
        const { img, thumb } = await loadImage(f);
        shots[j.id] = { orientation: shots[j.id]?.orientation || "vertical", img, thumb };
        analyse(j);
      } catch (e) {
        setState(li, `Could not read the photo: ${e.message}`, "err");
      }
    });
    li.querySelector(".pc-orient").addEventListener("click", () => {
      const s = shots[j.id] || (shots[j.id] = { orientation: "vertical" });
      s.orientation = s.orientation === "horizontal" ? "vertical" : "horizontal";
      li.querySelector(".pc-orient").textContent = s.orientation === "horizontal" ? "↔" : "↕";
      if (s.img) analyse(j);
    });
    list.append(li);
  }

  function setState(li, text, tone) {
    const st = li.querySelector(".pc-state");
    st.textContent = text;
    st.className = `pc-state badge ${tone}`;
  }

  function analyse(j) {
    const s = shots[j.id];
    const li = list.querySelector(`[data-joint="${j.id}"]`);
    s.result = analyzeJoint(s.img, { orientation: s.orientation, gapX: null, factor: jointFactor(j) });
    const th = li.querySelector(".pc-thumb");
    th.src = s.thumb;
    th.hidden = false;
    if (!s.result.reliable) {
      const why = s.result.warnings.map((w) => WARN[w] || w).filter(Boolean)[0] || "Retake";
      setState(li, `Retake: ${why}`, "");
      dotEls[j.id].setAttribute("class", "pc-dot retake");
    } else {
      const [text, tone] = VERDICT[s.result.verdict];
      setState(li, text, tone);
      dotEls[j.id].setAttribute("class", `pc-dot ${s.result.verdict}`);
    }
    update();
  }

  function update() {
    const panels = judgePanels(JOINTS.map((j) => ({ a: j.a, b: j.b, factor: jointFactor(j), result: shots[j.id]?.result }))
      .filter((x) => x.result));
    for (const [id, el] of Object.entries(panelEls)) {
      const st = panels[id];
      el.style.fill = st ? FILL[st.status] : "";
      el.classList.toggle("measured", Boolean(st));
    }
    const done = JOINTS.filter((j) => shots[j.id]?.result?.reliable).length;
    const flagged = Object.entries(panels).filter(([, p]) => p.status !== "original")
      .sort((a, b) => ["repainted", "suspect", "check"].indexOf(a[1].status) - ["repainted", "suspect", "check"].indexOf(b[1].status));
    resultBox.replaceChildren();
    const head = document.createElement("p");
    head.className = "pc-progress";
    head.textContent = `${done} of ${JOINTS.length} gaps measured`;
    resultBox.append(head);
    if (done) {
      const p = document.createElement("p");
      p.className = "pc-verdict";
      p.textContent = flagged.length ? "Gauge these panels:" : "No panel stands out. Gauge as usual.";
      resultBox.append(p);
      if (flagged.length) {
        const ul = document.createElement("ul");
        ul.className = "pc-flagged";
        for (const [id, st] of flagged) {
          const li = document.createElement("li");
          const name = document.createElement("span");
          name.textContent = PANELS[id].name;
          const badge = document.createElement("span");
          badge.className = `badge ${STATUS[st.status][1]}`;
          badge.textContent = STATUS[st.status][0];
          li.append(name, badge);
          ul.append(li);
        }
        resultBox.append(ul);
      }
    }
    if (onPanels) onPanels(panels);
  }
  // Demo car: gap photos cut automatically from its 9 blurred photos (scripts/paintcheck_demo_crops.py); the analysis
  // below runs live on them exactly as on a phone photo.
  const demoBox = container.querySelector(".pc-demo");
  let demo = null;
  fetch("paint/demo/index.json", { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      if (!d || !Array.isArray(d.joints) || !d.joints.length) return;
      demo = d.joints;
      const photos = new Set(demo.map((x) => x.photo)).size;
      demoBox.querySelector(".pc-demo-text").textContent = `Demo car: ${demo.length} gaps cut automatically from its ${photos} photos.`;
      demoBox.hidden = false;
    })
    .catch(() => {});
  demoBox.querySelector(".pc-demo-btn").addEventListener("click", async () => {
    if (!demo) return;
    const btn = demoBox.querySelector(".pc-demo-btn");
    btn.disabled = true;
    for (const d of demo) {
      const j = JOINTS.find((x) => x.id === d.joint);
      const li = list.querySelector(`[data-joint="${d.joint}"]`);
      if (!j || !li) continue;
      try {
        const blob = await (await fetch(`paint/demo/${d.file}`)).blob();
        const { img, thumb } = await loadImage(blob);
        shots[j.id] = { orientation: "vertical", img, thumb, demo: d.photo };
        analyse(j);
      } catch (e) {
        setState(li, `Could not read the photo: ${e.message}`, "err");
      }
    }
    btn.disabled = false;
  });

  update();
  return { shots };
}
