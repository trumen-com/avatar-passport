// AvtoPassport 3D viewer: the server-reconstructed point cloud (PLY) or Gaussian splats, with damage pins.
// three.js 0.186.1 (MIT) via the import map in index.html; Spark 2.3.1 (MIT, @sparkjsdev/spark) only for splats.
// Loaded on demand by app.js.
//
// Orientation: the server measures it from the walkaround cameras (recon3d.view = {center, up, forward0, radius};
// forward0 = horizontal direction from the car centre to the first camera, radius = walkaround radius). The model
// is rotated so `up` is the screen up and the default camera is an elevated 3/4 view from the side the video started
// on. Without a view (older inspections) the bounding box is used as before.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

function humanize(s) {
  if (s == null || s === "") return "unknown";
  return String(s).replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
}

function button(text, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "btn btn-sm";
  b.textContent = text;
  b.addEventListener("click", onClick);
  return b;
}

const vec3 = (a) => (Array.isArray(a) && a.length === 3 && a.every((x) => Number.isFinite(Number(x))) ? new THREE.Vector3(...a.map(Number)) : null);

/** Valid {center, up, forward0, radius} as THREE vectors, or null. */
export function parseView(view) {
  if (!view || typeof view !== "object") return null;
  const center = vec3(view.center);
  const up = vec3(view.up);
  const f0 = vec3(view.forward0);
  const radius = Number(view.radius);
  if (!center || !up || !f0 || !(radius > 0) || up.lengthSq() < 1e-12 || f0.lengthSq() < 1e-12) return null;
  up.normalize();
  f0.addScaledVector(up, -f0.dot(up)); // keep it horizontal
  if (f0.lengthSq() < 1e-12) return null;
  f0.normalize();
  const extent = Number(view.extent);
  // optional oriented box around the car (server-measured): rows of axes + half sizes
  let box = null;
  const b = view.box;
  if (b && Array.isArray(b.axes) && b.axes.length === 3 && Array.isArray(b.half) && b.half.length === 3) {
    const axes = b.axes.map(vec3);
    const half = b.half.map(Number);
    if (axes.every(Boolean) && half.every((x) => x > 0)) box = { axes: axes.map((a) => a.normalize()), half };
  }
  return { center, up, f0, radius, extent: extent > 0 ? extent : null, box };
}

/**
 * Robust box of (a sample of) the positions in the view's axes (forward0, side, up): 5-95 percentile per axis.
 * Returns {center (world), extent (half the box diagonal)}: the car without stray points, or null.
 */
export function robustBox(positions, v, maxSamples = 20000) {
  if (!positions || !positions.count) return null;
  const side = new THREE.Vector3().crossVectors(v.up, v.f0);
  const axes = [v.f0, side, v.up];
  const n = positions.count;
  const step = Math.max(1, Math.floor(n / maxSamples));
  const cols = [[], [], []];
  const p = new THREE.Vector3();
  for (let i = 0; i < n; i += step) {
    p.set(positions.getX(i), positions.getY(i), positions.getZ(i)).sub(v.center);
    if (![p.x, p.y, p.z].every(Number.isFinite)) continue;
    for (let k = 0; k < 3; k++) cols[k].push(p.dot(axes[k]));
  }
  if (cols[0].length < 10) return null;
  const center = v.center.clone();
  const half = [];
  for (let k = 0; k < 3; k++) {
    const c = cols[k].sort((a, b) => a - b);
    const lo = c[Math.floor(0.05 * (c.length - 1))];
    const hi = c[Math.ceil(0.95 * (c.length - 1))];
    center.addScaledVector(axes[k], (lo + hi) / 2);
    half.push((hi - lo) / 2);
  }
  return { center, half, extent: Math.hypot(...half) };
}

/**
 * Distance along `dir` (unit, target -> eye) at which every corner of the box (half sizes `half` along the unit
 * `axes`, centred on the target) is inside the camera's vertical and horizontal field of view.
 */
export function fitBoxDistance(dir, axes, half, fovDeg, aspect, margin = 1.08) {
  const tv = Math.tan(THREE.MathUtils.degToRad(fovDeg / 2));
  const th = tv * aspect;
  const fwd = dir.clone().negate();
  let right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 1, 0));
  if (right.lengthSq() < 1e-12) right = new THREE.Vector3(1, 0, 0);
  right.normalize();
  const upc = new THREE.Vector3().crossVectors(right, fwd).normalize();
  let need = 0;
  const c = new THREE.Vector3();
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        c.set(0, 0, 0)
          .addScaledVector(axes[0], sx * half[0])
          .addScaledVector(axes[1], sy * half[1])
          .addScaledVector(axes[2], sz * half[2]);
        const z = c.dot(dir); // towards the camera
        need = Math.max(need, z + Math.abs(c.dot(right)) / th, z + Math.abs(c.dot(upc)) / tv);
      }
    }
  }
  return need * margin;
}

/**
 * Mount a viewer into `container`.
 * @param {HTMLElement} container
 * @param {{plyUrl?: string, splatUrl?: string, view?: object, pins?: Array<{xyz:number[], damage:string, part:string}>}} opts
 *   plyUrl: coloured point cloud; splatUrl: 3DGS .ply rendered with Spark (one of the two).
 * @returns {Promise<{dispose: () => void}>} resolves once the model is loaded and shown
 */
export function createViewer(container, { plyUrl, splatUrl, view = null, pins = [] } = {}) {
  return new Promise((resolve, reject) => {
    container.replaceChildren();
    container.classList.add("v3d");

    const stage = document.createElement("div");
    stage.className = "v3d-stage";
    const loading = document.createElement("div");
    loading.className = "v3d-loading";
    loading.textContent = "Downloading 3D model…";
    stage.append(loading);
    const toolbar = document.createElement("div");
    toolbar.className = "v3d-toolbar";
    container.append(stage, toolbar);

    const size = () => [Math.max(stage.clientWidth, 1), Math.max(stage.clientHeight, 1)];

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: !splatUrl, alpha: true, preserveDrawingBuffer: true });  // keeps the canvas readable for screenshots
    } catch (e) {
      container.replaceChildren();
      reject(new Error(`WebGL is not available (${e && e.message ? e.message : e})`));
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const [w0, h0] = size();
    renderer.setSize(w0, h0);
    stage.append(renderer.domElement);

    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(w0, h0);
    labelRenderer.domElement.className = "v3d-labels";
    labelRenderer.domElement.style.pointerEvents = "none";
    stage.append(labelRenderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, w0 / h0, 0.01, 1000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;

    // root (flip) > aligned (measured up -> +Y) > model (car centre at the origin)
    const root = new THREE.Group();
    const aligned = new THREE.Group();
    const model = new THREE.Group();
    aligned.add(model);
    root.add(aligned);
    scene.add(root);

    let disposed = false;
    let raf = 0;
    const disposables = [];
    const labelObjects = [];

    const ro = new ResizeObserver(() => {
      const [w, h] = size();
      renderer.setSize(w, h);
      labelRenderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    ro.observe(stage);

    function loop() {
      if (disposed) return;
      raf = requestAnimationFrame(loop);
      if (stage.clientWidth === 0) return; // its tab is hidden: skip drawing until it is shown again
      controls.update();
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    }

    function dispose() {
      if (disposed) return;
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      for (const o of labelObjects) o.element.remove();
      for (const d of disposables) {
        try {
          d.dispose();
        } catch {
          /* ignore */
        }
      }
      renderer.dispose();
      container.replaceChildren();
    }

    function fail(err) {
      const msg = err && err.message ? err.message : String(err);
      dispose();
      reject(new Error(`could not load the 3D model file (${msg})`));
    }

    function addPins(pinRadius) {
      for (const p of pins) {
        if (!p || !Array.isArray(p.xyz) || p.xyz.length < 3) continue;
        const [x, y, z] = p.xyz.map(Number);
        if (![x, y, z].every(Number.isFinite)) continue;
        const g = new THREE.SphereGeometry(pinRadius, 16, 12);
        const m = new THREE.MeshBasicMaterial({ color: 0xe0302b, depthTest: false, transparent: true, opacity: 0.95 });
        disposables.push(g, m);
        const sphere = new THREE.Mesh(g, m);
        sphere.renderOrder = 10;
        sphere.position.set(x, y, z);
        model.add(sphere);

        const outer = document.createElement("div");
        const label = document.createElement("span");
        label.className = "v3d-label";
        label.textContent = `${humanize(p.damage)} on ${humanize(p.part)}`;
        outer.append(label);
        const obj = new CSS2DObject(outer);
        obj.position.set(x, y, z);
        model.add(obj);
        labelObjects.push(obj);
      }
    }

    // Place the model and the default camera; `bbox` is the fallback when the server sent no orientation.
    // extentOf(view) -> {center, extent} robust car box (null when unknown).
    function frame(bbox, extentOf) {
      const v = parseView(view);
      let dist;
      if (v) {
        aligned.quaternion.setFromUnitVectors(v.up, new THREE.Vector3(0, 1, 0));
        model.position.copy(v.center).negate();
        // Elevated 3/4 view from the side the walkaround started on: direction (1.2 x horizontal, 0.5 x up),
        // pulled back until the car's extent fits the narrower field of view (the walkaround radius alone is
        // often closer than that for phone videos).
        const box = extentOf ? extentOf(v) : null;
        if (box) model.position.copy(box.center).negate(); // robust box middle, steadier than the point median
        const extent = (box && box.extent) || v.extent || v.radius * 0.6;
        const side = v.f0.clone().applyQuaternion(aligned.quaternion); // horizontal after the alignment
        const dir = side.clone().multiplyScalar(1.2).add(new THREE.Vector3(0, 0.5, 0)).normalize();
        let fitDist;
        const q = aligned.quaternion;
        if (box && box.half) {
          // the robust box's corners, in the displayed (aligned) frame: forward0, side, up -> rotated axes
          const axes = [v.f0.clone().applyQuaternion(q), new THREE.Vector3().crossVectors(v.up, v.f0).applyQuaternion(q),
            new THREE.Vector3(0, 1, 0)];
          fitDist = fitBoxDistance(dir, axes, box.half, camera.fov, camera.aspect);
        } else if (v.box) {
          fitDist = fitBoxDistance(dir, v.box.axes.map((a) => a.clone().applyQuaternion(q)), v.box.half, camera.fov, camera.aspect);
        } else {
          const halfV = THREE.MathUtils.degToRad(camera.fov / 2);
          const halfMin = Math.min(halfV, Math.atan(Math.tan(halfV) * camera.aspect));
          fitDist = extent / Math.sin(halfMin); // sphere around the car inside the narrower field of view
        }
        // Frame the car itself: the walkaround radius (where the phone was) leaves the car small on screen.
        const eyeDist = fitDist > 0 ? Math.max(fitDist, 0.35 * v.radius) : Math.hypot(1.2, 0.5) * v.radius;
        camera.position.copy(dir).multiplyScalar(eyeDist);
        dist = Math.max(v.radius, extent);
      } else {
        const center = bbox.getCenter(new THREE.Vector3());
        const diag = bbox.getSize(new THREE.Vector3()).length() || 1;
        model.position.copy(center).negate();
        camera.position.set(0, diag * 0.35, diag * 1.1);
        dist = diag;
      }
      camera.up.set(0, 1, 0);
      camera.near = dist / 200;
      camera.far = dist * 100;
      camera.lookAt(0, 0, 0);
      camera.updateProjectionMatrix();
      controls.target.set(0, 0, 0);
      controls.minDistance = dist * 0.05;
      controls.maxDistance = dist * 10;
      controls.update();
      controls.saveState();
      return { oriented: Boolean(v), dist };
    }

    function finish(extraButtons, infoText) {
      let labelsOn = true;
      const turn = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2);
      const buttons = [
        button("Reset view", () => controls.reset()),
        button("Turn 90°", () => {
          camera.position.sub(controls.target).applyQuaternion(turn).add(controls.target);
          controls.update();
        }),
        button("Flip upside down", () => {
          root.rotation.x = root.rotation.x === 0 ? Math.PI : 0;
        }),
        ...extraButtons,
      ];
      if (labelObjects.length) {
        buttons.push(
          button("Hide labels", (e) => {
            labelsOn = !labelsOn;
            for (const o of labelObjects) o.visible = labelsOn;
            e.currentTarget.textContent = labelsOn ? "Hide labels" : "Show labels";
          }),
        );
      }
      toolbar.append(...buttons);
      const info = document.createElement("p");
      info.className = "small muted";
      info.style.margin = "0.4rem 0 0";
      info.textContent = infoText;
      container.append(info);
      loading.remove();
      loop();
      resolve({ dispose });
    }

    const orientNote = (o) => (o ? " Oriented from the walkaround camera path." : " No orientation from the server; use Flip if the car is upside down.");

    if (splatUrl) {
      loading.textContent = "Loading the splat renderer…";
      import("@sparkjsdev/spark")
        .then(async ({ SparkRenderer, SplatMesh }) => {
          if (disposed) return;
          const spark = new SparkRenderer({ renderer });
          scene.add(spark);
          disposables.push(spark);
          loading.textContent = "Downloading high-quality 3D model…";
          const splat = new SplatMesh({ url: splatUrl });
          disposables.push(splat);
          await splat.initialized;
          if (disposed) return;
          model.add(splat);
          const bbox = splat.getBoundingBox(true);
          const { oriented, dist } = frame(bbox, null);
          addPins(0.02 * dist);
          const n = Number(splat.numSplats);
          finish([], `${Number.isFinite(n) ? n.toLocaleString() + " Gaussian splats" : "Gaussian splats"}.${orientNote(oriented)}`);
        })
        .catch(fail);
      return;
    }

    const loader = new PLYLoader();
    loader.load(
      plyUrl,
      (geom) => {
        if (disposed) {
          geom.dispose();
          return;
        }
        disposables.push(geom);
        geom.computeBoundingBox();
        const bbox = geom.boundingBox;
        const diag = bbox.getSize(new THREE.Vector3()).length() || 1;
        const nPoints = geom.getAttribute("position") ? geom.getAttribute("position").count : 0;
        const hasColor = geom.hasAttribute("color");
        const v = parseView(view);
        const box = v ? robustBox(geom.getAttribute("position"), v) : null;
        const ext = box ? box.extent : null;
        let pointSize = ext ? 0.01 * ext : 0.008 * diag;
        const material = new THREE.PointsMaterial({
          size: pointSize,
          sizeAttenuation: true,
          vertexColors: hasColor,
          color: hasColor ? 0xffffff : 0x9fb3c8,
        });
        disposables.push(material);
        model.add(new THREE.Points(geom, material));
        const { oriented } = frame(bbox, () => box);
        addPins(ext ? 0.03 * ext : 0.012 * diag);
        finish(
          [
            button("Points −", () => {
              pointSize /= 1.5;
              material.size = pointSize;
            }),
            button("Points +", () => {
              pointSize *= 1.5;
              material.size = pointSize;
            }),
          ],
          `${nPoints.toLocaleString()} points${hasColor ? "" : " (no colours in the model file)"}.${orientNote(oriented)}`,
        );
      },
      (ev) => {
        if (ev && ev.lengthComputable && ev.total) {
          loading.textContent = `Downloading 3D model… ${Math.round((ev.loaded / ev.total) * 100)}%`;
        } else if (ev && ev.loaded) {
          loading.textContent = `Downloading 3D model… ${(ev.loaded / 1048576).toFixed(1)} MB`;
        }
      },
      fail,
    );
  });
}
