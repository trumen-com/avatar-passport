// AvtoPassport 3D viewer: the server-reconstructed point cloud (PLY) with damage pins.
// three.js 0.186.1 (MIT) via the import map in index.html. Loaded on demand by app.js.

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

/**
 * Mount a viewer into `container`.
 * @param {HTMLElement} container
 * @param {{plyUrl: string, pins?: Array<{xyz:number[], damage:string, part:string, score?:number, n_views?:number}>}} opts
 * @returns {Promise<{dispose: () => void}>} resolves once the model is loaded and shown
 */
export function createViewer(container, { plyUrl, pins = [] } = {}) {
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
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });  // keeps the canvas readable for screenshots
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

    const root = new THREE.Group(); // rotated by "flip"; holds the model centred at the origin
    const model = new THREE.Group();
    root.add(model);
    scene.add(root);

    let disposed = false;
    let raf = 0;
    let geometry = null;
    let material = null;
    const pinGeometries = [];
    const pinMaterials = [];
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
      if (geometry) geometry.dispose();
      if (material) material.dispose();
      for (const g of pinGeometries) g.dispose();
      for (const m of pinMaterials) m.dispose();
      renderer.dispose();
      container.replaceChildren();
    }

    const loader = new PLYLoader();
    loader.load(
      plyUrl,
      (geom) => {
        if (disposed) {
          geom.dispose();
          return;
        }
        geometry = geom;
        geometry.computeBoundingBox();
        const bbox = geometry.boundingBox;
        const center = bbox.getCenter(new THREE.Vector3());
        const diag = bbox.getSize(new THREE.Vector3()).length() || 1;
        const nPoints = geometry.getAttribute("position") ? geometry.getAttribute("position").count : 0;
        const hasColor = geometry.hasAttribute("color");

        let pointSize = 0.01 * diag;
        material = new THREE.PointsMaterial({
          size: pointSize,
          sizeAttenuation: true,
          vertexColors: hasColor,
          color: hasColor ? 0xffffff : 0x9fb3c8,
        });
        const points = new THREE.Points(geometry, material);
        model.add(points);
        model.position.copy(center).negate();

        // Damage pins: small red spheres with "damage on part" labels.
        const pinRadius = 0.012 * diag;
        for (const p of pins) {
          if (!p || !Array.isArray(p.xyz) || p.xyz.length < 3) continue;
          const [x, y, z] = p.xyz.map(Number);
          if (![x, y, z].every(Number.isFinite)) continue;
          const g = new THREE.SphereGeometry(pinRadius, 16, 12);
          const m = new THREE.MeshBasicMaterial({ color: 0xe0302b, depthTest: false, transparent: true, opacity: 0.95 });
          pinGeometries.push(g);
          pinMaterials.push(m);
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

        // Camera: look at the model centre from a distance that fits the bounding box.
        camera.near = diag / 1000;
        camera.far = diag * 100;
        camera.position.set(0, diag * 0.35, diag * 1.1);
        camera.lookAt(0, 0, 0);
        camera.updateProjectionMatrix();
        controls.target.set(0, 0, 0);
        controls.minDistance = diag * 0.05;
        controls.maxDistance = diag * 10;
        controls.update();
        controls.saveState();

        let labelsOn = true;
        const buttons = [
          button("Reset view", () => controls.reset()),
          button("Flip upside down", () => {
            root.rotation.x = root.rotation.x === 0 ? Math.PI : 0;
          }),
          button("Points −", () => {
            pointSize /= 1.5;
            material.size = pointSize;
          }),
          button("Points +", () => {
            pointSize *= 1.5;
            material.size = pointSize;
          }),
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
        info.textContent = `${nPoints.toLocaleString()} points${hasColor ? "" : " (no colours in the model file)"}.`;
        container.append(info);

        loading.remove();
        loop();
        resolve({ dispose });
      },
      (ev) => {
        if (ev && ev.lengthComputable && ev.total) {
          loading.textContent = `Downloading 3D model… ${Math.round((ev.loaded / ev.total) * 100)}%`;
        } else if (ev && ev.loaded) {
          loading.textContent = `Downloading 3D model… ${(ev.loaded / 1048576).toFixed(1)} MB`;
        }
      },
      (err) => {
        const msg = err && err.message ? err.message : String(err);
        dispose();
        reject(new Error(`could not load the 3D model file (${msg})`));
      },
    );
  });
}
