// AvtoPassport web client. Plain ES module, no build step.
// Everything shown on the page comes from the API; when the API reports an error or a missing value we say so.

import { getLang, partSearch, setLang, startTranslator } from "./i18n.js";

const REPO_URL = "https://github.com/trumen-com/avatar-passport"; // e.g. "https://github.com/<org>/avtopassport"; the footer link is hidden while empty.

const POLL_MS = 1000;          // job polling while queued/running
const RECON_POLL_MS = 5000;    // 3D reconstruction polling after the passport is done
const RECON_MAX_POLLS = 720;   // stop watching after ~1 hour
const MAX_POLL_FAILURES = 30;  // consecutive network failures before giving up
const RECON_HQ_MAX_POLLS = 1800; // ~2.5 hours while the optional high-quality 3D is building
const TASHKENT = { lat: 41.3111, lon: 69.2797 };
// Findings whose best detection score is below this go into a collapsed "please verify" group and out of the
// headline total. Uncalibrated default; replace with per-class thresholds from Phase 2 (scripts/eval.py).
const LOW_CONFIDENCE = 0.5;
const SHOP_RADIUS_M = 5000;

const state = {
  health: null,
  jobId: null,
  statusUrl: null,
  pollToken: 0,
  job: null,
  viewer: null,
  viewerPly: null,
  viewerHq: null,
  viewerSplat: null,
  palette: null,
  tryonToken: 0,
  tryonUrl: null,
  tab: "overview",
  initialTab: null,
  pending3d: { cloud: null, hq: null, parts: null }, // finished 3D models, mounted when the 3D tab is first opened
  viewerParts: null,
  viewerPartsUrl: null,
  partsData: null,       // {url, promise}: one download shared by the Car tab and the 3D tab
  damageByPart: null,    // confirmed findings by part, for the Damage colours of the parts model
  ovParts: null,         // parts viewer in the Car tab
  ovPartsUrl: null,
  ovView: "photos",
  ovViewChosen: false,
};

const capture = { video: null, vinPhoto: null, demo: false }; // demo: use the demo files already on the server

// ---------------------------------------------------------------- helpers

const $ = (sel, root = document) => root.querySelector(sel);

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = String(v);
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v === true) node.setAttribute(k, "");
    else node.setAttribute(k, String(v));
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

class ApiError extends Error {
  constructor(message, status = 0, data = null) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

function errorMessage(data, text, status, statusText) {
  let msg = "";
  if (data && typeof data === "object") {
    const d = data.detail ?? data.error ?? data.message;
    if (typeof d === "string") msg = d;
    else if (Array.isArray(d)) msg = d.map((x) => (x && x.msg) || JSON.stringify(x)).join("; ");
    else if (d) msg = JSON.stringify(d);
  }
  if (!msg && text && text.length < 300 && !/^\s*</.test(text)) msg = text.trim();
  const head = status ? `HTTP ${status}${statusText ? " " + statusText : ""}` : "Request failed";
  return msg ? `${head}: ${msg}` : head;
}

async function apiJSON(url, opts = {}) {
  let res;
  try {
    res = await fetch(url, { headers: { Accept: "application/json" }, cache: "no-store", ...opts });
  } catch (e) {
    throw new ApiError(`Network error: ${e && e.message ? e.message : e}`, 0);
  }
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!res.ok) throw new ApiError(errorMessage(data, text, res.status, res.statusText), res.status, data);
  if (data === null) throw new ApiError(`HTTP ${res.status}: the server did not return JSON`, res.status);
  return data;
}

const numberFmt = new Intl.NumberFormat("ru-RU");
const fmtNum = (n) => numberFmt.format(n);
const fmtUZS = (n) => `${fmtNum(n)} UZS`;

function fmtBytes(b) {
  if (!Number.isFinite(b)) return "?";
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(0)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function fmtDist(m) {
  if (!Number.isFinite(m)) return "";
  return m < 1000 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(1)} km`;
}

function fmtDate(iso) {
  if (!iso) return "not reported";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  return `${d.toLocaleString()} (${iso})`;
}

function fmtDateShort(iso) {
  const d = iso ? new Date(iso) : null;
  return d && !Number.isNaN(d.getTime()) ? d.toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "";
}

function fmtMs(ms) {
  if (!Number.isFinite(ms)) return "";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

function humanize(s) {
  if (s == null || s === "") return "unknown";
  return String(s).replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
}

function sentence(s) {
  const t = humanize(s);
  return t.charAt(0).toUpperCase() + t.slice(1);
}

function pct(x) {
  return `${(x * 100).toFixed(3)}%`;
}

// Only allow http(s) or same-origin relative links from API data in href/src.
function safeHref(u) {
  if (typeof u !== "string" || !u) return null;
  try {
    const x = new URL(u, location.href);
    return x.protocol === "http:" || x.protocol === "https:" ? x.href : null;
  } catch {
    return null;
  }
}

function baseName(u) {
  const s = String(u || "").split("#")[0].split("?")[0];
  let b = s.substring(s.lastIndexOf("/") + 1);
  try {
    b = decodeURIComponent(b);
  } catch {
    /* keep raw */
  }
  return b;
}

function stemKey(u) {
  return baseName(u).replace(/\.[a-z0-9]+$/i, "").toLowerCase();
}

function normVin(s) {
  return String(s || "").toUpperCase().replace(/[\s-]+/g, "");
}

// The job may carry diff/media/recon3d/report at top level or inside result; accept both.
function pick(job, key) {
  if (!job) return undefined;
  if (job[key] !== undefined && job[key] !== null) return job[key];
  if (job.result && job.result[key] !== undefined) return job.result[key];
  return undefined;
}

function kvList(obj, depth = 0) {
  if (obj == null) return el("span", { class: "muted" }, "not reported");
  if (typeof obj !== "object") return el("span", {}, String(obj));
  if (Array.isArray(obj)) {
    if (!obj.length) return el("span", { class: "muted" }, "none");
    if (obj.every((x) => x == null || typeof x !== "object")) return el("span", {}, obj.map(String).join(", "));
    if (depth > 2) return el("code", {}, JSON.stringify(obj));
    return el("ul", { class: "notes" }, obj.map((x) => el("li", {}, kvList(x, depth + 1))));
  }
  const entries = Object.entries(obj);
  if (!entries.length) return el("span", { class: "muted" }, "none");
  if (depth > 2) return el("code", {}, JSON.stringify(obj));
  return el("dl", { class: "kv" }, entries.map(([k, v]) => [el("dt", {}, humanize(k)), el("dd", {}, kvList(v, depth + 1))]));
}

// ---------------------------------------------------------------- health

async function loadHealth() {
  const pill = $("#health");
  const banner = $("#health-banner");
  try {
    const h = await apiJSON("/api/health");
    state.health = h;
    const ready = h.ok && h.models_ready;
    pill.textContent = h.ok ? `Server ${ready ? "ready" : "up"}${h.device ? " · " + h.device : ""}` : "Server problem";
    pill.className = `pill ${ready ? "ok" : "warn"}`;
    if (h.version) $("#version-text").textContent = `v${h.version}`;
    if (h.demo && h.demo.video) {
      $("#demo-files").textContent = `${h.demo.video}${h.demo.vin_photo ? ` + ${h.demo.vin_photo}` : ""} · already on the server, no upload`;
      $("#demo-box").hidden = false;
      apiJSON("/api/inspections/demo/latest")
        .then((d) => {
          const a = $("#demo-ready");
          a.href = `?inspection=${encodeURIComponent(d.id)}`;
          a.title = `A finished demo passport made ${fmtDateShort(d.created_utc)}`;
          a.hidden = false;
        })
        .catch(() => {}); // none finished yet: only the live run is offered
    }
    const msgs = [];
    if (!h.ok) msgs.push("The server health check reports a problem.");
    if (h.models_ready === false) msgs.push("The models are not loaded on the server yet, so an inspection may wait or fail.");
    if (h.recon3d_available === false) msgs.push("3D reconstruction is not available on this server; the 360° view is still built.");
    if (msgs.length) {
      banner.textContent = msgs.join(" ");
      banner.className = h.ok ? "banner info" : "banner";
      banner.hidden = false;
    } else {
      banner.hidden = true;
    }
  } catch (e) {
    pill.textContent = "Server unreachable";
    pill.className = "pill err";
    banner.textContent = `Could not reach the AvtoPassport server (${e.message}). Uploads will fail until it is back.`;
    banner.className = "banner err";
    banner.hidden = false;
  }
}

// ---------------------------------------------------------------- capture

function readVideoDuration(file) {
  return new Promise((resolve) => {
    const v = document.createElement("video");
    v.preload = "metadata";
    v.muted = true;
    v.playsInline = true;
    const url = URL.createObjectURL(file);
    let settled = false;
    const done = (d) => {
      if (settled) return;
      settled = true;
      URL.revokeObjectURL(url);
      v.removeAttribute("src");
      resolve(d);
    };
    v.onloadedmetadata = () => done(Number.isFinite(v.duration) ? v.duration : null);
    v.onerror = () => done(null);
    setTimeout(() => done(null), 8000);
    v.src = url;
  });
}

function useDemo() {
  const d = state.health && state.health.demo;
  if (!d) return;
  capture.demo = true;
  capture.video = null;
  capture.vinPhoto = null;
  for (const id of ["#video-camera", "#video-library", "#vin-camera", "#vin-library"]) $(id).value = "";
  $("#video-info").textContent = `Demo: ${d.video} (${fmtBytes(d.video_bytes)})`;
  $("#video-info").className = "file-info";
  $("#vin-photo-info").textContent = d.vin_photo ? `Demo: ${d.vin_photo}` : "No VIN photo selected.";
  $("#vin-photo-info").className = d.vin_photo ? "file-info" : "file-info muted";
  $("#vin-thumb").hidden = true;
  $("#demo-btn").classList.add("active");
  updateSubmit();
}

function clearDemo() {
  if (!capture.demo) return;
  capture.demo = false;
  $("#demo-btn").classList.remove("active");
  if (!capture.video) {
    $("#video-info").textContent = "No video selected.";
    $("#video-info").className = "file-info muted";
  }
  if (!capture.vinPhoto) {
    $("#vin-photo-info").textContent = "No VIN photo selected.";
    $("#vin-photo-info").className = "file-info muted";
  }
}

async function onVideoPicked(file) {
  capture.video = file || null;
  clearDemo();
  const info = $("#video-info");
  if (!file) {
    info.textContent = "No video selected.";
    info.className = "file-info muted";
    updateSubmit();
    return;
  }
  info.className = "file-info";
  info.textContent = `${file.name || "video"} · ${fmtBytes(file.size)}`;
  updateSubmit();
  const dur = await readVideoDuration(file);
  if (capture.video !== file) return;
  if (dur == null) {
    info.textContent = `${file.name || "video"} · ${fmtBytes(file.size)} · duration not readable in this browser`;
    return;
  }
  let advice = "";
  if (dur < 20) advice = " · shorter than recommended (30–60 s); the spin and 3D model may have gaps";
  else if (dur > 90) advice = " · longer than recommended (30–60 s); processing will take longer";
  info.textContent = `${file.name || "video"} · ${fmtBytes(file.size)} · ${dur.toFixed(0)} s${advice}`;
}

function onVinPhotoPicked(file) {
  capture.vinPhoto = file || null;
  clearDemo();
  const info = $("#vin-photo-info");
  const thumb = $("#vin-thumb");
  if (thumb.dataset.url) URL.revokeObjectURL(thumb.dataset.url);
  if (!file) {
    info.textContent = "No VIN photo selected.";
    info.className = "file-info muted";
    thumb.hidden = true;
    return;
  }
  info.className = "file-info";
  info.textContent = `${file.name || "photo"} · ${fmtBytes(file.size)}`;
  const url = URL.createObjectURL(file);
  thumb.dataset.url = url;
  thumb.src = url;
  thumb.hidden = false;
}

function onVinTextInput() {
  const v = normVin($("#vin-text").value);
  const hint = $("#vin-text-hint");
  if (!v) {
    hint.textContent = "Leave empty to read the VIN from the photo.";
    hint.className = "file-info muted small";
    return;
  }
  const problems = [];
  if (v.length !== 17) problems.push(`${v.length} of 17 characters`);
  if (/[IOQ]/.test(v)) problems.push("VINs never contain I, O or Q");
  if (/[^A-Z0-9]/.test(v)) problems.push("only letters and digits are allowed");
  hint.textContent = problems.length ? problems.join(" · ") : "17 characters. The server validates the VIN.";
  hint.className = `file-info small ${problems.length ? "" : "muted"}`;
  hint.style.color = problems.length ? "var(--warn)" : "";
}

function updateSubmit() {
  const btn = $("#submit-btn");
  btn.disabled = !capture.video && !capture.demo;
  $("#submit-hint").textContent = capture.demo
    ? "Uses the demo files on the server. Nothing is uploaded."
    : capture.video
      ? "Uploads the video (and VIN photo) to the server."
      : "Choose a walkaround video to start.";
}

function setUpload(fraction, text) {
  const f = Math.max(0, Math.min(1, fraction));
  $("#upload-bar").style.width = pct(f);
  $("#upload-progress").setAttribute("aria-valuenow", String(Math.round(f * 100)));
  $("#upload-text").textContent = text;
}

function setFormDisabled(disabled) {
  for (const node of document.querySelectorAll("#capture-form input, #capture-form button")) node.disabled = disabled;
  if (!disabled) updateSubmit();
}

function showJobError(msg) {
  const e = $("#job-error");
  e.textContent = msg;
  e.hidden = false;
  $("#retry-btn").hidden = false;
  $("#job").hidden = false;
}

async function startDemo() {
  const fd = new FormData();
  const vinText = normVin($("#vin-text").value);
  if (vinText) fd.append("vin_text", vinText);
  $("#job").hidden = false;
  $("#job-error").hidden = true;
  $("#retry-btn").hidden = true;
  $("#steps").replaceChildren();
  $("#job-status").textContent = "";
  setFormDisabled(true);
  setUpload(1, "Demo files are already on the server.");
  $("#job").scrollIntoView({ block: "center", behavior: "smooth" });
  try {
    const data = await apiJSON("/api/inspections/demo", { method: "POST", body: fd });
    startJob(String(data.id), data.status_url);
  } catch (e) {
    setFormDisabled(false);
    showJobError(`Could not start the demo: ${e.message}`);
  }
}

function startUpload() {
  if (capture.demo) {
    startDemo();
    return;
  }
  if (!capture.video) return;
  const fd = new FormData();
  fd.append("video", capture.video, capture.video.name || "walkaround.mp4");
  if (capture.vinPhoto) fd.append("vin_photo", capture.vinPhoto, capture.vinPhoto.name || "vin.jpg");
  const vinText = normVin($("#vin-text").value);
  if (vinText) fd.append("vin_text", vinText);

  $("#job").hidden = false;
  $("#job-error").hidden = true;
  $("#retry-btn").hidden = true;
  $("#steps").replaceChildren();
  $("#job-status").textContent = "";
  setFormDisabled(true);
  setUpload(0, "Starting upload…");
  $("#job").scrollIntoView({ block: "center", behavior: "smooth" }); // the progress sits below the form

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/inspections");
  xhr.setRequestHeader("Accept", "application/json");
  xhr.upload.onprogress = (ev) => {
    if (ev.lengthComputable) setUpload(ev.loaded / ev.total, `Uploading… ${fmtBytes(ev.loaded)} of ${fmtBytes(ev.total)}`);
    else setUpload(0, `Uploading… ${fmtBytes(ev.loaded)}`);
  };
  xhr.upload.onload = () => setUpload(1, "Upload complete. Waiting for the server…");
  xhr.onload = () => {
    let data = null;
    try {
      data = JSON.parse(xhr.responseText);
    } catch {
      data = null;
    }
    if (xhr.status >= 200 && xhr.status < 300 && data && data.id != null) {
      setUpload(1, "Upload complete.");
      startJob(String(data.id), data.status_url);
    } else {
      setFormDisabled(false);
      showJobError(`Upload rejected. ${errorMessage(data, xhr.responseText, xhr.status, xhr.statusText)}`);
    }
  };
  xhr.onerror = () => {
    setFormDisabled(false);
    showJobError("Network error during upload. Check the connection and try again.");
  };
  xhr.onabort = () => {
    setFormDisabled(false);
    showJobError("Upload was cancelled.");
  };
  xhr.send(fd);
}

// ---------------------------------------------------------------- job polling

function inspectionUrl(id) {
  return `/api/inspections/${encodeURIComponent(id)}`;
}

function setShareUrl(id) {
  const u = new URL(location.href);
  u.searchParams.set("inspection", id);
  history.replaceState(null, "", u.pathname + u.search + (tabFromHash() ? location.hash : ""));
}

function startJob(id, statusUrl) {
  state.jobId = id;
  state.statusUrl = safeHref(statusUrl) ? statusUrl : inspectionUrl(id);
  setShareUrl(id);
  $("#job").hidden = false;
  $("#job-status").textContent = "Waiting for the server…";
  const token = ++state.pollToken;
  pollJob(token, 0);
}

function renderSteps(job) {
  const list = $("#steps");
  const progress = Array.isArray(job.progress) ? job.progress : [];
  const finished = job.status === "done";
  const failed = job.status === "error";
  list.replaceChildren(
    ...progress.map((p, i) => {
      const f = Number(p && p.fraction);
      const isLast = i === progress.length - 1;
      const done = finished || f >= 1 || !isLast;
      const broken = failed && isLast && !done;
      const li = el(
        "li",
        { class: broken ? "failed" : done ? "done" : "active" },
        el("span", { class: "mark", "aria-hidden": "true" }, broken ? "✕" : done ? "✓" : el("span", { class: "spinner" })),
        el("span", {}, humanize(p && p.step)),
        el("span", { class: "pct" }, Number.isFinite(f) ? `${Math.round(f * 100)}%` : ""),
      );
      return li;
    }),
  );
}

const STATUS_TEXT = {
  queued: "Queued on the server…",
  running: "Analysing the walkaround…",
  done: "Passport ready.",
  error: "The inspection failed.",
};

async function pollJob(token, failures) {
  if (token !== state.pollToken) return;
  let job;
  try {
    job = await apiJSON(state.statusUrl);
  } catch (e) {
    if (token !== state.pollToken) return;
    const retriable = e.status === 0 || e.status >= 500 || e.status === 429;
    if (e.status === 404) {
      showJobError(`Inspection ${state.jobId} was not found on the server.`);
      setFormDisabled(false);
      return;
    }
    if (retriable && failures < MAX_POLL_FAILURES) {
      $("#job-status").textContent = `Could not get the status (${e.message}). Retrying…`;
      setTimeout(() => pollJob(token, failures + 1), POLL_MS * Math.min(1 + failures, 5));
      return;
    }
    showJobError(`Could not get the inspection status: ${e.message}`);
    setFormDisabled(false);
    return;
  }
  if (token !== state.pollToken) return;
  state.job = job;
  renderSteps(job);
  $("#job-status").textContent = STATUS_TEXT[job.status] || `Status: ${job.status}`;

  if (job.status === "queued" || job.status === "running") {
    setTimeout(() => pollJob(token, 0), POLL_MS);
  } else if (job.status === "error") {
    showJobError(`Inspection failed: ${job.error || "the server did not return an error message."}`);
    setFormDisabled(false);
  } else if (job.status === "done") {
    try {
      renderPassport(job);
    } catch (e) {
      console.error(e);
      showJobError(`The passport arrived but could not be displayed: ${e.message}`);
    }
    watchRecon(token, job);
  } else {
    showJobError(`Unexpected inspection status from the server: ${JSON.stringify(job.status)}`);
    setFormDisabled(false);
  }
}

// ---------------------------------------------------------------- passport

function renderPassport(job) {
  const r = job.result;
  if (!r || typeof r !== "object") throw new Error("the server returned status done without a result");
  const extra = r.extra || {};
  const media = pick(job, "media") || {};

  $("#capture").hidden = true;
  $("#passport").hidden = false;

  const pdf = safeHref(media.pdf);
  const pdfTop = $("#pdf-link-top");
  if (pdf) {
    pdfTop.href = pdfInLang(pdf);
    pdfTop.hidden = false;
  } else {
    pdfTop.hidden = true;
  }

  const findings = Array.isArray(r.findings) ? r.findings : [];
  spin.sevByKey = new Map(findings.map((f) => [`${f.part}|${f.damage}`, f.severity]));
  state.damageByPart = damageByPart(findings);
  renderVin(r, extra);
  setupSpin(media, extra);
  renderDamage(r);
  renderSummary(r);
  renderCarCard(r, extra);
  renderDiff(job, r);
  renderHistory(r, job);
  setupTryon();
  renderVerification(job, r, extra, media);
  const car = carInfo(extra);
  state.carName = car.name || null; // make (and model when decoded) for the part searches
  $("#passport-title").textContent = car.name ? `${car.name}${car.d.model_year != null ? ` · ${car.d.model_year}` : " · condition passport"}` : "Condition passport";
  $("#passport-sub").textContent = [`Inspection ${String(job.id ?? state.jobId).slice(0, 8)}`, fmtDateShort(r.created_utc)].filter(Boolean).join(" · ");
  $("#tabs").hidden = false;
  $("#share-btn").hidden = false;
  document.body.classList.add("has-passport");
  showTab(state.initialTab || tabFromHash() || "overview", { scroll: false });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------- sidebar car card and the Car tab's inspection summary (only data the server returned)

function titleCase(s) {
  const t = String(s || "").trim();
  return t.length > 3 && t === t.toUpperCase() ? t.toLowerCase().replace(/\b\p{L}/gu, (c) => c.toUpperCase()) : t;
}

function carInfo(extra) {
  const d = extra && extra.vin_decode && typeof extra.vin_decode === "object" ? extra.vin_decode : {};
  const make = d.make || d.manufacturer;
  return { d, name: [make ? titleCase(make) : null, d.model].filter(Boolean).join(" ") };
}

function renderCarCard(r, extra) {
  const { d, name } = carInfo(extra);
  const thumb = $("#car-thumb");
  thumb.hidden = !spin.urls.length;
  if (spin.urls.length) thumb.src = spin.urls[0];
  $("#car-name").textContent = name || "Make not decoded";
  const meta = [];
  if (d.model_year != null) meta.push(`Model year ${d.model_year}`);
  else if (Array.isArray(d.model_year_candidates) && d.model_year_candidates.length) meta.push(`Model year ${d.model_year_candidates.join(" or ")}`);
  if (d.country) meta.push(titleCase(d.country));
  $("#car-meta").textContent = meta.join(" · ");
  $("#car-vin").textContent = r.vin || extra.vin_masked || "not read";
  const st = VIN_STATUS[r.vin_status] || [r.vin_status || "unknown", "", ""];
  const badges = [el("span", { class: `badge ${st[1]}`, title: st[2] }, `VIN ${st[0]}`)];
  const v = extra.vin_validation && typeof extra.vin_validation === "object" ? extra.vin_validation : {};
  if (v.check_digit_ok === true) badges.push(el("span", { class: "badge ok" }, "✓ Check digit"));
  else if (v.check_digit_ok === false) badges.push(el("span", { class: "badge warn" }, "Check digit fails"));
  $("#car-badges").replaceChildren(...badges);
  $("#car-card").hidden = false;
}

const SEV_RANK = { severe: 3, moderate: 2, minor: 1 };

function sevBars(sev) {
  const n = SEV_RANK[sev] || 0;
  return el("span", { class: "bars", "aria-hidden": "true" }, [1, 2, 3].map((i) => el("i", { class: i <= n ? "on" : "" })));
}

function goToPart(part) {
  showTab("defects");
  const rows = [...document.querySelectorAll("#damage-body tr[data-part]")].filter((tr) => tr.dataset.part === part);
  const low = rows.length && rows.every((tr) => tr.closest("details")) ? rows[0].closest("details") : null;
  if (low) low.open = true;
  for (const tr of rows) {
    tr.classList.remove("flash");
    void tr.offsetWidth; // restart the highlight animation
    tr.classList.add("flash");
  }
  if (rows[0]) rows[0].scrollIntoView({ block: "center", behavior: "smooth" });
}

function renderSummary(r) {
  const findings = Array.isArray(r.findings) ? r.findings : [];
  const main = findings.filter((f) => !isLowConfidence(f));
  const low = findings.filter(isLowConfidence);
  $("#sum-count").textContent = String(main.length);
  $("#sum-label").textContent = main.length === 1 ? "damage found" : "damages found";
  $("#spin-chip").textContent = main.length ? `${main.length} damage${main.length === 1 ? "" : "s"} found` : "No confirmed damage";
  const byRank = (a, b) => (SEV_RANK[b.severity] || 0) - (SEV_RANK[a.severity] || 0) || String(a.part).localeCompare(String(b.part));
  $("#sum-chips").replaceChildren(
    ...[...main].sort(byRank).map((f) =>
      el("button", { class: `sev-chip sev-${f.severity || "none"}`, type: "button", title: `${humanize(f.damage)} · ${f.severity || "not rated"}`, onclick: () => goToPart(f.part) },
        sevBars(f.severity), sentence(f.part)),
    ),
  );
  const notes = [];
  if (low.length) notes.push(`+${low.length} uncertain.`);
  if (findings.length) notes.push("Check each one on the car.");
  else notes.push("Nothing detected. Still check the car.");
  $("#sum-note").textContent = notes.join(" ");
  const t = state.damageTotals || { sum: 0, unpriced: 0 };
  $("#sum-total").textContent = main.length ? (t.sum > 0 ? `from ${fmtUZS(t.sum)}` : "no price data") : "nothing to repair";
  $("#sum-total-note").textContent = main.length && t.sum > 0 ? `Starting prices${t.unpriced ? ` · ${t.unpriced} item${t.unpriced === 1 ? "" : "s"} without a price` : ""}` : "";

  // Parts with findings: worst confirmed severity per part, or "check" when the part only has low-confidence findings.
  const parts = new Map();
  for (const f of findings) {
    const p = parts.get(f.part) || { part: f.part, sev: null, damages: new Set(), lowOnly: true };
    p.damages.add(humanize(f.damage));
    if (!isLowConfidence(f)) {
      p.lowOnly = false;
      if ((SEV_RANK[f.severity] || 0) > (SEV_RANK[p.sev] || 0)) p.sev = f.severity;
    }
    parts.set(f.part, p);
  }
  const list = [...parts.values()].sort((a, b) => a.lowOnly - b.lowOnly || (SEV_RANK[b.sev] || 0) - (SEV_RANK[a.sev] || 0) || String(a.part).localeCompare(String(b.part)));
  $("#parts-title").hidden = true;
  $("#parts-list").hidden = true;
  $("#parts-title").textContent = `Parts with findings · ${list.length}`;
  $("#parts-list").replaceChildren(
    ...(list.length
      ? list.map((p) =>
          el("li", {}, el("button", { class: "part-row", type: "button", onclick: () => goToPart(p.part) },
            el("span", { class: "part-name" }, sentence(p.part), el("span", { class: "small muted" }, [...p.damages].join(", "))),
            p.lowOnly ? el("span", { class: "badge warn" }, "check") : el("span", { class: `badge sev-${p.sev || "none"}` }, sevBars(p.sev), p.sev || "not rated"))))
      : [el("li", { class: "small muted" }, "No damage was detected on any part.")]),
  );
}

// ---------- Car tab: the parts 3D model is the main view once it is built; the 360° photos are one tap away

function damageByPart(findings) {
  const out = {};
  for (const f of findings) {
    if (isLowConfidence(f) || !f.part || f.part === "unknown") continue;
    const item = `${humanize(f.damage)} (${f.severity || "not rated"})`;
    const cur = out[f.part];
    if (!cur) out[f.part] = { severity: f.severity, items: [item] };
    else {
      cur.items.push(item);
      if ((SEV_RANK[f.severity] || 0) > (SEV_RANK[cur.severity] || 0)) cur.severity = f.severity;
    }
  }
  return out;
}

function partsData(url) {
  if (!state.partsData || state.partsData.url !== url) {
    const promise = apiJSON(url);
    state.partsData = { url, promise };
    promise.catch(() => {
      if (state.partsData && state.partsData.promise === promise) state.partsData = null;
    });
  }
  return state.partsData.promise;
}

function setOverviewView(view) {
  const show3d = view === "3d" && Boolean(state.ovParts);
  state.ovView = show3d ? "3d" : "photos";
  $("#overview-3d").hidden = !show3d;
  $("#ov-photos").hidden = show3d;
  for (const b of document.querySelectorAll("#view-toggle button")) b.setAttribute("aria-pressed", String(b.dataset.view === state.ovView));
  if (show3d) toggleSpinPlay(false);
}

function showPartCaption(p) {
  const cap = $("#ov3d-caption");
  if (!p) {
    cap.textContent = "Tap a part to see its damage.";
    return;
  }
  cap.replaceChildren(el("strong", {}, p.name), " · ");
  if (p.damage) {
    cap.append(p.damage.items.join(", "), " · ",
      el("a", { href: "#defects", onclick: (e) => { e.preventDefault(); goToPart(p.key); } }, "View in Damage list"), " · ");
  }
  const s = partSearch(p.key, state.carName, p.name);
  cap.append(el("a", { href: s.olx, target: "_blank", rel: "noopener", title: s.query }, "OLX.uz"), " · ",
    el("a", { href: s.google, target: "_blank", rel: "noopener" }, "Google"));
}

async function mountOverviewParts(url) {
  state.ovPartsUrl = url;
  try {
    const [mod, data] = await Promise.all([import("./partsviewer.js"), partsData(url)]);
    if (state.ovParts) state.ovParts.dispose();
    state.ovParts = mod.createPartsViewer($("#ov3d-viewer"), data,
      { damage: state.damageByPart || {}, mode: 2, compact: true, onSelect: showPartCaption, car: state.carName });
    $("#view-toggle").hidden = false;
    if (!state.ovViewChosen) setOverviewView("3d");
  } catch (e) {
    console.error(e);
    state.ovPartsUrl = null; // the 360° photos stay as the view
  }
}

// ---------- Paint check (trade-in): Trumen Paint Check engine; flagged panels are coloured on the parts 3D model

async function initPaint() {
  if (state.paint) return;
  state.paint = "loading";
  try {
    const mod = await import("./paintcheck.js");
    state.paint = mod.createPaintCheck($("#paint-check"), { onPanels: (p) => showPaintOn3d(p, mod) });
  } catch (e) {
    console.error(e);
    state.paint = null;
    $("#paint-check").replaceChildren(el("p", { class: "error" }, `Paint check could not start: ${e.message}`));
  }
}

async function showPaintOn3d(panels, mod) {
  const flagged = Object.entries(panels).filter(([, s]) => s.status !== "original");
  setTabCount("paint", flagged.length ? String(flagged.length) : "", "warn");
  const url = state.ovPartsUrl || state.viewerPartsUrl;
  const card = $("#paint3d-card");
  if (!flagged.length || !url) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const damage = {};
  for (const [id, s] of flagged) damage[mod.PAINT_TO_SCAN[id]] = { severity: mod.STATUS[s.status][2], items: [mod.STATUS[s.status][0]] };
  try {
    const [pv, data] = await Promise.all([import("./partsviewer.js"), partsData(url)]);
    if (state.paint3d) state.paint3d.dispose();
    state.paint3d = pv.createPartsViewer($("#paint3d-viewer"), data, { damage, mode: 2, compact: true, car: state.carName });
  } catch (e) {
    console.error(e);
    card.hidden = true;
  }
}

// ---------- light / dark theme (light by default, like the reference design; the choice is kept in this browser)

function syncThemeButton() {
  const dark = document.documentElement.dataset.theme === "dark";
  const b = $("#theme-btn");
  b.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
  b.title = dark ? "Light mode" : "Dark mode";
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = dark ? "#11151b" : "#ffffff";
}

function toggleTheme() {
  const dark = document.documentElement.dataset.theme !== "dark";
  if (dark) document.documentElement.dataset.theme = "dark";
  else delete document.documentElement.dataset.theme;
  try {
    localStorage.setItem("avto-theme", dark ? "dark" : "light");
  } catch {
    /* storage blocked: the theme still switches for this page */
  }
  syncThemeButton();
}

const pdfInLang = (u) => (u && getLang() !== "en" ? `${u}?lang=${getLang()}` : u);

// ---------- tabs (a scrolling top bar on phones, a left sidebar on wide screens; see style.css)

const TABS = ["overview", "model3d", "tryon", "defects", "paint", "shops", "vin", "report"]; // sidebar order

function tabFromHash() {
  const h = decodeURIComponent(location.hash.replace(/^#/, ""));
  return TABS.includes(h) ? h : null;
}

function showTab(name, { focus = false, scroll = true } = {}) {
  if (!TABS.includes(name)) name = "overview";
  state.tab = name;
  for (const t of TABS) {
    const on = t === name;
    const btn = $(`#tab-${t}`);
    btn.setAttribute("aria-selected", String(on));
    btn.tabIndex = on ? 0 : -1;
    $(`#panel-${t}`).hidden = !on;
  }
  const btn = $(`#tab-${name}`);
  if (focus) btn.focus();
  const bar = $("#tabs");
  if (bar.scrollWidth > bar.clientWidth) {
    // keep the active tab visible in the horizontally scrolling phone tab bar
    const l = btn.offsetLeft;
    const r = l + btn.offsetWidth;
    if (l < bar.scrollLeft + 24) bar.scrollTo({ left: Math.max(0, l - 32), behavior: "smooth" });
    else if (r > bar.scrollLeft + bar.clientWidth - 24) bar.scrollTo({ left: r - bar.clientWidth + 32, behavior: "smooth" });
  }
  if (location.hash !== `#${name}`) history.replaceState(null, "", `${location.pathname}${location.search}#${name}`);
  if (scroll) {
    // after scrolling down a long tab, start the next one at its top (just below the sticky bars)
    const panels = $(".panels");
    const margin = parseFloat(getComputedStyle(panels).scrollMarginTop) || 0;
    if (panels.getBoundingClientRect().top < margin) panels.scrollIntoView({ block: "start" });
  }
  if (name === "model3d") mountPending3d();
  if (name === "paint") initPaint();
  if (name !== "overview") toggleSpinPlay(false);
}

function bindTabs() {
  const bar = $("#tabs");
  bar.addEventListener("click", (e) => {
    const b = e.target.closest("[data-tab]");
    if (b) showTab(b.dataset.tab);
  });
  bar.addEventListener("keydown", (e) => {
    const i = TABS.indexOf(state.tab);
    const next = { ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1, Home: 0, End: TABS.length - 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    showTab(TABS[(next + TABS.length) % TABS.length], { focus: true });
  });
  for (const t of document.querySelectorAll("[data-go]")) t.addEventListener("click", () => showTab(t.dataset.go));
  window.addEventListener("hashchange", () => {
    const t = tabFromHash();
    if (t && t !== state.tab && !$("#passport").hidden) showTab(t);
  });
}

// Overview tiles and the small counters on the tabs; filled by the renderers below from the same data they show.
function setTile(key, value, sub, tone = "") {
  const v = $(`#tile-${key}`);
  if (v) {
    v.textContent = value;
    v.dataset.tone = tone;
  }
  const s = $(`#tile-${key}-sub`);
  if (s) s.textContent = sub || "";
}

function setTabCount(key, text, tone = "") {
  const c = $(`#tabc-${key}`);
  if (!c) return;
  c.textContent = text || "";
  c.dataset.tone = tone;
}

// ---------- VIN card

const VIN_STATUS = {
  read: ["read", "ok", "VIN read from the photo"],
  manual: ["typed", "info", "VIN typed by the user"],
  ambiguous: ["ambiguous", "warn", "Some characters could not be read with certainty"],
  could_not_read: ["could not read", "err", "The VIN could not be read"],
};

const SOURCE_TEXT = { wmi_table: "WMI table", vpic: "NHTSA", vin_rules: "VIN rules", ocr: "photo", manual: "typed" };
const sourceText = (k) => SOURCE_TEXT[k] || humanize(k);

const DECODE_FIELDS = [
  ["manufacturer", "Manufacturer"],
  ["country", "Country"],
  ["make", "Make"],
  ["model", "Model"],
  ["model_year", "Model year"],
  ["body_class", "Body"],
  ["fuel_type", "Fuel"],
  ["plant_country", "Plant country"],
  ["plant_city", "Plant city"],
  ["wmi", "WMI"],
];

function renderVin(r, extra) {
  const st = VIN_STATUS[r.vin_status] || [r.vin_status || "unknown", "", ""];
  const badge = $("#vin-status");
  badge.textContent = st[0];
  badge.className = `badge ${st[1]}`;
  badge.title = st[2];

  {
    const d = extra.vin_decode && typeof extra.vin_decode === "object" ? extra.vin_decode : {};
    const car = [d.make || d.manufacturer, d.model, d.model_year].filter((x) => x != null && x !== "").join(" ");
    setTile("vin", r.vin || extra.vin_masked || "Not read", [titleCase(car), `VIN ${st[0]}`].filter(Boolean).join(" · "), r.vin ? "" : "warn");
  }

  const vinEl = $("#vin-value");
  if (r.vin) {
    vinEl.textContent = r.vin;
    vinEl.classList.remove("muted");
  } else {
    vinEl.textContent = extra.vin_masked ? `${extra.vin_masked} (partly read)` : "VIN not read";
    vinEl.classList.add("muted");
  }

  const box = $("#vin-decode");
  box.replaceChildren();
  const dec = extra.vin_decode;
  if (!dec || typeof dec !== "object") {
    box.append(el("p", { class: "muted small" }, r.vin ? "The server returned no VIN decode." : "Not decoded: no VIN."));
    return;
  }
  const sources = dec.sources || {};
  const rows = [];
  for (const [key, label] of DECODE_FIELDS) {
    let v = dec[key];
    if (v == null || v === "") continue;
    if (key === "model_year" && Array.isArray(dec.model_year_candidates) && dec.model_year_candidates.length > 1) {
      v = `${v} (candidates: ${dec.model_year_candidates.join(", ")})`;
    }
    rows.push(el("dt", {}, label), el("dd", {}, String(v), sources[key] ? el("span", { class: "badge src", title: "Source" }, sourceText(sources[key])) : null));
  }
  if (dec.model_year == null && Array.isArray(dec.model_year_candidates) && dec.model_year_candidates.length) {
    rows.push(
      el("dt", {}, "Model year"),
      el("dd", {}, `one of ${dec.model_year_candidates.join(", ")}`, sources.model_year_candidates ? el("span", { class: "badge src" }, sourceText(sources.model_year_candidates)) : el("span", { class: "badge src" }, "position 10")),
    );
  }
  const KEY = new Set(["Manufacturer", "Country", "Model", "Model year"]);
  const main = [];
  for (let i = 0; i < rows.length; i += 2) if (KEY.has(rows[i].textContent)) main.push(rows[i].cloneNode(true), rows[i + 1].cloneNode(true));
  for (const b of main.flatMap((n) => [...n.querySelectorAll(".src")])) b.remove();
  if (main.length) box.append(el("dl", { class: "kv" }, main));
  if (rows.length) box.append(el("details", { class: "more" }, el("summary", {}, "Details"), el("dl", { class: "kv" }, rows)));
  else box.append(el("p", { class: "muted small" }, "No decode fields were found for this VIN."));
  const notes = Array.isArray(dec.notes) ? dec.notes : [];
  if (notes.length) {
    box.append(el("details", { class: "more" }, el("summary", {}, `Decode notes (${notes.length})`), el("ul", { class: "notes" }, notes.map((n) => el("li", {}, n)))));
  }
}

// ---------- spin viewer

const spin = {
  urls: [],
  names: [],
  idx: 0,
  images: [],
  frameSize: null,
  pinsByFrame: new Map(),
  bound: false,
  drag: null,
  timer: 0, // auto-turn interval
  sevByKey: null, // "part|damage" -> severity, for pin colours
};

function setupSpin(media, extra) {
  const urls = (Array.isArray(media.spin) && media.spin.length ? media.spin : Array.isArray(media.frames) ? media.frames : [])
    .map(safeHref)
    .filter(Boolean);
  spin.urls = urls;
  spin.names = urls.map(baseName);
  spin.idx = 0;
  spin.frameSize = Array.isArray(extra.frame_size) && extra.frame_size.length >= 2 ? extra.frame_size.map(Number) : null;
  spin.pinsByFrame = new Map();

  const fromFrames = !(Array.isArray(media.spin) && media.spin.length);
  const empty = $("#spin-empty");
  const wrap = $("#spin-wrap");
  if (!urls.length) {
    empty.textContent = "The server returned no spin frames for this inspection.";
    empty.hidden = false;
    wrap.hidden = true;
    $("#spin-count").textContent = "";
    return;
  }
  empty.hidden = !fromFrames;
  if (fromFrames) empty.textContent = "No dedicated spin sequence was returned; showing the extracted frames instead.";
  wrap.hidden = false;
  $("#spin-count").textContent = `${urls.length} frames`;

  // Index damage pins by frame name.
  const pins = Array.isArray(extra.pins2d) ? extra.pins2d : [];
  const keys = new Set(spin.names.map(stemKey));
  let offSpin = 0;
  for (const p of pins) {
    if (!p || !Array.isArray(p.box) || p.box.length < 4) continue;
    const k = stemKey(p.frame);
    if (!keys.has(k)) {
      offSpin += 1;
      continue;
    }
    if (!spin.pinsByFrame.has(k)) spin.pinsByFrame.set(k, []);
    spin.pinsByFrame.get(k).push(p);
  }
  const note = $("#spin-pin-note");
  if (!spin.frameSize && pins.length) {
    note.textContent = "Damage pins cannot be placed: the server did not report the frame size.";
    note.hidden = false;
  } else if (offSpin) {
    note.textContent = `${offSpin} more detection${offSpin === 1 ? " is" : "s are"} on other frames (see Damage).`;
    note.hidden = false;
  } else {
    note.hidden = true;
  }

  const range = $("#spin-range");
  range.max = String(urls.length - 1);
  range.value = "0";

  // Chips for frames with pins.
  const chips = $("#spin-pin-frames");
  chips.replaceChildren();
  spin.names.forEach((name, i) => {
    const list = spin.pinsByFrame.get(stemKey(name));
    if (!list || !spin.frameSize) return;
    chips.append(el("button", { class: "chip", type: "button", "data-idx": String(i), onclick: () => showSpin(i) }, `View ${i + 1} · ${list.length} box${list.length === 1 ? "" : "es"}`));
  });

  // Preload frames.
  const loading = $("#spin-loading");
  let loaded = 0;
  loading.hidden = false;
  loading.textContent = `Loading frames 0/${urls.length}…`;
  spin.images = urls.map((u) => {
    const img = new Image();
    img.decoding = "async";
    const finish = () => {
      loaded += 1;
      loading.textContent = `Loading frames ${loaded}/${urls.length}…`;
      if (loaded >= urls.length) loading.hidden = true;
    };
    img.onload = finish;
    img.onerror = finish;
    img.src = u;
    return img;
  });

  const first = spin.images[0];
  const sizeFrom = (w, h) => {
    if (!w || !h) return;
    const frame = $("#spin-frame");
    frame.style.aspectRatio = `${w} / ${h}`;
    frame.style.width = `min(100%, calc(var(--stage-h) * ${(w / h).toFixed(4)}))`;
  };
  if (spin.frameSize) sizeFrom(spin.frameSize[0], spin.frameSize[1]);
  first.addEventListener("load", () => sizeFrom(first.naturalWidth, first.naturalHeight), { once: true });

  bindSpinEvents();
  showSpin(0);
}

function showSpin(i) {
  const n = spin.urls.length;
  if (!n) return;
  spin.idx = ((i % n) + n) % n;
  const img = $("#spin-img");
  img.src = spin.urls[spin.idx];
  img.alt = `Walkaround frame ${spin.idx + 1} of ${n}`;
  $("#spin-range").value = String(spin.idx);
  const pins = spin.pinsByFrame.get(stemKey(spin.names[spin.idx])) || [];
  $("#spin-caption").textContent = `View ${spin.idx + 1} of ${n}${pins.length ? ` · ${pins.length} damage box${pins.length === 1 ? "" : "es"}` : ""}`;
  for (const c of document.querySelectorAll("#spin-pin-frames .chip")) c.classList.toggle("active", Number(c.dataset.idx) === spin.idx);
  renderSpinPins(pins);
}

function renderSpinPins(pins) {
  const layer = $("#spin-pins");
  layer.replaceChildren();
  if (!spin.frameSize) return;
  const [fw, fh] = spin.frameSize;
  if (!(fw > 0 && fh > 0)) return;
  for (const p of pins) {
    const [x1, y1, x2, y2] = p.box.map(Number);
    if (![x1, y1, x2, y2].every(Number.isFinite)) continue;
    const left = Math.max(0, Math.min(x1, x2)) / fw;
    const top = Math.max(0, Math.min(y1, y2)) / fh;
    const w = Math.abs(x2 - x1) / fw;
    const h = Math.abs(y2 - y1) / fh;
    const score = Number(p.score);
    const sev = (spin.sevByKey && spin.sevByKey.get(`${p.part}|${p.damage}`)) || "none";
    const low = Number.isFinite(score) && score < LOW_CONFIDENCE;
    const box = el(
      "div",
      { class: `pin-box sev-${sev}${low ? " low" : ""}${top < 0.08 ? " label-below" : ""}`, style: { left: pct(left), top: pct(top), width: pct(w), height: pct(h) } },
      el("span", { class: "pin-label" }, `${humanize(p.damage)} on ${humanize(p.part)}${Number.isFinite(score) ? ` · ${score.toFixed(2)}` : ""}`),
    );
    layer.append(box);
  }
}

function toggleSpinPlay(force) {
  const on = force === undefined ? !spin.timer : Boolean(force) && spin.urls.length > 1;
  clearInterval(spin.timer);
  spin.timer = on ? setInterval(() => showSpin(spin.idx + 1), 140) : 0;
  const b = $("#spin-play");
  b.setAttribute("aria-pressed", String(Boolean(spin.timer)));
  b.classList.toggle("active", Boolean(spin.timer));
}

function bindSpinEvents() {
  if (spin.bound) return;
  spin.bound = true;
  const stage = $("#spin-stage");
  stage.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    toggleSpinPlay(false);
    spin.drag = { x: e.clientX, idx: spin.idx, id: e.pointerId };
    stage.classList.add("dragging");
    try {
      stage.setPointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
  });
  stage.addEventListener("pointermove", (e) => {
    if (!spin.drag || e.pointerId !== spin.drag.id) return;
    const n = spin.urls.length;
    if (n < 2) return;
    const perFrame = Math.max(4, stage.clientWidth / n); // one full drag across = one full turn
    const d = Math.round((e.clientX - spin.drag.x) / perFrame);
    const target = spin.drag.idx - d;
    if ((((target % n) + n) % n) !== spin.idx) showSpin(target);
  });
  const end = (e) => {
    if (!spin.drag || e.pointerId !== spin.drag.id) return;
    spin.drag = null;
    stage.classList.remove("dragging");
  };
  stage.addEventListener("pointerup", end);
  stage.addEventListener("pointercancel", end);
  stage.addEventListener("lostpointercapture", end);
  stage.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") {
      showSpin(spin.idx - 1);
      e.preventDefault();
    } else if (e.key === "ArrowRight") {
      showSpin(spin.idx + 1);
      e.preventDefault();
    }
  });
  $("#spin-prev").addEventListener("click", () => {
    toggleSpinPlay(false);
    showSpin(spin.idx - 1);
  });
  $("#spin-next").addEventListener("click", () => {
    toggleSpinPlay(false);
    showSpin(spin.idx + 1);
  });
  $("#spin-play").addEventListener("click", () => toggleSpinPlay());
  $("#spin-range").addEventListener("input", (e) => {
    toggleSpinPlay(false);
    showSpin(Number(e.target.value));
  });
}

// ---------- damage list

const ACTION_TEXT = { polish: "Polish", paint: "Paint", repair: "Repair", replace: "Replace" };

function costCell(cost) {
  if (!cost) return el("span", { class: "muted" }, "no price data");
  if (cost.price_uzs == null) {
    return el(
      "details",
      { class: "cost-details" },
      el("summary", { class: "muted" }, "no price data"),
      el("div", { class: "basis" }, cost.basis || "No matching row in the price table."),
    );
  }
  const kind = cost.price_kind || "from";
  const label = kind === "from" ? `from ${fmtUZS(cost.price_uzs)}` : `${fmtUZS(cost.price_uzs)} (${kind})`;
  return el(
    "details",
    { class: "cost-details" },
    el("summary", {}, label, cost.is_estimate ? el("span", { class: "badge warn", style: { marginLeft: "0.35rem" } }, "estimate") : null),
    el(
      "div",
      { class: "basis" },
      `Basis: ${cost.basis || "not reported"}`,
      cost.is_estimate ? el("div", {}, "This price is our estimate, not an observed price for this exact service.") : null,
    ),
  );
}

function isLowConfidence(f) {
  const score = Number(f && f.max_score);
  return Number.isFinite(score) && score < LOW_CONFIDENCE; // no score reported -> not hidden
}

function isWindshieldGlass(f) {
  const dmg = String((f && f.damage) || "").toLowerCase().replace(/[_-]+/g, " ");
  const part = String((f && f.part) || "").toLowerCase();
  return dmg.includes("glass shatter") && /windshield|windscreen|front.?glass/.test(part);
}

function findingRows(list, costFor, counter) {
  return list.map((f) => {
    const cost = costFor(f);
    if (!cost || cost.price_uzs == null) counter.unpriced += 1;
    else counter.sum += Number(cost.price_uzs);
    const sev = f.severity || null;
    return el(
      "tr",
      { "data-part": f.part },
      el("td", { "data-label": "Part" }, el("strong", {}, sentence(f.part))),
      el(
        "td",
        { "data-label": "Damage" },
        el(
          "span",
          {},
          humanize(f.damage),
          isWindshieldGlass(f) ? el("div", { class: "small finding-note" }, "Often a reflection.") : null,
        ),
      ),
      el(
        "td",
        { "data-label": "Severity" },
        sev ? el("span", { class: `badge sev-${sev}`, title: f.severity_rule || "" }, sev) : el("span", { class: "muted" }, "not rated"),
      ),
      el("td", { "data-label": "Action" }, f.action ? ACTION_TEXT[f.action] || humanize(f.action) : el("span", { class: "muted" }, "none")),
      el("td", { "data-label": "Cost", class: "num" }, costCell(cost)),
    );
  });
}

function findingTable(rows) {
  return el(
    "div",
    { class: "table-wrap" },
    el(
      "table",
      { class: "data responsive" },
      el("thead", {}, el("tr", {}, el("th", {}, "Part"), el("th", {}, "Damage"), el("th", {}, "Severity"), el("th", {}, "Action"), el("th", { class: "num" }, "Cost"))),
      el("tbody", {}, rows),
    ),
  );
}

function renderDamage(r) {
  const findings = Array.isArray(r.findings) ? r.findings : [];
  const costs = Array.isArray(r.costs) ? r.costs : [];
  const body = $("#damage-body");
  body.replaceChildren();
  const main = findings.filter((f) => !isLowConfidence(f));
  const low = findings.filter(isLowConfidence);
  $("#damage-count").textContent = low.length ? `${main.length} + ${low.length} to verify` : String(main.length);

  const usedCosts = new Set();
  const costFor = (f) => {
    const i = costs.findIndex((c, j) => !usedCosts.has(j) && c.part === f.part && c.damage === f.damage);
    if (i < 0) return null;
    usedCosts.add(i);
    return costs[i];
  };
  const mainSum = { sum: 0, unpriced: 0 };
  const lowSum = { sum: 0, unpriced: 0 };

  if (!findings.length) {
    body.append(el("p", {}, "No damage was detected in the walkaround frames."));
    body.append(el("p", { class: "small muted" }, "This means the models found nothing above their thresholds, not that the car is guaranteed damage-free. Check the 360° view."));
  } else {
    if (main.length) {
      body.append(findingTable(findingRows(main, costFor, mainSum)));
    } else {
      body.append(el("p", {}, `No finding reached the confidence threshold (score ${LOW_CONFIDENCE.toFixed(2)}).`));
    }
    if (low.length) {
      body.append(
        el(
          "details",
          { class: "more low-conf" },
          el("summary", {}, `Uncertain - please check (${low.length})`),
          el("p", { class: "small muted" }, `Score below ${LOW_CONFIDENCE.toFixed(2)}. Often reflections or dirt. Not in the total.`),
          findingTable(findingRows(low, costFor, lowSum)),
        ),
      );
    }

    const rules = findings.filter((f) => f.severity_rule);
    if (rules.length) {
      body.append(
        el(
          "details",
          { class: "more" },
          el("summary", {}, "How severity was rated"),
          el("ul", { class: "notes" }, rules.map((f) => el("li", {}, `${humanize(f.damage)} on ${humanize(f.part)}: ${f.severity_rule}`))),
        ),
      );
    }
  }

  // Cost lines that did not match a finding (should not happen, but never hide data).
  const extraCosts = costs.filter((_, j) => !usedCosts.has(j));
  if (extraCosts.length && findings.length) {
    body.append(
      el(
        "details",
        { class: "more" },
        el("summary", {}, `Other cost lines (${extraCosts.length})`),
        el("ul", { class: "notes" }, extraCosts.map((c) => el("li", {}, `${humanize(c.damage)} on ${humanize(c.part)} · ${humanize(c.action)} · ${c.price_uzs != null ? `from ${fmtUZS(c.price_uzs)}` : "no price data"} · ${c.basis || ""}`))),
      ),
    );
  }

  state.damageTotals = { sum: mainSum.sum, unpriced: mainSum.unpriced };
  const tileSub = [];
  if (main.length && mainSum.sum > 0) tileSub.push(`repairs from ${fmtUZS(mainSum.sum)}`);
  if (low.length) tileSub.push(`+ ${low.length} low confidence, to verify`);
  setTile("defects", !findings.length ? "None found" : main.length ? `${main.length} confirmed` : "None confirmed", tileSub.join(" · "), main.length ? "warn" : "ok");
  setTabCount("defects", main.length ? String(main.length) : "", "warn");

  // Totals are summed here from the same cost lines the server returned (its total_from_uzs covers all findings).
  const total = $("#damage-total");
  const anyPriced = costs.some((c) => c && c.price_uzs != null);
  if (!findings.length) {
    total.hidden = true;
    return;
  }
  if (!anyPriced) {
    total.replaceChildren(el("span", {}, "Estimated repair total"), el("span", { class: "muted" }, "no price data"));
    total.hidden = false;
    return;
  }
  const missing = mainSum.unpriced;
  const lines = [];
  if (missing > 0) lines.push(el("div", { class: "small muted" }, `${missing} item${missing === 1 ? "" : "s"} without a price`));
  if (low.length) {
    const all = mainSum.sum + lowSum.sum;
    lines.push(el("div", { class: "small muted" }, `with uncertain items: from ${fmtUZS(all)}`));
  }
  total.replaceChildren(
    el("span", {}, "Estimated repair total", ...lines),
    main.length && mainSum.sum > 0 ? el("strong", {}, `from ${fmtUZS(mainSum.sum)}`) : el("span", { class: "muted" }, main.length ? "no price data" : "nothing above the threshold"),
  );
  total.hidden = false;
}

// ---------- diff & history

function diffItemText(x) {
  if (x == null) return "";
  if (typeof x !== "object") return String(x);
  const f = x.finding || x.after || x.current || x;
  const head = f.damage || f.part ? `${humanize(f.damage)} on ${humanize(f.part)}` : JSON.stringify(x);
  const prev = x.previous_severity ?? (x.before && x.before.severity) ?? x.old_severity;
  const cur = f.severity ?? x.severity ?? x.new_severity;
  if (!cur) return head;
  return `${head} (${cur}${prev && prev !== cur ? `, was ${prev}` : ""})`;
}

function renderDiff(job, r) {
  const box = $("#diff-body");
  box.replaceChildren();
  const diff = pick(job, "diff");
  const nNew = diff && Array.isArray(diff.new) ? diff.new.length : 0;
  setTabCount("vin", nNew ? String(nNew) : "", "err");
  if (!diff || typeof diff !== "object") setTile("diff", "Not compared", r.vin ? "no comparison returned" : "the VIN was not read");
  else if (diff.first_inspection) setTile("diff", "First inspection", "of this VIN with AvtoPassport", "info");
  else if (!Array.isArray(diff.new)) setTile("diff", "Unknown", "the server did not report new damage");
  else setTile("diff", nNew ? `${nNew} new damage` : "No new damage", diff.previous_created_utc ? `compared with ${fmtDateShort(diff.previous_created_utc)}` : "", nNew ? "err" : "ok");
  if (!diff || typeof diff !== "object") {
    box.append(
      el("p", { class: "muted" }, r.vin ? "The server returned no comparison with earlier inspections." : "No comparison: the VIN was not read, so this inspection cannot be linked to earlier ones."),
    );
    return;
  }
  if (diff.first_inspection) {
    box.append(el("p", {}, el("span", { class: "badge info" }, "First inspection of this VIN")));
    box.append(el("p", { class: "small muted" }, "The next inspection will show what changed."));
    return;
  }
  if (!Array.isArray(diff.new)) {
    box.append(el("p", { class: "muted" }, "The server did not report which damage is new."));
  } else if (!diff.new.length) {
    box.append(el("p", {}, el("span", { class: "badge ok" }, "No new damage since the last inspection")));
  }
  const groups = [
    ["new", "New damage since last inspection"],
    ["worse", "Got worse"],
    ["resolved", "No longer seen (repaired or not visible)"],
  ];
  for (const [key, title] of groups) {
    const items = Array.isArray(diff[key]) ? diff[key] : [];
    if (!items.length) continue;
    box.append(el("div", { class: `diff-group ${key}` }, el("h3", {}, `${title} (${items.length})`), el("ul", {}, items.map((x) => el("li", {}, diffItemText(x))))));
  }
  const uncertain = Array.isArray(diff.uncertain) ? diff.uncertain : [];
  if (uncertain.length) {
    box.append(el("div", { class: "diff-group uncertain" }, el("h3", {}, `Now uncertain (${uncertain.length})`), el("ul", {}, uncertain.map((x) => el("li", {}, diffItemText(x))))));
  }
  const unchanged = Array.isArray(diff.unchanged) ? diff.unchanged : [];
  if (unchanged.length) {
    box.append(el("details", { class: "more" }, el("summary", {}, `Unchanged (${unchanged.length})`), el("ul", { class: "notes" }, unchanged.map((x) => el("li", {}, diffItemText(x))))));
  }
  const possible = Array.isArray(diff.possible_changes) ? diff.possible_changes : [];
  if (possible.length) {
    box.append(el("details", { class: "more" },
      el("summary", {}, `Possible changes - please check (${possible.length})`),
      el("ul", { class: "notes" }, possible.map((x) => el("li", {}, `${diffItemText(x)} (${x.direction === "appeared" ? "appeared" : "disappeared"})`)))));
  }
  if (diff.previous_created_utc) box.append(el("p", { class: "small muted" }, `Compared with the inspection of ${fmtDateShort(diff.previous_created_utc)}.`));
  if (diff.rule) box.append(el("details", { class: "more" }, el("summary", {}, "How changes are judged"), el("p", { class: "small muted" }, `${diff.rule}.`)));
}

async function renderHistory(r, job) {
  const box = $("#history-body");
  box.replaceChildren();
  if (!r.vin || r.vin_status === "could_not_read") return;
  try {
    const h = await apiJSON(`/api/vin/${encodeURIComponent(r.vin)}/history`);
    const list = (Array.isArray(h.inspections) ? h.inspections : []).slice().sort((a, b) => String(b.created_utc).localeCompare(String(a.created_utc)));
    if (!list.length) return;
    const currentIds = new Set([String(state.jobId), String(job.inspection_id)]);
    const item = (it) => {
      const isCurrent = currentIds.has(String(it.id)) || currentIds.has(String(it.job_id));
      const u = new URL(location.href);
      u.search = "";
      u.hash = "";
      u.searchParams.set("inspection", it.job_id ?? it.id); // the API serves jobs by job id
      return el("li", { class: isCurrent ? "current" : "" },
        el("span", { class: "tl-date" }, fmtDateShort(it.created_utc) || "?"),
        el("span", { class: "tl-what" }, `${it.n_findings ?? "?"} finding${it.n_findings === 1 ? "" : "s"}`),
        isCurrent ? el("span", { class: "badge info" }, "this inspection") : el("a", { href: u.pathname + u.search }, "Open"));
    };
    box.append(el("h3", { class: "tl-title" }, `Inspection history (${list.length})`), el("ol", { class: "timeline" }, list.slice(0, 3).map(item)));
    if (list.length > 3) box.append(el("details", { class: "more" }, el("summary", {}, `Show all (${list.length})`), el("ol", { class: "timeline" }, list.slice(3).map(item))));
    box.append(el("p", { class: "small muted", style: { marginTop: "0.6rem" } }, "Owners, accidents, insurance: ",
      el("a", { href: "sandbox.html" }, "registry demo (synthetic data)")));
  } catch (e) {
    box.replaceChildren(el("p", { class: "small error" }, `Could not load the history: ${e.message}`));
  }
}

// ---------- paint try-on

// The try-on has its own view picker, so it works without going back to the 360° view in the Defects tab.
const tryon = { idx: 0, colour: null, hex: null };

async function setupTryon() {
  const status = $("#palette-status");
  const sw = $("#swatches");
  $("#tryon-figure").hidden = true;
  $("#tryon-controls").hidden = true;
  $("#tryon-status").textContent = "";
  tryon.colour = null;
  tryon.hex = null;
  if (!spin.urls.length) {
    status.textContent = "Try-on needs a walkaround frame, and none was returned.";
    sw.replaceChildren();
    setTile("tryon", "", "no frames to recolour");
    return;
  }
  $("#tryon-range").max = String(spin.urls.length - 1);
  $("#tryon-controls").hidden = false;
  showTryonFrame(0);
  if (!state.palette) {
    status.textContent = "Loading colours…";
    try {
      const p = await apiJSON("/api/palette");
      state.palette = Array.isArray(p) ? p : Array.isArray(p.colours) ? p.colours : [];
    } catch (e) {
      status.textContent = "";
      sw.replaceChildren(el("p", { class: "error" }, `Could not load the colour palette: ${e.message}`));
      return;
    }
  }
  const palette = state.palette;
  status.textContent = palette.length ? "" : "The server returned an empty palette.";
  setTile("tryon", "", palette.length ? `${palette.length} colours on the real photos` : "");
  const original = el(
    "button",
    {
      class: "swatch active",
      type: "button",
      onclick: (e) => {
        markSwatch(e.currentTarget);
        tryon.colour = null;
        tryon.hex = null;
        showOriginal();
      },
    },
    el("span", { class: "dot original", "aria-hidden": "true" }),
    "Original",
  );
  sw.replaceChildren(
    original,
    ...palette.map((c) => {
      const hex = String(c.hex || "").replace(/^#/, "");
      const valid = /^[0-9a-f]{6}$/i.test(hex);
      return el(
        "button",
        {
          class: "swatch",
          type: "button",
          title: `${c.name || hex}${c.model ? " · " + c.model : ""}`,
          disabled: !valid,
          onclick: (e) => {
            markSwatch(e.currentTarget);
            tryon.colour = c;
            tryon.hex = hex;
            loadTryon(c, hex);
          },
        },
        el("span", { class: "dot", style: { background: valid ? `#${hex}` : "transparent" }, "aria-hidden": "true" }),
        el("span", {}, c.name || `#${hex}`),
        c.model ? el("span", { class: "muted" }, c.model) : null,
      );
    }),
  );
}

function markSwatch(btn) {
  for (const b of document.querySelectorAll("#swatches .swatch")) b.classList.toggle("active", b === btn);
}

function showTryonFrame(i) {
  const n = spin.urls.length;
  if (!n) return;
  tryon.idx = ((i % n) + n) % n;
  $("#tryon-range").value = String(tryon.idx);
  if (tryon.hex) loadTryon(tryon.colour, tryon.hex);
  else showOriginal();
}

function viewText() {
  return `view ${tryon.idx + 1} of ${spin.urls.length}`;
}

function showOriginal() {
  state.tryonToken += 1;
  $("#tryon-img").src = spin.urls[tryon.idx];
  $("#tryon-label").textContent = `Original · ${viewText()}`;
  $("#tryon-status").textContent = "";
  $("#tryon-figure").hidden = false;
}

async function loadTryon(c, hex) {
  const frame = spin.names[tryon.idx];
  if (!frame || !state.jobId) return;
  const token = ++state.tryonToken;
  const st = $("#tryon-status");
  st.replaceChildren(el("span", { class: "spinner", "aria-hidden": "true" }), ` Painting ${c.name || "#" + hex}…`);
  st.className = "small";
  const url = `${inspectionUrl(state.jobId)}/tryon?colour=${encodeURIComponent(hex)}&frame=${encodeURIComponent(frame)}`;
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      const text = await res.text();
      let data = null;
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }
      throw new Error(errorMessage(data, text, res.status, res.statusText));
    }
    const blob = await res.blob();
    if (token !== state.tryonToken) return;
    if (state.tryonUrl) URL.revokeObjectURL(state.tryonUrl);
    state.tryonUrl = URL.createObjectURL(blob);
    $("#tryon-img").src = state.tryonUrl;
    const src = safeHref(c.source_url);
    const label = $("#tryon-label");
    label.replaceChildren(`${c.name || "#" + hex} · ${viewText()}`);
    if (src) label.append(" · ", el("a", { href: src, target: "_blank", rel: "noopener" }, "source"));
    $("#tryon-figure").hidden = false;
    st.textContent = "";
  } catch (e) {
    if (token !== state.tryonToken) return;
    st.textContent = `Try-on failed: ${e.message}`;
    st.className = "small error";
  }
}

// ---------- 3D reconstruction

function reconOf(job) {
  const r = pick(job, "recon3d");
  return r && typeof r === "object" ? r : null;
}

const isReconFinal = (s) => s === "done" || s === "error" || s === "unavailable";

// A finished build of exactly the same 3D input (same blurred frames + settings) attached instead of rebuilt.
// Every number comes from the original build's stored state; nothing here is estimated.
function fmtBuildTime(sec) {
  const s = Number(sec);
  if (!Number.isFinite(s)) return null;
  return s < 120 ? `${s.toFixed(0)} s` : `${(s / 60).toFixed(s < 600 ? 1 : 0)} min`;
}

function fmtDevice(dev) {
  const d = String(dev || "");
  if (!d) return null;
  if (d.startsWith("cuda")) return "the GPU (CUDA)";
  if (d === "cpu") return "the CPU";
  return `the ${d}`;
}

function reuseNote(r) {
  if (!r || !r.reused) return null;
  const d = r.reused_from_created_utc ? new Date(r.reused_from_created_utc) : null;
  const when = d && !Number.isNaN(d.getTime()) ? fmtDateShort(r.reused_from_created_utc) : r.reused_from_created_utc || "an earlier date";
  const took = fmtBuildTime(r.original_seconds);
  const dev = fmtDevice(r.original_device);
  const orig = took ? ` (first built in ${took}${dev ? ` on ${dev}` : ""})` : "";
  return el("span", { class: "small muted", style: { display: "block" } },
    `Reused: same video as on ${when}.`);
}
const HQ_ACTIVE = new Set(["waiting", "queued", "running"]);

function hqOf(job) {
  const h = job && job.recon3d_hq;
  return h && typeof h === "object" ? h : null;
}

function keepWatching(job) {
  const r = reconOf(job);
  const h = hqOf(job);
  return Boolean((r && !isReconFinal(r.status)) || (h && HQ_ACTIVE.has(h.status)));
}

// The 3D models (the splat file is ~28 MB) are downloaded only once the 3D tab is opened.
function mountPending3d() {
  if (state.tab !== "model3d") return;
  const { cloud, hq, parts } = state.pending3d;
  if (parts && state.viewerPartsUrl !== parts) mountPartsViewer(parts);
  if (hq && state.viewerSplat !== hq.url) mountHqViewer(hq.url, hq.pins, hq.view);
  if (cloud && state.viewerPly !== cloud.url) mountViewer(cloud.url, cloud.pins, cloud.view);
}

function update3dSummary(job) {
  const r = reconOf(job);
  const h = hqOf(job);
  const hqOn = Boolean(h && ["waiting", "queued", "running", "done", "error"].includes(h.status));
  const building = Boolean((r && ["not_started", "queued", "running"].includes(r.status)) || (hqOn && HQ_ACTIVE.has(h.status)));
  const ready = [];
  if (hqOn && h.status === "done") ready.push("high-quality model");
  if (r && r.status === "done") ready.push("point cloud");
  if (ready.length) setTile("3d", "Ready", `${ready.join(" + ")}${building ? " · still building the rest" : ""}`, "ok");
  else if (building) setTile("3d", "Building…", "this can take several minutes", "info");
  else if (r && r.status === "unavailable") setTile("3d", "Not available", "on this server");
  else if ((r && r.status === "error") || (hqOn && h.status === "error")) setTile("3d", "Failed", "see the 3D model tab", "err");
  else setTile("3d", "Not reported", "");
  setTabCount("model3d", building ? "…" : "", "busy");
}

function watchRecon(token, job) {
  renderRecon(reconOf(job));
  renderHq(hqOf(job));
  update3dSummary(job);
  if (!keepWatching(job)) return;
  setTimeout(() => pollRecon(token, 1, 0), RECON_POLL_MS);
}

async function pollRecon(token, n, failures) {
  if (token !== state.pollToken) return;
  let job;
  try {
    job = await apiJSON(state.statusUrl);
  } catch (e) {
    if (token !== state.pollToken) return;
    if (failures < MAX_POLL_FAILURES) {
      $("#recon-status").textContent = `Could not check the 3D model status (${e.message}). Retrying…`;
      setTimeout(() => pollRecon(token, n + 1, failures + 1), RECON_POLL_MS);
    } else {
      $("#recon-status").textContent = `Stopped checking the 3D model status: ${e.message}. Reload the page to try again.`;
    }
    return;
  }
  if (token !== state.pollToken) return;
  state.job = job;
  renderRecon(reconOf(job));
  renderHq(hqOf(job));
  update3dSummary(job);
  if (!keepWatching(job)) return;
  const hqActive = Boolean(hqOf(job) && HQ_ACTIVE.has(hqOf(job).status));
  if (n >= (hqActive ? RECON_HQ_MAX_POLLS : RECON_MAX_POLLS)) {
    $("#recon-status").textContent = "Stopped watching after one hour; the 3D model is still not finished. Reload the page to check again.";
    return;
  }
  setTimeout(() => pollRecon(token, n + 1, 0), RECON_POLL_MS);
}

function renderRecon(r) {
  const st = $("#recon-status");
  const badge = $("#recon-badge");
  const errEl = $("#recon-error");
  errEl.hidden = true;
  if (!r) {
    badge.textContent = "";
    st.textContent = "The server did not report a 3D reconstruction status for this inspection.";
    return;
  }
  badge.textContent = humanize(r.status);
  switch (r.status) {
    case "not_started":
      badge.className = "badge";
      st.textContent = "Waiting to start…";
      break;
    case "queued":
    case "running":
      badge.className = "badge info";
      st.replaceChildren(el("span", { class: "spinner", "aria-hidden": "true" }), " Building the 3D model… (a few minutes)");
      break;
    case "done": {
      badge.className = "badge ok";
      const secs = Number(r.seconds);
      st.textContent = `Built${Number.isFinite(secs) ? ` in ${secs.toFixed(0)} s` : ""}.`;
      const reusedNote = reuseNote(r);
      if (reusedNote) st.append(reusedNote);
      const ply = safeHref(r.ply);
      if (!ply) {
        errEl.textContent = "The server reports the 3D model as done but returned no model file.";
        errEl.hidden = false;
      } else if (state.viewerPly !== ply && $("#recon-card").hidden === false) {
        state.pending3d.cloud = { url: ply, pins: confidentPins(r.pins), view: r.view || null };
        mountPending3d();
      }
      renderReconPins(Array.isArray(r.pins) ? r.pins : []);
      break;
    }
    case "error":
      badge.className = "badge err";
      st.textContent = "";
      errEl.textContent = `3D reconstruction failed: ${r.error || "the server did not return an error message."}`;
      errEl.hidden = false;
      break;
    case "unavailable":
      badge.className = "badge warn";
      st.textContent = `Not available on this server${r.error ? ` (${r.error})` : ""}. Use the 360° view in the Car tab.`;
      break;
    default:
      badge.className = "badge";
      st.textContent = `Unknown 3D status from the server: ${String(r.status)}`;
  }
}

// Pins drawn in the 3D viewers: same LOW_CONFIDENCE rule as the damage list (no score reported -> drawn).
function confidentPins(pins) {
  return (Array.isArray(pins) ? pins : []).filter((p) => !(Number.isFinite(Number(p && p.score)) && Number(p.score) < LOW_CONFIDENCE));
}

function renderReconPins(pins) {
  const box = $("#recon-pins");
  box.replaceChildren();
  if (!pins.length) {
    box.append(el("p", { class: "small muted" }, "No damage pins in 3D."));
    return;
  }
  box.append(
    el("h3", { style: { marginTop: "0.75rem" } }, `Damage pins in 3D (${pins.length})`),
    el(
      "ul",
      { class: "notes" },
      pins.map((p) => {
        const score = Number(p.score);
        const low = Number.isFinite(score) && score < LOW_CONFIDENCE;
        return el(
          "li",
          { class: low ? "muted" : "" },
          `${humanize(p.damage)} on ${humanize(p.part)}${p.n_views != null ? ` · seen in ${p.n_views} view${p.n_views === 1 ? "" : "s"}` : ""}${Number.isFinite(score) ? ` · score ${score.toFixed(2)}` : ""}${low ? " · low confidence, not drawn in 3D" : ""}`,
        );
      }),
    ),
  );
}

async function mountViewer(plyUrl, pins, view) {
  state.viewerPly = plyUrl;
  const box = $("#viewer3d");
  const errEl = $("#recon-error");
  box.hidden = false;
  box.replaceChildren(el("p", { class: "small muted" }, "Loading the 3D viewer…"));
  try {
    const mod = await import("./viewer3d.js");
    if (state.viewer) state.viewer.dispose();
    state.viewer = await mod.createViewer(box, { plyUrl, pins, view });
  } catch (e) {
    console.error(e);
    box.hidden = true;
    errEl.textContent = `The 3D model is ready but could not be shown in this browser: ${e && e.message ? e.message : e}`;
    errEl.hidden = false;
    state.viewerPly = null;
  }
}

// ---------- optional high-quality 3D (Gaussian splats, GPU worker only)

function fmtMin(sec) {
  const s = Number(sec);
  return Number.isFinite(s) ? `${(s / 60).toFixed(s < 600 ? 1 : 0)} min` : null;
}

function renderParts3d(h) {
  // ready right after the camera poses, while the splats may still be training
  const partsUrl = safeHref(h && h.parts3d);
  $("#parts3d-card").hidden = !partsUrl;
  $("#recon-card").hidden = Boolean(partsUrl); // the parts model replaces the quick point cloud
  if (partsUrl) state.pending3d.cloud = null;
  if (!partsUrl) return;
  if (state.viewerPartsUrl !== partsUrl) {
    state.pending3d.parts = partsUrl;
    mountPending3d();
  }
  if (state.ovPartsUrl !== partsUrl) mountOverviewParts(partsUrl);
}

function renderHq(h) {
  renderParts3d(h);
  const card = $("#hq-card");
  // disabled / unavailable / not started: this server has no GPU worker for it, so nothing is shown (no fake).
  if (!h || !["waiting", "queued", "running", "done", "error"].includes(h.status)) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const st = $("#hq-status");
  const badge = $("#hq-badge");
  const errEl = $("#hq-error");
  errEl.hidden = true;
  badge.textContent = humanize(h.status);
  const prev = h.previous_run && typeof h.previous_run === "object" ? h.previous_run : null;
  switch (h.status) {
    case "waiting":
    case "queued":
    case "running": {
      badge.className = "badge info";
      let text = h.status === "waiting" ? " Waiting for the quick model first" : " Building on the GPU";
      const prevMin = prev ? fmtMin(prev.seconds_total) : null;
      if (prevMin) text += ` (last build took ${prevMin})`;
      st.replaceChildren(el("span", { class: "spinner", "aria-hidden": "true" }), `${text}…`);
      break;
    }
    case "done": {
      badge.className = "badge ok";
      const rep = h.report && typeof h.report === "object" ? h.report : {};
      const parts = [];
      const total = fmtMin(h.seconds);
      if (total) parts.push(`built in ${total}`);
      if (rep.n_registered != null && rep.n_images != null) parts.push(`${rep.n_registered}/${rep.n_images} frames calibrated by structure-from-motion`);
      if (Number.isFinite(Number(rep.mean_reprojection_error_px))) parts.push(`mean reprojection error ${Number(rep.mean_reprojection_error_px).toFixed(2)} px`);
      if (Number.isFinite(Number(rep.psnr))) {
        const pe = rep.psnr_eval && typeof rep.psnr_eval === "object" ? rep.psnr_eval : {};
        const nv = Number.isFinite(Number(pe.n_views)) ? ` on ${pe.n_views} eval views` : "";
        const car = Number.isFinite(Number(rep.psnr_car)) ? `, ${Number(rep.psnr_car).toFixed(1)} dB on the car` : "";
        parts.push(`PSNR ${Number(rep.psnr).toFixed(1)} dB whole image${car}${nv}`);
      }
      if (rep.steps) parts.push(`${Number(rep.steps).toLocaleString()} training steps`);
      const reg = rep.n_registered != null && rep.n_images != null ? ` from ${rep.n_registered} frames` : "";
      st.textContent = total ? `Built in ${total}${reg}.` : "Built.";
      $("#hq-details").replaceChildren(...parts.map((p) => el("li", {}, p)));
      const hqReuse = reuseNote(h);
      if (hqReuse) st.append(hqReuse);
      if (h.pins_note) $("#hq-details").append(el("li", {}, `${h.pins_note}.`));
      const crop = rep.crop && typeof rep.crop === "object" ? rep.crop : null;
      const full = safeHref(h.splat_full);
      const note = $("#hq-crop");
      note.replaceChildren();
      if (h.cropped && crop && Number.isFinite(Number(crop.n_splats_car))) {
        note.append(
          `Cropped to the car: ${Number(crop.n_splats_car).toLocaleString()} of ${Number(crop.n_splats).toLocaleString()} Gaussians kept.`,
          full ? el("span", {}, " ", el("a", { href: full, download: "" }, "Download the uncropped model (.ply)")) : null,
        );
      }
      const splat = safeHref(h.splat);
      if (!splat) {
        errEl.textContent = "The server reports the high-quality 3D as done but returned no model file.";
        errEl.hidden = false;
      } else if (state.viewerSplat !== splat) {
        state.pending3d.hq = { url: splat, pins: confidentPins(h.pins), view: h.view || null };
        mountPending3d();
      }
      break;
    }
    case "error":
      badge.className = "badge err";
      st.textContent = "";
      errEl.textContent = `High-quality 3D failed: ${h.error || "the server did not return an error message."}`;
      errEl.hidden = false;
      break;
    default:
      break;
  }
}

async function mountPartsViewer(url) {
  state.viewerPartsUrl = url;
  const box = $("#viewer-parts");
  box.replaceChildren(el("p", { class: "small muted" }, "Loading the parts model…"));
  try {
    const [mod, data] = await Promise.all([import("./partsviewer.js"), partsData(url)]);
    if (state.viewerParts) state.viewerParts.dispose();
    state.viewerParts = mod.createPartsViewer(box, data, { damage: state.damageByPart || {}, car: state.carName });
    const m = data.meta || {};
    $("#parts3d-status").textContent = `${(m.parts || []).length} parts${m.source ? ` from ${m.source}` : ""}. Slide "Take apart" to split the car.`;
  } catch (e) {
    console.error(e);
    box.replaceChildren(el("p", { class: "error" }, `The parts model could not be shown: ${e && e.message ? e.message : e}`));
    state.viewerPartsUrl = null;
  }
}

async function mountHqViewer(splatUrl, pins, view) {
  state.viewerSplat = splatUrl;
  const box = $("#viewer-hq");
  const errEl = $("#hq-error");
  box.hidden = false;
  box.replaceChildren(el("p", { class: "small muted" }, "Loading the splat viewer…"));
  try {
    const mod = await import("./viewer3d.js");
    if (state.viewerHq) state.viewerHq.dispose();
    state.viewerHq = await mod.createViewer(box, { splatUrl, pins, view });
  } catch (e) {
    console.error(e);
    box.hidden = true;
    errEl.textContent = `The high-quality 3D is ready but could not be shown in this browser: ${e && e.message ? e.message : e}`;
    errEl.hidden = false;
    state.viewerSplat = null;
  }
}

// ---------- body shops

function setShopsStatus(text, isError = false) {
  const s = $("#shops-status");
  s.textContent = text;
  s.className = `small${isError ? " error" : ""}`;
}

function findShopsHere() {
  if (!("geolocation" in navigator)) {
    setShopsStatus("Location is not available here. Use Tashkent centre.", true);
    return;
  }
  setShopsStatus("Asking for your location…");
  navigator.geolocation.getCurrentPosition(
    (pos) => fetchShops(pos.coords.latitude, pos.coords.longitude, "your location"),
    (err) => {
      const why = err && err.code === 1 ? "permission denied" : err && err.code === 3 ? "timed out" : (err && err.message) || "unavailable";
      setShopsStatus(`Location ${why}. Use Tashkent centre.`, true);
    },
    { enableHighAccuracy: false, timeout: 15000, maximumAge: 300000 },
  );
}

async function fetchShops(lat, lon, label) {
  setShopsStatus(`Searching within ${SHOP_RADIUS_M / 1000} km of ${label}…`);
  const list = $("#shops-list");
  const attr = $("#shops-attr");
  list.replaceChildren();
  attr.hidden = true;
  const q = new URLSearchParams({ lat: lat.toFixed(5), lon: lon.toFixed(5), radius_m: String(SHOP_RADIUS_M) });
  let data;
  try {
    data = await apiJSON(`/api/shops?${q}`);
  } catch (e) {
    setShopsStatus(`Body shop search is unavailable: ${e.message}`, true);
    return;
  }
  const shops = Array.isArray(data.shops) ? data.shops : [];
  setShopsStatus(
    shops.length
      ? `${shops.length} place${shops.length === 1 ? "" : "s"} within ${SHOP_RADIUS_M / 1000} km of ${label}.`
      : `No shops found within ${SHOP_RADIUS_M / 1000} km of ${label}.`,
  );
  for (const s of shops) {
    const tags = s.tags && typeof s.tags === "object" ? s.tags : {};
    const meta = [];
    const addr = [tags["addr:street"], tags["addr:housenumber"]].filter(Boolean).join(" ");
    if (addr) meta.push(addr);
    if (tags.opening_hours) meta.push(`Open ${tags.opening_hours}`);
    const links = [];
    if (Number.isFinite(Number(s.lat)) && Number.isFinite(Number(s.lon))) {
      const la = Number(s.lat).toFixed(6);
      const lo = Number(s.lon).toFixed(6);
      links.push(el("a", { href: `https://www.openstreetmap.org/?mlat=${la}&mlon=${lo}#map=18/${la}/${lo}`, target: "_blank", rel: "noopener" }, "Map"));
    }
    const phone = tags.phone || tags["contact:phone"];
    if (phone) links.push(el("a", { href: `tel:${String(phone).replace(/[^\d+]/g, "")}` }, String(phone)));
    const web = safeHref(tags.website || tags["contact:website"]);
    if (web) links.push(el("a", { href: web, target: "_blank", rel: "noopener" }, "Website"));
    list.append(
      el(
        "li",
        { class: "shop" },
        el("div", { class: "shop-head" }, el("span", { class: "shop-name" }, s.name || "Unnamed place"), el("span", { class: "shop-dist" }, fmtDist(Number(s.distance_m)))),
        meta.length ? el("div", { class: "shop-meta" }, meta.join(" · ")) : null,
        links.length ? el("div", { class: "shop-links" }, links) : null,
      ),
    );
  }
  const attrHref = safeHref(data.attribution_url) || "https://www.openstreetmap.org/copyright";
  attr.replaceChildren(
    "Data ",
    el("a", { href: attrHref, target: "_blank", rel: "noopener" }, "© OpenStreetMap contributors"),

  );
  attr.hidden = false;
}

// ---------- verification

function renderVerification(job, r, extra, media) {
  const kv = $("#verify-kv");
  kv.replaceChildren(
    el("dt", {}, "Fingerprint (SHA-256)"),
    el("dd", {}, r.frames_sha256 ? el("div", { class: "hash" }, r.frames_sha256) : el("span", { class: "muted" }, "not reported")),
    el("dt", {}, "Created"),
    el("dd", {}, fmtDateShort(r.created_utc) || "not reported"),
    el("dt", {}, "Inspection"),
    el("dd", {}, el("code", {}, String(job.id ?? state.jobId))),
  );
  if (Number.isFinite(Number(extra.frames))) kv.append(el("dt", {}, "Frames analysed"), el("dd", {}, String(extra.frames)));
  else if (Array.isArray(extra.frames)) kv.append(el("dt", {}, "Frames analysed"), el("dd", {}, String(extra.frames.length)));

  const links = $("#report-links");
  links.replaceChildren();
  const pdf = safeHref(media.pdf);
  setTile("report", pdf ? "PDF ready" : "No PDF", r.frames_sha256 ? `frames SHA-256 ${String(r.frames_sha256).slice(0, 12)}…` : "", pdf ? "ok" : "warn");
  if (pdf) links.append(el("a", { class: "btn btn-primary", href: pdfInLang(pdf), download: "" }, "Download PDF passport"));
  else links.append(el("span", { class: "muted small" }, "PDF not available for this inspection."));
  const report = pick(job, "report");
  if (typeof report === "string" && safeHref(report)) {
    links.append(el("a", { class: "btn", href: safeHref(report), target: "_blank", rel: "noopener" }, "JSON report"));
  } else if (report && typeof report === "object") {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    links.append(el("a", { class: "btn", href: URL.createObjectURL(blob), download: `avtopassport-${state.jobId}.json` }, "Download JSON report"));
  }

  // Processing details: timings, quality, privacy, models, notes.
  const pb = $("#processing-body");
  pb.replaceChildren();
  const t = r.timings_ms && typeof r.timings_ms === "object" ? r.timings_ms : {};
  const steps = Object.entries(t);
  if (steps.length) {
    const rows = steps.map(([name, v]) => {
      if (v && typeof v === "object") {
        return el(
          "tr",
          {},
          el("td", { "data-label": "Step" }, humanize(name)),
          el("td", { "data-label": "Count", class: "num" }, String(v.count ?? "")),
          el("td", { "data-label": "Total", class: "num" }, fmtMs(Number(v.total_ms))),
          el("td", { "data-label": "Median", class: "num" }, fmtMs(Number(v.median_ms))),
          el("td", { "data-label": "Max", class: "num" }, fmtMs(Number(v.max_ms))),
        );
      }
      return el("tr", {}, el("td", { "data-label": "Step" }, humanize(name)), el("td", { "data-label": "Count", class: "num" }, ""), el("td", { "data-label": "Total", class: "num" }, fmtMs(Number(v))), el("td", { "data-label": "Median", class: "num" }, ""), el("td", { "data-label": "Max", class: "num" }, ""));
    });
    pb.append(
      el("h3", { style: { marginTop: "0.5rem" } }, "Timings (measured on the server)"),
      el(
        "div",
        { class: "table-wrap" },
        el(
          "table",
          { class: "data responsive" },
          el("thead", {}, el("tr", {}, el("th", {}, "Step"), el("th", { class: "num" }, "Count"), el("th", { class: "num" }, "Total"), el("th", { class: "num" }, "Median"), el("th", { class: "num" }, "Max"))),
          el("tbody", {}, rows),
        ),
      ),
    );
  }
  for (const [key, title] of [
    ["privacy", "Privacy (blurring)"],
    ["quality", "Frame quality"],
    ["models", "Models"],
  ]) {
    if (extra[key] != null) pb.append(el("h3", { style: { marginTop: "0.75rem" } }, title), kvList(extra[key]));
  }
  const notes = Array.isArray(r.notes) ? r.notes : [];
  if (notes.length) pb.append(el("h3", { style: { marginTop: "0.75rem" } }, "Notes"), el("ul", { class: "notes" }, notes.map((n) => el("li", {}, String(n)))));
  if (!pb.childNodes.length) pb.append(el("p", { class: "muted small" }, "The server returned no processing details."));
}

// ---------------------------------------------------------------- share / reset

async function sharePassport() {
  const url = location.href;
  const status = $("#share-status");
  try {
    if (navigator.share) {
      await navigator.share({ title: "AvtoPassport", text: "Car condition passport", url });
      status.textContent = "";
      return;
    }
    await navigator.clipboard.writeText(url);
    status.textContent = "Link copied to the clipboard.";
  } catch (e) {
    if (e && e.name === "AbortError") return;
    status.textContent = `Copy this link: ${url}`;
  }
}

function resetAll() {
  location.href = location.pathname;
}

// ---------------------------------------------------------------- init

function init() {
  startTranslator(getLang());
  for (const b of document.querySelectorAll("#lang-switch button")) {
    b.setAttribute("aria-pressed", String(b.dataset.lang === getLang()));
    b.addEventListener("click", () => {
      if (b.dataset.lang === getLang()) return;
      setLang(b.dataset.lang);
      location.reload(); // every text is rendered again in the new language
    });
  }
  const repo = $("#repo-link");
  if (REPO_URL && safeHref(REPO_URL)) {
    repo.href = REPO_URL;
    repo.hidden = false;
  } else {
    repo.hidden = true;
  }

  for (const id of ["#video-camera", "#video-library"]) {
    $(id).addEventListener("change", (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) onVideoPicked(f);
    });
  }
  for (const id of ["#vin-camera", "#vin-library"]) {
    $(id).addEventListener("change", (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) onVinPhotoPicked(f);
    });
  }
  $("#vin-text").addEventListener("input", onVinTextInput);
  $("#demo-btn").addEventListener("click", useDemo);
  $("#capture-form").addEventListener("submit", (e) => {
    e.preventDefault();
    startUpload();
  });
  $("#retry-btn").addEventListener("click", resetAll);
  $("#new-btn").addEventListener("click", resetAll);
  $("#share-btn").addEventListener("click", sharePassport);
  $("#shops-geo").addEventListener("click", findShopsHere);
  $("#shops-tashkent").addEventListener("click", () => fetchShops(TASHKENT.lat, TASHKENT.lon, "Tashkent centre"));
  $("#tryon-prev").addEventListener("click", () => showTryonFrame(tryon.idx - 1));
  $("#tryon-next").addEventListener("click", () => showTryonFrame(tryon.idx + 1));
  let tryonTimer = 0;
  $("#tryon-range").addEventListener("input", (e) => {
    const i = Number(e.target.value);
    clearTimeout(tryonTimer);
    if (!tryon.hex) showTryonFrame(i);
    else tryonTimer = setTimeout(() => showTryonFrame(i), 250); // each recoloured view is a server request
  });
  bindTabs();
  state.initialTab = tabFromHash();
  $("#theme-btn").addEventListener("click", toggleTheme);
  $("#view-toggle").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-view]");
    if (!b) return;
    state.ovViewChosen = true;
    setOverviewView(b.dataset.view);
  });
  syncThemeButton();

  loadHealth();

  const id = new URLSearchParams(location.search).get("inspection");
  if (id) {
    $("#capture-title").textContent = "Loading inspection";
    setFormDisabled(true);
    startJob(id, null);
  }
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
else init();
