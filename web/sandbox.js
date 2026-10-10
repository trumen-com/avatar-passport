// Registry connector sandbox: lists the SYNTHETIC registry records served by the API.
// Every record view carries the SYNTHETIC DEMO DATA badge; nothing here is linked to scanned VINs.

const BADGE = "SYNTHETIC DEMO DATA";

const $ = (sel, root = document) => root.querySelector(sel);

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
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

function humanize(s) {
  if (s == null || s === "") return "";
  return String(s).replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
}

async function apiJSON(url) {
  let res;
  try {
    res = await fetch(url, { headers: { Accept: "application/json" }, cache: "no-store" });
  } catch (e) {
    throw new Error(`Network error: ${e && e.message ? e.message : e}`);
  }
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!res.ok) {
    const d = data && (data.detail ?? data.error ?? data.message);
    const msg = typeof d === "string" ? d : d ? JSON.stringify(d) : "";
    throw new Error(`HTTP ${res.status}${msg ? ": " + msg : ""}`);
  }
  if (data === null) throw new Error(`HTTP ${res.status}: the server did not return JSON`);
  return data;
}

function badge() {
  return el("span", { class: "synthetic-badge" }, BADGE);
}

function valueNode(v) {
  if (v == null || v === "") return el("span", { class: "muted" }, "—");
  if (typeof v === "boolean") return el("span", {}, v ? "yes" : "no");
  if (typeof v !== "object") return el("span", {}, String(v));
  if (Array.isArray(v)) {
    if (!v.length) return el("span", { class: "muted" }, "none");
    if (v.every((x) => x == null || typeof x !== "object")) return el("span", {}, v.map(String).join(", "));
    return objectTable(v);
  }
  return el(
    "dl",
    { class: "kv" },
    Object.entries(v).map(([k, x]) => [el("dt", {}, humanize(k)), el("dd", {}, valueNode(x))]),
  );
}

// Table for an array of objects; columns are the union of their keys.
function objectTable(rows) {
  const cols = [];
  for (const r of rows) {
    if (r && typeof r === "object") for (const k of Object.keys(r)) if (!cols.includes(k)) cols.push(k);
  }
  return el(
    "div",
    { class: "table-wrap" },
    el(
      "table",
      { class: "data" },
      el("thead", {}, el("tr", {}, cols.map((c) => el("th", {}, humanize(c))))),
      el(
        "tbody",
        {},
        rows.map((r) => el("tr", {}, cols.map((c) => el("td", { "data-label": humanize(c) }, valueNode(r && typeof r === "object" ? r[c] : r))))),
      ),
    ),
  );
}

const SECTIONS = [
  ["owners", "Owners"],
  ["accidents", "Accidents"],
  ["insurance_claims", "Insurance claims"],
  ["mileage_km", "Mileage (km)"],
];
const SUMMARY = ["vin", "make", "model", "model_year", "colour", "source", "synthetic"];

let items = [];
let activeVin = null;

function renderList() {
  const q = $("#filter").value.trim().toUpperCase();
  const list = $("#registry-list");
  const shown = items.filter((it) => !q || `${it.vin} ${it.make || ""} ${it.model || ""}`.toUpperCase().includes(q));
  list.replaceChildren(
    ...shown.map((it) =>
      el(
        "li",
        {},
        el(
          "button",
          {
            class: `registry-item${it.vin === activeVin ? " active" : ""}`,
            type: "button",
            onclick: () => openRecord(it.vin),
          },
          el("span", {}, el("code", {}, it.vin), el("div", { class: "small muted" }, [it.make, it.model].filter(Boolean).join(" ") || "—")),
          badge(),
        ),
      ),
    ),
  );
  $("#list-count").textContent = q ? `${shown.length} of ${items.length}` : String(items.length);
}

async function loadList() {
  const st = $("#list-status");
  st.textContent = "Loading synthetic records…";
  try {
    const data = await apiJSON("/api/sandbox/registry");
    items = Array.isArray(data) ? data : Array.isArray(data.records) ? data.records : [];
    if (data && !Array.isArray(data) && typeof data.notice === "string") {
      const n = $("#api-notice");
      n.textContent = `Server notice: ${data.notice}`;
      n.hidden = false;
    }
    const notSynthetic = items.filter((it) => it.synthetic !== true).length;
    st.textContent = items.length ? "" : "The sandbox registry is empty.";
    if (notSynthetic) {
      st.textContent = `Warning: ${notSynthetic} record(s) from the server are not flagged as synthetic.`;
      st.className = "small error";
    }
    renderList();
    const fromHash = decodeURIComponent(location.hash.slice(1));
    if (fromHash && items.some((it) => it.vin === fromHash)) openRecord(fromHash);
  } catch (e) {
    st.textContent = `Could not load the sandbox registry: ${e.message}`;
    st.className = "small error";
  }
}

async function openRecord(vin) {
  activeVin = vin;
  renderList();
  history.replaceState(null, "", `#${encodeURIComponent(vin)}`);
  const card = $("#record-card");
  const body = $("#record-body");
  card.hidden = false;
  $("#record-title").textContent = `Synthetic record ${vin}`;
  body.replaceChildren(el("p", { class: "small muted" }, "Loading…"));
  card.scrollIntoView({ behavior: "smooth", block: "start" });
  let rec;
  try {
    rec = await apiJSON(`/api/sandbox/registry/${encodeURIComponent(vin)}`);
  } catch (e) {
    if (activeVin !== vin) return;
    body.replaceChildren(el("p", { class: "error" }, `Could not load this record: ${e.message}`));
    return;
  }
  if (activeVin !== vin) return;
  const parts = [];
  if (rec.synthetic !== true) {
    parts.push(el("p", { class: "error" }, "Warning: the server did not flag this record as synthetic."));
  }
  parts.push(
    el(
      "dl",
      { class: "kv" },
      SUMMARY.filter((k) => k in rec).map((k) => [el("dt", {}, humanize(k)), el("dd", {}, k === "vin" ? el("code", {}, rec.vin) : valueNode(rec[k]))]),
    ),
  );
  for (const [key, title] of SECTIONS) {
    if (!(key in rec)) continue;
    const v = rec[key];
    const count = Array.isArray(v) ? ` (${v.length})` : "";
    parts.push(el("h3", { style: { marginTop: "1rem" } }, `${title}${count} `, badge()), valueNode(v));
  }
  const known = new Set([...SECTIONS.map((s) => s[0]), ...SUMMARY]);
  const rest = Object.entries(rec).filter(([k]) => !known.has(k));
  if (rest.length) {
    parts.push(el("h3", { style: { marginTop: "1rem" } }, "Other fields ", badge()), valueNode(Object.fromEntries(rest)));
  }
  body.replaceChildren(...parts);
}

function init() {
  $("#filter").addEventListener("input", renderList);
  loadList();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
else init();
