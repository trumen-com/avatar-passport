# web/ — AvtoPassport frontend

Static files served by the FastAPI app (same origin). No build step, no npm.

| File | What it does |
|---|---|
| `index.html` | Single mobile-first page: capture form, then the passport (VIN card, 360° spin, damage list, diff, try-on, 3D model, body shops, verification). |
| `app.js` | All page logic (ES module). Uploads with `XMLHttpRequest` (progress bar), polls the job every 2 s, then polls `recon3d` every 5 s until `done` / `error` / `unavailable`. `REPO_URL` at the top sets the footer "Source code (AGPL-3.0)" link (hidden while empty). |
| `viewer3d.js` | three.js 0.186.1 point-cloud viewer (PLYLoader, OrbitControls, CSS2D labels for damage pins). Imported only when a 3D model is ready. |
| `sandbox.html`, `sandbox.js` | "Registry connector — sandbox": lists the synthetic registry records with a SYNTHETIC DEMO DATA badge on every view. |
| `style.css` | Light/dark theme via `prefers-color-scheme`, large touch targets. |

Third-party code: three.js (MIT), loaded from `cdn.jsdelivr.net/npm/three@0.186.1` via the import map in `index.html`.

## API used (same origin)

- `GET /api/health`
- `POST /api/inspections` (multipart: `video`, optional `vin_photo`, optional `vin_text`) → `202 {id, status_url}`
- `GET /api/inspections/<id>` (job + result; `diff`, `media`, `recon3d`, `report` are read from the job's top level, or from `result` if absent there)
- `GET /api/inspections/<id>/tryon?colour=<hex>&frame=<file name>` → JPEG
- `GET /api/palette`, `GET /api/vin/<vin>/history`, `GET /api/shops?lat=&lon=&radius_m=5000`
- `GET /api/sandbox/registry`, `GET /api/sandbox/registry/<vin>`

Errors are shown as returned (`detail` of a FastAPI error, else the HTTP status). Nothing is invented on the client.

A passport can be reopened or shared with `index.html?inspection=<id>`.

## Run locally

Serve through the API app (it mounts `web/`). Opening the files directly or with `python -m http.server` shows the pages, but every API call fails because there is no backend on that server.
