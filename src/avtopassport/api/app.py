"""AvtoPassport FastAPI app: JSON API under /api, the static mobile page (web/) at /.

Run locally (from the repo root, .venv active):
    uvicorn avtopassport.api.app:app --host 0.0.0.0 --port 8000

Environment:
    AVTO_DATA_DIR        jobs, SQLite and caches (default REPO_ROOT/var, gitignored)
    AVTO_DEVICE          cpu | cuda | cuda:0 ... (default: cuda if torch sees a GPU, else cpu)
    AVTO_RECON3D_DEVICE  device for the 3D job (default: AVTO_DEVICE)
    AVTO_RECON3D=0       disable the 3D job (status "unavailable")
    AVTO_HQ3D=1          optional GPU-only high-quality 3D (COLMAP SfM + Brush Gaussian splats) after the 3D job
    AVTO_HQ3D_STEPS      Brush training steps for it (default 30000)
    AVTO_REUSE_3D=0      always rebuild 3D; by default a finished build of identical blurred frames + settings is
                         attached (labelled "reused", api/reuse3d.py) instead of being rebuilt
    AVTO_FPS / AVTO_MAX_FRAMES  frames kept per inspection (defaults 2.5 / 72); more frames = better HQ 3D
    AVTO_WARMUP=0        do not load the models at startup (they load on the first inspection)
    AVTO_MAX_VIDEO_MB    upload limit (default 200)
    AVTO_COMMIT_SHA      deployed commit, shown by /api/version (AGPL-3.0 section 13 source link)
    AVTO_REPO_URL        repository URL for the source link
    AVTO_SNAPSHOT_REPO + HF_TOKEN  opt-in: restore/snapshot passports to a private HF dataset repo (api/persist.py)
    AVTO_SNAPSHOT_EVERY_MIN        snapshot interval in minutes (default 5)

The API contract the web page codes against is the docstring of each route; every response is JSON except the
media routes (jpg / png / pdf / ply).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..paths import REPO_ROOT
from .jobs import API_PREFIX, Backends, JobManager, Settings, jsonable
from .persist import Persister
from .shops import ATTRIBUTION_URL, SOURCE, ShopFinder, ShopsUnavailable

log = logging.getLogger("avtopassport.api")

SYNTHETIC_BADGE = "SYNTHETIC DEMO DATA"
VIDEO_TYPES: dict[str, str] = {"video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm"}
VIDEO_EXT = frozenset({".mp4", ".mov", ".webm"})
PHOTO_TYPES: dict[str, str] = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
                               "image/heic": ".heic", "image/heif": ".heif"}
PHOTO_EXT = frozenset({".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"})
GENERIC_TYPES = frozenset({"", "application/octet-stream"})
HEX_RE = re.compile(r"^#?([0-9A-Fa-f]{6})$")
VIN_PATH_RE = re.compile(r"^[A-Z0-9]{1,20}$")
CHUNK = 1 << 20


def commit_sha() -> str:
    env = os.environ.get("AVTO_COMMIT_SHA", "").strip()
    if env:
        return env
    try:
        return (REPO_ROOT / "COMMIT_SHA").read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"


def _pick_ext(upload: UploadFile, types: dict[str, str], exts: frozenset[str]) -> str | None:
    """Saved extension if the upload is an accepted type, else None (content type or file extension)."""
    ctype = (upload.content_type or "").split(";")[0].strip().lower()
    ext = Path(upload.filename or "").suffix.lower()
    if ctype in types:
        return ext if ext in exts else types[ctype]
    if ctype in GENERIC_TYPES and ext in exts:  # some phones send .mov / .heic as octet-stream
        return ext
    return None


async def _save_upload(upload: UploadFile, dest: Path, limit: int, what: str) -> int:
    size = 0
    with dest.open("wb") as f:
        while chunk := await upload.read(CHUNK):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"{what} is larger than {limit // (1024 * 1024)} MB")
            f.write(chunk)
    return size


def _findings_of(inspection: dict[str, Any]) -> list[Any]:
    result = inspection.get("result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            result = {}
    return list((result or {}).get("findings") or []) if isinstance(result, dict) else []


def create_app(*, settings: Settings | None = None, backends: Backends | None = None,
               shops: ShopFinder | None = None, web_dir: Path | None = None,
               persister: Persister | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    persist = persister or Persister.from_env(settings.data_dir)
    jobs = JobManager(settings, backends or Backends(), on_stored=persist.notify)
    finder = shops or ShopFinder(settings.data_dir / "cache")
    web = web_dir or REPO_ROOT / "web"
    upload_limit = settings.max_video_bytes + settings.max_photo_bytes + CHUNK

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(persist.restore)  # before jobs.start(): it reloads job states from disk
        jobs.start()
        persist.start()
        yield
        jobs.stop()
        await asyncio.to_thread(persist.stop)     # last snapshot, bounded

    app = FastAPI(title="AvtoPassport", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json",
                  swagger_ui_oauth2_redirect_url="/api/docs/oauth2-redirect")
    app.state.jobs = jobs
    app.state.shops = finder
    app.state.persist = persist

    @app.middleware("http")
    async def reject_huge_uploads(request: Request, call_next: Any) -> Response:
        # refuse before the multipart body is spooled to disk
        if request.method == "POST" and request.url.path == API_PREFIX:
            try:
                length = int(request.headers.get("content-length") or 0)
            except ValueError:
                length = 0
            if length > upload_limit:
                return JSONResponse({"detail": f"upload larger than {settings.max_video_bytes // (1024 * 1024)} MB"},
                                    status_code=413)
        return await call_next(request)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled API error")
        return JSONResponse({"detail": f"internal error: {type(exc).__name__}"}, status_code=500)

    # ------------------------------------------------------------------ service
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """{"ok": true, "device", "models_ready", "recon3d_available", "version", "persistence", ...}"""
        ok3d, msg3d = jobs.recon3d_available_nowait()
        return {"ok": True, "device": jobs.device, "models_ready": jobs.models_ready,
                "models_error": jobs.models_error, "recon3d_available": ok3d, "recon3d_message": msg3d,
                "hq3d_enabled": settings.enable_hq3d,
                "version": __version__, "commit": commit_sha(), "queue": jobs.queue_sizes(),
                "persistence": persist.status()}

    @app.get("/api/version")
    def version() -> dict[str, Any]:
        """Version + deployed commit + source link (AGPL-3.0 section 13)."""
        sha, repo = commit_sha(), os.environ.get("AVTO_REPO_URL", "").rstrip("/")
        return {"version": __version__, "commit": sha, "licence": "AGPL-3.0-only",
                "source_url": f"{repo}/tree/{sha}" if repo and sha != "unknown" else (repo or None)}

    # ------------------------------------------------------------------ inspections
    @app.post("/api/inspections", status_code=202)
    async def create_inspection(
        video: Annotated[UploadFile, File(description="walkaround video: mp4 / mov / webm")],
        vin_photo: Annotated[UploadFile | None, File(description="optional VIN plate photo")] = None,
        vin_text: Annotated[str | None, Form(max_length=40)] = None,
    ) -> dict[str, Any]:
        """Multipart upload -> 202 {"id", "status_url"}. Poll status_url until status is done or error."""
        vext = _pick_ext(video, VIDEO_TYPES, VIDEO_EXT)
        if vext is None:
            raise HTTPException(415, f"unsupported video type {video.content_type!r}; send mp4, mov or webm")
        pext: str | None = None
        if vin_photo is not None and (vin_photo.filename or vin_photo.size):
            pext = _pick_ext(vin_photo, PHOTO_TYPES, PHOTO_EXT)
            if pext is None:
                raise HTTPException(415, f"unsupported VIN photo type {vin_photo.content_type!r}; send jpg/png/webp/heic")
        job_id = jobs.new_job_id()
        updir = jobs.upload_dir(job_id)
        updir.mkdir(parents=True, exist_ok=True)
        try:
            video_path = updir / f"video{vext}"
            if await _save_upload(video, video_path, settings.max_video_bytes, "video") == 0:
                raise HTTPException(400, "the video file is empty")
            photo_path: Path | None = None
            if pext is not None and vin_photo is not None:
                photo_path = updir / f"vin_photo{pext}"
                if await _save_upload(vin_photo, photo_path, settings.max_photo_bytes, "VIN photo") == 0:
                    photo_path.unlink(missing_ok=True)
                    photo_path = None
        except BaseException:
            shutil.rmtree(jobs.job_dir(job_id), ignore_errors=True)
            raise
        text = (vin_text or "").strip() or None
        jobs.submit(job_id, video_path, vin_photo=photo_path, vin_text=text)
        return {"id": job_id, "status_url": f"{API_PREFIX}/{job_id}"}

    def _job_or_404(job_id: str) -> dict[str, Any]:
        st = jobs.get(job_id)
        if st is None:
            raise HTTPException(404, "inspection job not found")
        return st

    @app.get("/api/inspections/{job_id}")
    def get_inspection(job_id: str) -> dict[str, Any]:
        """{"id","status","progress","error","inspection_id","result","report","diff","media","recon3d",...}"""
        return _job_or_404(job_id)

    @app.get("/api/inspections/{job_id}/frames/{name}")
    def get_frame(job_id: str, name: str) -> FileResponse:
        _job_or_404(job_id)
        path = jobs.resolve_frame(job_id, name)
        if path is None:
            raise HTTPException(404, "frame not found")
        media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return FileResponse(path, media_type=media, headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/inspections/{job_id}/masks/{name:path}")
    def get_mask(job_id: str, name: str) -> FileResponse:
        _job_or_404(job_id)
        path = jobs.resolve_mask(job_id, name)
        if path is None:
            raise HTTPException(404, "mask not found")
        media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return FileResponse(path, media_type=media, headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/inspections/{job_id}/report.pdf")
    def get_report_pdf(job_id: str) -> FileResponse:
        st = _job_or_404(job_id)
        path = jobs.resolve_pdf(job_id)
        if path is None:
            raise HTTPException(404, "PDF not available" if st["status"] == "done" else "inspection not finished")
        name = f"avtopassport_{st.get('inspection_id') or job_id[:8]}.pdf"
        return FileResponse(path, media_type="application/pdf",
                            headers={"Content-Disposition": f'inline; filename="{name}"'})

    @app.get("/api/inspections/{job_id}/recon3d/points.ply")
    def get_points_ply(job_id: str) -> FileResponse:
        st = _job_or_404(job_id)
        path = jobs.resolve_ply(job_id)
        if path is None:
            raise HTTPException(404, f"3D model not ready (status: {st['recon3d'].get('status')})")
        return FileResponse(path, media_type="application/octet-stream",
                            headers={"Cache-Control": "private, max-age=86400"})

    def _splat(job_id: str, full: bool) -> FileResponse:
        st = _job_or_404(job_id)
        path = jobs.resolve_splat(job_id, full=full)
        if path is None:
            status = (st.get("recon3d_hq") or {}).get("status")
            what = "uncropped high-quality 3D" if full else "high-quality 3D"
            raise HTTPException(404, f"{what} not available (status: {status})")
        return FileResponse(path, media_type="application/octet-stream",
                            headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/inspections/{job_id}/recon3d/splat.ply")
    def get_splat_ply(job_id: str) -> FileResponse:
        """High-quality Gaussian splats (standard 3DGS .ply, cropped to the car box when the crop succeeded) when
        recon3d_hq.status is done; else 404."""
        return _splat(job_id, full=False)

    @app.get("/api/inspections/{job_id}/recon3d/splat_full.ply")
    def get_splat_full_ply(job_id: str) -> FileResponse:
        """The uncropped splats (car + background) when a car crop is served as splat.ply; else 404."""
        return _splat(job_id, full=True)

    def _body_mask(job_id: str, frame: str, extra: dict[str, Any]) -> Path | None:
        explicit = extra.get("body_masks")
        if isinstance(explicit, dict) and isinstance(explicit.get(frame), str):
            p = jobs.resolve_mask(job_id, Path(explicit[frame]).name)
            if p is not None:
                return p
        mdir = jobs.masks_dir(job_id)
        if not mdir.is_dir():
            return None
        stem = Path(frame).stem
        for cand in (f"{stem}_body.png", f"body_{stem}.png", f"{stem}.body.png", f"body/{stem}.png",
                     f"{stem}/body.png", f"{stem}_body_mask.png", f"{stem}.png"):  # pipeline: masks/<frame stem>.png
            p = jobs.resolve_mask(job_id, cand)
            if p is not None:
                return p
        for p in sorted(mdir.rglob("*.png")):
            rel = p.relative_to(mdir).as_posix()
            if stem in rel and "body" in rel.lower():
                return p
        return None

    @app.get("/api/inspections/{job_id}/tryon")
    def tryon(job_id: str, colour: Annotated[str, Query(description="hex colour without '#', e.g. 1F3A93")],
              frame: Annotated[str | None, Query(description="frame name; default the first spin frame")] = None,
              ) -> Response:
        """Paint try-on: a blurred frame recoloured inside its body-panel mask -> image/jpeg."""
        m = HEX_RE.match(colour.strip())
        if m is None:
            raise HTTPException(400, "colour must be 6 hex digits, e.g. 1F3A93")
        hex_colour = m.group(1).upper()
        st = jobs.internal(job_id)
        if st is None:
            raise HTTPException(404, "inspection job not found")
        if st.get("status") != "done":
            raise HTTPException(409, "inspection not finished")
        extra = (st.get("result") or {}).get("extra") or {}
        if frame is not None:
            names = [frame]
        else:
            names = [u.rsplit("/", 1)[-1] for u in st["media"].get("spin", []) + st["media"].get("frames", [])]
        chosen: tuple[Path, Path] | None = None
        for name in names:
            fpath = jobs.resolve_frame(job_id, name)
            if fpath is None:
                continue
            mpath = _body_mask(job_id, fpath.name, extra)
            if mpath is not None:
                chosen = (fpath, mpath)
                break
        if chosen is None:
            raise HTTPException(404, "frame or its body mask not found")
        fpath, mpath = chosen
        cache = jobs.job_dir(job_id) / "tryon" / f"{fpath.stem}_{hex_colour}.jpg"
        if cache.is_file():
            return Response(cache.read_bytes(), media_type="image/jpeg")

        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        mask_img = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if img is None or mask_img is None:
            raise HTTPException(500, "could not read the frame or its mask")
        if mask_img.shape[:2] != img.shape[:2]:
            mask_img = cv2.resize(mask_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        try:
            out = jobs.backends.recolor(img, mask_img > 127, f"#{hex_colour}")
        except Exception as exc:
            log.exception("recolor failed")
            raise HTTPException(503, f"try-on unavailable: {type(exc).__name__}: {exc}") from exc
        ok, buf = cv2.imencode(".jpg", np.asarray(out), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise HTTPException(500, "JPEG encoding failed")
        data = buf.tobytes()
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
        except OSError:
            pass
        return Response(data, media_type="image/jpeg")

    @app.get("/api/palette")
    def palette() -> Any:
        """tryon.recolor.load_palette(): paint colours with their source."""
        try:
            return jsonable(jobs.backends.load_palette())
        except Exception as exc:
            raise HTTPException(503, f"palette unavailable: {type(exc).__name__}: {exc}") from exc

    # ------------------------------------------------------------------ passports by VIN
    @app.get("/api/vin/{vin}/history")
    def vin_history(vin: str) -> dict[str, Any]:
        """Real AvtoPassport inspections of this VIN, newest first, + diff of the latest vs the one before."""
        v = vin.strip().upper()
        if not VIN_PATH_RE.match(v):
            raise HTTPException(400, "VIN must be letters and digits")
        try:
            with jobs.store() as store:
                hist = list(store.history(v) or [])
        except Exception as exc:
            raise HTTPException(503, f"passport store unavailable: {type(exc).__name__}: {exc}") from exc
        items = []
        for h in hist:
            media = h.get("media") if isinstance(h.get("media"), dict) else {}
            job_id = media.get("job_id")
            items.append({"id": h.get("id"), "created_utc": h.get("created_utc"),
                          "n_findings": len(_findings_of(h)), "frames_sha256": h.get("frames_sha256"),
                          "job_id": job_id, "status_url": f"{API_PREFIX}/{job_id}" if job_id else None})
        diff_latest: dict[str, Any] | None = None
        if len(hist) >= 2:
            try:
                diff_latest = jobs.compute_diff(_findings_of(hist[1]), _findings_of(hist[0]))
                diff_latest.setdefault("previous_inspection_id", hist[1].get("id"))
                diff_latest.setdefault("inspection_id", hist[0].get("id"))
            except Exception as exc:
                raise HTTPException(503, f"diff unavailable: {type(exc).__name__}: {exc}") from exc
        return {"vin": v, "inspections": jsonable(items), "diff_latest": diff_latest}

    # ------------------------------------------------------------------ synthetic sandbox registry
    @app.get("/api/sandbox/registry")
    def sandbox_registry() -> list[dict[str, Any]]:
        """Synthetic sandbox VINs ("registry connector - sandbox"). Every item: synthetic=true + badge."""
        from ..synthetic import SOURCE_LABEL, load_registry  # noqa: PLC0415

        return [{"vin": vin, "make": r.get("make"), "model": r.get("model"), "model_year": r.get("model_year"),
                 "colour": r.get("colour"), "source": r.get("source", SOURCE_LABEL),
                 "synthetic": True, "badge": SYNTHETIC_BADGE}
                for vin, r in sorted(load_registry().items())]

    @app.get("/api/sandbox/registry/{vin}")
    def sandbox_record(vin: str) -> dict[str, Any]:
        """One synthetic record, or 404. Only sandbox VINs; a scanned VIN is never looked up here."""
        from ..synthetic import RegistryRefusal, lookup  # noqa: PLC0415

        rec = lookup(vin, origin="sandbox")
        if rec is None or isinstance(rec, RegistryRefusal):
            raise HTTPException(404, "not a sandbox VIN")
        return {**jsonable(rec), "synthetic": True, "badge": SYNTHETIC_BADGE}

    # ------------------------------------------------------------------ OpenStreetMap shops
    @app.get("/api/shops")
    def nearby_shops(lat: Annotated[float, Query(ge=-90, le=90)], lon: Annotated[float, Query(ge=-180, le=180)],
                     radius_m: Annotated[int, Query(ge=100, le=25_000)] = 5000) -> dict[str, Any]:
        """Body/paint shops near (lat, lon) from OpenStreetMap; 503 when Overpass is unreachable (never faked)."""
        try:
            return finder.find(lat, lon, radius_m)
        except ShopsUnavailable as exc:
            raise HTTPException(503, f"{exc} (data: {SOURCE}, {ATTRIBUTION_URL})") from exc

    # ------------------------------------------------------------------ static web page (after the API routes)
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    else:
        @app.get("/", include_in_schema=False)
        def no_web() -> JSONResponse:
            return JSONResponse({"detail": f"web/ not found at {web}; the API is under /api"}, status_code=404)

    return app


app = create_app()
