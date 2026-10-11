"""API tests: FastAPI TestClient, every heavy piece stubbed (no models, no network, no torch, no 3D env).

The stubs only stand in for the modules written elsewhere (pipeline.run, passport.*, recon3d.*, tryon.recolor) so the
API wiring - uploads, job lifecycle, persistence, media routes, diff on a repeat VIN, sandbox badges, shops - is tested
on its own. Overpass is mocked; nothing here touches the network.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import threading
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from avtopassport.api import jobs as jobs_mod
from avtopassport.api.app import create_app
from avtopassport.api.jobs import Backends, JobManager, Settings, inspection_overrides, read_view
from avtopassport.api.shops import OVERPASS_ENDPOINTS, ShopFinder
from avtopassport.recon3d.build import DEFAULT_MODEL as DA3_MODEL
from avtopassport.recon3d.worker import DEFAULT_PROCESS_RES, MODEL_REVISIONS
from avtopassport.schema import DamageFinding, InspectionResult

VIN = "JTMHV05J604123456"  # made-up string for the stub pipeline; never sent to the sandbox
MP4 = ("walk.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64, "video/mp4")


# --------------------------------------------------------------------------------------------- stubs

def _finding(part: str, damage: str, score: float = 0.8) -> DamageFinding:
    return DamageFinding(part=part, damage=damage, n_observations=2, sources=["f0000", "f0001"],
                         best_source="f0000", max_score=score, max_area_ratio=0.05, severity="minor",
                         action="polish", severity_rule="stub")


@dataclass
class StubInspector:
    """Writes blank 'blurred' frames + body masks, returns a fake InspectionResult (more damage on each repeat)."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    fail: bool = False
    frames_sha256: str = "ab" * 32  # pipeline: SHA-256 over the saved blurred frames (the 3D input)
    fps: float = 2.5

    def run(self, video_path: Path, *, out_dir: Path, vin_image_path: Path | None = None,
            vin_text: str | None = None, progress: Any = None) -> InspectionResult:
        self.calls.append({"video_exists": Path(video_path).is_file(),
                           "vin_photo_exists": bool(vin_image_path and Path(vin_image_path).is_file()),
                           "vin_text": vin_text})
        if self.fail:
            raise RuntimeError("stub pipeline exploded")
        frames, masks = Path(out_dir) / "frames", Path(out_dir) / "masks"
        frames.mkdir(parents=True, exist_ok=True)
        masks.mkdir(parents=True, exist_ok=True)
        names = []
        for i in range(6):  # same layout as pipeline.run: frames/f_<i:05d>.jpg + masks/f_<i:05d>.png
            name = f"f_{i:05d}.jpg"
            cv2.imwrite(str(frames / name), np.full((48, 64, 3), 40 * i, np.uint8))
            m = np.zeros((48, 64), np.uint8)
            m[10:40, 8:56] = 255
            cv2.imwrite(str(masks / f"f_{i:05d}.png"), m)
            names.append(f"frames/{name}")
            if progress:
                progress("frames", (i + 1) / 6)
        if progress:
            progress("damage", 1.0)
        n_same_vin = sum(1 for c in self.calls if c["vin_text"] == vin_text)
        findings = [_finding("front_bumper", "scratch")]
        if n_same_vin >= 2:
            findings.append(_finding("left_front_door", "dent", 0.9))
        return InspectionResult(
            vin=vin_text, vin_status="manual" if vin_text else "could_not_read", created_utc="2026-10-11T01:00:00Z",
            frames_sha256=self.frames_sha256, findings=findings, timings_ms={"total": 12.5},
            extra={"frames": names, "spin": [0, 2, 4], "masks": [f"masks/f_{i:05d}.png" for i in range(6)],
                   "pins2d": [{"frame": names[0], "box": [10.0, 10.0, 30.0, 30.0], "damage": "scratch",
                               "part": "front_bumper", "score": 0.8}],
                   "frame_size": [64, 48], "vin_decode": None, "quality": {"kept": 6},
                   "config": {"device": "cpu", "fps": self.fps, "max_frames": 72, "work_side": 1280},
                   "np_value": np.float32(0.5), "big_mask": np.zeros((100, 100), bool)})


class StubStore:
    """In-memory stand-in for passport.store.PassportStore (shared across open_store calls)."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def add_inspection(self, result: InspectionResult, media: dict[str, Any]) -> int:
        with self.lock:
            iid = len(self.rows) + 1
            self.rows.append({"id": iid, "vin": result.vin, "created_utc": f"2026-10-11T01:00:{iid:02d}Z",
                              "frames_sha256": result.frames_sha256,
                              "result": jobs_mod.jsonable(result.to_dict()), "media": media})
            return iid

    def get_inspection(self, inspection_id: int) -> dict[str, Any] | None:
        return next((r for r in self.rows if r["id"] == inspection_id), None)

    def history(self, vin: str) -> list[dict[str, Any]]:
        return sorted((r for r in self.rows if r["vin"] == vin), key=lambda r: -r["id"])

    def previous(self, vin: str, inspection_id: int) -> dict[str, Any] | None:
        older = [r for r in self.history(vin) if r["id"] < inspection_id]
        return older[0] if older else None


def stub_diff(previous: list[dict[str, Any]] | None, current: list[dict[str, Any]]) -> dict[str, Any]:
    key = lambda f: (f["part"], f["damage"])  # noqa: E731
    prev = {key(f): f for f in previous or []}
    cur = {key(f): f for f in current}
    out: dict[str, Any] = {"new": [cur[k] for k in cur if k not in prev],
                           "resolved": [prev[k] for k in prev if k not in cur],
                           "unchanged": [cur[k] for k in cur if k in prev], "worse": []}
    if previous is None:
        out["first_inspection"] = True
    return out


@dataclass
class World:
    inspector: StubInspector = field(default_factory=StubInspector)
    store: StubStore = field(default_factory=StubStore)
    recon_ok: bool = False
    recon_build_ok: bool = True
    pdf_images: list[int] = field(default_factory=list)
    overpass_calls: list[str] = field(default_factory=list)
    overpass_fail: set[str] = field(default_factory=set)
    recon_builds: int = 0

    def backends(self) -> Backends:
        def build_json(result: Any, *, diff: Any, decode: Any, extra: dict[str, Any]) -> dict[str, Any]:
            return {"vin": result.vin, "n_findings": len(result.findings), "diff": diff,
                    "inspection_id": extra["inspection_id"]}

        def build_pdf(report: dict[str, Any], out_path: Path, images: list[Path]) -> Path:
            assert all(Path(p).is_file() for p in images)
            self.pdf_images.append(len(images))
            Path(out_path).write_bytes(b"%PDF-1.4\n% stub\n%%EOF\n")
            return Path(out_path)

        def recon3d_build(frames_dir: Path, out_dir: Path, device: str, masks_dir: Path | None) -> dict[str, Any]:
            assert len(list(Path(frames_dir).glob("*.jpg"))) == 6 and masks_dir is not None and masks_dir.is_dir()
            if not self.recon_build_ok:
                return {"ok": False, "result": None, "error": "stub DA3 crashed", "seconds_wall": 0.1,
                        "out_dir": str(out_dir)}
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "points.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\nFULL")
            (Path(out_dir) / "points_web.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\nWEB")
            (Path(out_dir) / "cameras.json").write_text("{}")
            (Path(out_dir) / "depth.npz").write_bytes(b"stub depth")
            self.recon_builds += 1
            info = {"n_points": 0, "device": device, "model": DA3_MODEL, "revision": MODEL_REVISIONS[DA3_MODEL],
                    "process_res_requested": DEFAULT_PROCESS_RES["cpu"]}  # what the real worker reports
            return {"ok": True, "result": info, "error": None, "seconds_wall": 1.5,
                    "out_dir": str(out_dir), "ply": str(Path(out_dir) / "points.ply"),
                    "ply_web": str(Path(out_dir) / "points_web.ply"), "preview": None}

        def project_pins(out_dir: Path, pins2d: Any, frame_size: Any) -> list[dict[str, Any]]:
            assert frame_size == [64, 48] and len(pins2d) == 1
            return [{"xyz": [0.0, 0.1, 0.2], "damage": "scratch", "part": "front_bumper", "score": 0.8,
                     "n_views": 1, "frames": ["f_00000.jpg"]}]

        def recolor(img: np.ndarray, mask: np.ndarray, hex_colour: str) -> np.ndarray:
            h = hex_colour.lstrip("#")
            out = img.copy()
            out[mask] = (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))  # BGR
            return out

        return Backends(
            make_inspector=lambda device: self.inspector,
            open_store=lambda path: self.store,
            diff=stub_diff,
            build_json=build_json,
            build_pdf=build_pdf,
            recon3d_available=lambda: (True, "stub ok") if self.recon_ok else (False, "DA3 environment not installed"),
            recon3d_build=recon3d_build,
            project_pins=project_pins,
            load_palette=lambda: {"colours": [{"name": "White", "hex": "#FFFFFF"}], "source_note": "stub"},
            recolor=recolor,
        )

    def fetch(self, url: str, query: str, timeout_s: float) -> dict[str, Any]:
        self.overpass_calls.append(url)
        assert 'nwr["shop"="car_repair"](around:' in query and timeout_s == 25.0
        if url in self.overpass_fail:
            raise OSError("stub network down")
        return {"elements": [
            {"type": "node", "id": 1, "lat": 41.3115, "lon": 69.2800, "tags": {"shop": "car_repair", "name": "Avto Servis"}},
            {"type": "way", "id": 2, "center": {"lat": 41.3200, "lon": 69.2900},
             "tags": {"shop": "car_repair", "name": "Kuzov Usta"}},
            {"type": "node", "id": 3, "lat": 41.3300, "lon": 69.3000,
             "tags": {"shop": "car_repair", "service:vehicle:body_repair": "yes"}},
            {"type": "node", "id": 4, "lat": 45.0, "lon": 69.0, "tags": {"shop": "car_repair", "name": "far away"}},
            {"type": "relation", "id": 5, "tags": {"shop": "car_repair"}},  # no position -> skipped
        ]}


def make_client(tmp_path: Path, world: World, **settings: Any) -> TestClient:
    s = Settings(data_dir=tmp_path / "data", device="cpu", warmup=False, **settings)
    app = create_app(settings=s, backends=world.backends(),
                     shops=ShopFinder(s.data_dir / "cache", fetch=world.fetch), web_dir=tmp_path / "web")
    return TestClient(app)


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def client(tmp_path: Path, world: World) -> Iterator[TestClient]:
    with make_client(tmp_path, world) as c:
        yield c


def _jobs(c: TestClient) -> JobManager:
    return c.app.state.jobs  # type: ignore[attr-defined]


def run_inspection(c: TestClient, vin_text: str | None = VIN, **files: Any) -> dict[str, Any]:
    data = {"vin_text": vin_text} if vin_text else {}
    r = c.post("/api/inspections", files={"video": MP4, **files}, data=data)
    assert r.status_code == 202, r.text
    body = r.json()
    assert set(body) == {"id", "status_url"} and body["status_url"] == f"/api/inspections/{body['id']}"
    assert _jobs(c).wait_idle(15)
    st = c.get(body["status_url"]).json()
    assert st["status"] in {"done", "error"}
    return st


# --------------------------------------------------------------------------------------------- tests

def test_health(client: TestClient) -> None:
    _jobs(client).recon3d_available()  # the startup check runs in a thread; make it deterministic
    r = client.get("/api/health")
    assert r.status_code == 200
    h = r.json()
    assert h["ok"] is True and h["device"] == "cpu" and h["models_ready"] is False
    assert h["recon3d_available"] is False and "DA3" in h["recon3d_message"]
    assert isinstance(h["version"], str)


def test_upload_validation_type_and_empty(client: TestClient, tmp_path: Path) -> None:
    r = client.post("/api/inspections", files={"video": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = client.post("/api/inspections", files={"video": ("x.jpg", b"\xff\xd8", "image/jpeg")})
    assert r.status_code == 415
    r = client.post("/api/inspections", files={"video": ("clip.mov", b"\x00" * 10, "video/mp4"),
                                               "vin_photo": ("vin.txt", b"abc", "text/plain")})
    assert r.status_code == 415
    r = client.post("/api/inspections", files={"video": ("empty.mp4", b"", "video/mp4")})
    assert r.status_code == 400
    r = client.post("/api/inspections")
    assert r.status_code == 422
    assert not list((tmp_path / "data" / "jobs").glob("*/upload"))  # nothing left behind


def test_upload_accepts_octet_stream_mov_and_webm(client: TestClient, world: World) -> None:
    for name, ctype in (("IMG_0001.MOV", "application/octet-stream"), ("rec", "video/webm"),
                        ("clip.mov", "video/quicktime")):
        r = client.post("/api/inspections", files={"video": (name, b"\x00" * 32, ctype)})
        assert r.status_code == 202, (name, r.text)
    assert _jobs(client).wait_idle(15)
    assert len(world.inspector.calls) == 3


def test_upload_size_limits(tmp_path: Path, world: World) -> None:
    with make_client(tmp_path, world, max_video_bytes=2000, max_photo_bytes=1000) as c:
        r = c.post("/api/inspections", files={"video": ("big.mp4", b"\x00" * 5000, "video/mp4")})
        assert r.status_code == 413 and "MB" in r.json()["detail"]
        # bigger than the whole-request limit: refused from Content-Length before parsing
        r = c.post("/api/inspections", files={"video": ("huge.mp4", b"\x00" * (1_200_000), "video/mp4")})
        assert r.status_code == 413
    assert not any((tmp_path / "data" / "jobs").glob("*"))
    assert world.inspector.calls == []


def test_job_lifecycle_media_and_privacy(client: TestClient, world: World, tmp_path: Path) -> None:
    vin_photo = ("vin.jpg", cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes(), "image/jpeg")
    st = run_inspection(client, vin_photo=vin_photo)
    assert st["status"] == "done", st
    assert st["error"] is None and st["inspection_id"] == 1
    assert world.inspector.calls[0] == {"video_exists": True, "vin_photo_exists": True, "vin_text": VIN}
    steps = [p["step"] for p in st["progress"]]
    assert steps[0] == "load_models" and "frames" in steps and len(steps) == len(set(steps))
    assert all(0.0 <= p["fraction"] <= 1.0 for p in st["progress"])
    assert st["result"]["vin"] == VIN and len(st["result"]["findings"]) == 1
    assert st["result"]["extra"]["np_value"] == 0.5 and st["result"]["extra"]["big_mask"]["omitted"] == "array"
    assert st["report"]["n_findings"] == 1 and st["report"]["inspection_id"] == 1
    assert st["diff"]["first_inspection"] is True and len(st["diff"]["new"]) == 1
    media = st["media"]
    base = f"/api/inspections/{st['id']}"
    assert media["frames"] == [f"{base}/frames/f_{i:05d}.jpg" for i in range(6)]
    assert media["spin"] == [f"{base}/frames/f_{i:05d}.jpg" for i in (0, 2, 4)]
    assert media["pdf"] == f"{base}/report.pdf" and len(media["masks"]) == 6
    assert world.pdf_images == [3]  # first (up to 4) spin frames
    assert st["recon3d"] == {"status": "unavailable", "ply": None, "pins": [], "seconds": None,
                             "error": "DA3 environment not installed"}

    # privacy: the raw upload is gone, the blurred frames and the state stay
    job_dir = tmp_path / "data" / "jobs" / st["id"]
    assert not (job_dir / "upload").exists()
    assert json.loads((job_dir / "state.json").read_text())["status"] == "done"

    r = client.get(media["frames"][0])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR).shape == (48, 64, 3)
    r = client.get(media["masks"][0])
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    r = client.get(media["pdf"])
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF")
    assert client.get(f"{base}/frames/nope.jpg").status_code == 404
    assert client.get(f"{base}/frames/..%2Fstate.json").status_code == 404
    assert client.get(f"{base}/masks/../state.json").status_code == 404
    assert client.get(f"{base}/recon3d/points.ply").status_code == 404
    assert client.get("/api/inspections/" + "0" * 32).status_code == 404
    assert client.get("/api/inspections/not-a-job").status_code == 404


def test_second_inspection_same_vin_diff_and_history(client: TestClient) -> None:
    first = run_inspection(client)
    second = run_inspection(client)
    assert second["status"] == "done" and second["inspection_id"] == 2
    d = second["diff"]
    assert "first_inspection" not in d and d["previous_inspection_id"] == 1
    assert [(f["part"], f["damage"]) for f in d["new"]] == [("left_front_door", "dent")]
    assert [(f["part"], f["damage"]) for f in d["unchanged"]] == [("front_bumper", "scratch")]

    h = client.get(f"/api/vin/{VIN.lower()}/history").json()
    assert h["vin"] == VIN
    assert [(i["id"], i["n_findings"]) for i in h["inspections"]] == [(2, 2), (1, 1)]
    assert h["inspections"][1]["job_id"] == first["id"] and h["inspections"][0]["frames_sha256"] == "ab" * 32
    assert [(f["part"], f["damage"]) for f in h["diff_latest"]["new"]] == [("left_front_door", "dent")]

    other = client.get("/api/vin/WVWZZZ1JZXW000001/history").json()
    assert other["inspections"] == [] and other["diff_latest"] is None
    assert client.get("/api/vin/bad%20vin!/history").status_code == 400


def test_no_vin_means_no_diff(client: TestClient) -> None:
    st = run_inspection(client, vin_text=None)
    assert st["status"] == "done" and st["result"]["vin"] is None and st["diff"] is None


def test_job_pipeline_never_touches_sandbox(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import avtopassport.synthetic as synthetic

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("the job pipeline must not call the sandbox registry")

    monkeypatch.setattr(synthetic, "lookup", boom)
    monkeypatch.setattr(synthetic, "load_registry", boom)
    st = run_inspection(client, vin_text="0AP4JPMXXGF392722")  # even a sandbox-looking VIN typed by a user
    assert st["status"] == "done" and st["warnings"] == []


def test_inspection_error_is_captured(client: TestClient, world: World, tmp_path: Path) -> None:
    world.inspector.fail = True
    st = run_inspection(client)
    assert st["status"] == "error" and "stub pipeline exploded" in st["error"]
    assert not (tmp_path / "data" / "jobs" / st["id"] / "upload").exists()
    assert st["recon3d"]["status"] == "not_started"
    world.inspector.fail = False
    assert run_inspection(client)["status"] == "done"  # the worker survived
    assert client.get("/api/health").json()["models_ready"] is True


def test_state_survives_restart(tmp_path: Path, world: World) -> None:
    with make_client(tmp_path, world) as c:
        done = run_inspection(c)
        frame_url = done["media"]["frames"][1]
    # a job that was mid-flight when the server died
    stuck_id = "f" * 32
    stuck_dir = tmp_path / "data" / "jobs" / stuck_id
    (stuck_dir / "upload").mkdir(parents=True)
    (stuck_dir / "upload" / "video.mp4").write_bytes(b"raw")
    (stuck_dir / "state.json").write_text(json.dumps({**jobs_mod._new_state(stuck_id), "status": "running"}))

    with make_client(tmp_path, World()) as c2:
        st = c2.get(done["status_url"]).json()
        assert st["status"] == "done" and st["result"]["vin"] == VIN
        assert c2.get(frame_url).status_code == 200
        assert c2.get(done["media"]["pdf"]).status_code == 200
        stuck = c2.get(f"/api/inspections/{stuck_id}").json()
        assert stuck["status"] == "error" and "restart" in stuck["error"]
        assert not (stuck_dir / "upload").exists()


def test_tryon_returns_recoloured_jpeg(client: TestClient) -> None:
    st = run_inspection(client)
    base = f"/api/inspections/{st['id']}"
    r = client.get(f"{base}/tryon", params={"colour": "FF0000", "frame": "f_00002.jpg"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    b, g, red = (int(v) for v in img[25, 30])
    assert red > 200 and g < 60 and b < 60  # inside the body mask
    assert tuple(int(v) for v in img[2, 2]) == pytest.approx((80, 80, 80), abs=8)  # outside: untouched frame
    assert client.get(f"{base}/tryon", params={"colour": "FF0000", "frame": "f_00002.jpg"}).content == r.content
    assert client.get(f"{base}/tryon", params={"colour": "1f3a93"}).status_code == 200  # default frame
    assert client.get(f"{base}/tryon", params={"colour": "red"}).status_code == 400
    assert client.get(f"{base}/tryon", params={"colour": "FF0000", "frame": "nope.jpg"}).status_code == 404
    assert client.get("/api/inspections/" + "a" * 32 + "/tryon", params={"colour": "FF0000"}).status_code == 404


def test_demo_inspection_uses_server_files_and_keeps_them(tmp_path: Path, world: World) -> None:
    demo = tmp_path / "demo"
    demo.mkdir()
    video, photo = demo / "walk.mp4", demo / "vin.jpg"
    video.write_bytes(MP4[1])
    photo.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
    with make_client(tmp_path, world, demo_video=video, demo_vin=photo) as c:
        info = c.get("/api/health").json()["demo"]
        assert info == {"video": "walk.mp4", "video_bytes": video.stat().st_size,
                        "vin_photo": "vin.jpg", "vin_photo_bytes": photo.stat().st_size}
        r = c.post("/api/inspections/demo", data={"vin_text": VIN})
        assert r.status_code == 202, r.text
        assert _jobs(c).wait_idle(15)
        assert c.get(r.json()["status_url"]).json()["status"] == "done"
    assert video.read_bytes() == MP4[1] and photo.is_file()  # the worker deleted only its own linked copies


def test_demo_inspection_absent_without_files(client: TestClient, tmp_path: Path) -> None:
    assert client.get("/api/health").json()["demo"] is None
    assert client.post("/api/inspections/demo").status_code == 404


def test_demo_paths_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from avtopassport.api.jobs import DEFAULT_DEMO_VIDEO
    monkeypatch.delenv("AVTO_DEMO_VIDEO", raising=False)
    assert Settings.from_env(tmp_path).demo_video == DEFAULT_DEMO_VIDEO
    monkeypatch.setenv("AVTO_DEMO_VIDEO", "")
    assert Settings.from_env(tmp_path).demo_video is None  # empty turns the demo button off
    monkeypatch.setenv("AVTO_DEMO_VIDEO", str(tmp_path / "x.mp4"))
    assert Settings.from_env(tmp_path).demo_video == tmp_path / "x.mp4"


def test_palette(client: TestClient) -> None:
    r = client.get("/api/palette")
    assert r.status_code == 200 and r.json()["colours"][0]["hex"] == "#FFFFFF"


def test_tryon_colours_drops_only_marked_swatches() -> None:
    from avtopassport.api.app import _tryon_colours
    pal = [{"name": "A", "hex": "#FFFFFF"}, {"name": "B", "hex": "#000000", "tryon": False},
           {"name": "C", "hex": "#2A3678", "tryon": True}]
    assert [c["name"] for c in _tryon_colours(pal)] == ["A", "C"]
    assert [c["name"] for c in _tryon_colours({"colours": pal, "x": 1})["colours"]] == ["A", "C"]


def test_sandbox_registry_carries_synthetic_badge(client: TestClient) -> None:
    items = client.get("/api/sandbox/registry").json()
    assert isinstance(items, list) and items
    for it in items:
        assert it["synthetic"] is True and it["badge"] == "SYNTHETIC DEMO DATA"
        assert it["vin"].startswith("0") and it["make"] and it["model"]
        assert it["source"] == "registry connector - sandbox"
    rec = client.get(f"/api/sandbox/registry/{items[0]['vin']}").json()
    assert rec["synthetic"] is True and rec["badge"] == "SYNTHETIC DEMO DATA" and rec["vin"] == items[0]["vin"]
    assert client.get(f"/api/sandbox/registry/{VIN}").status_code == 404


def test_recon3d_done_serves_ply_and_pins(tmp_path: Path, world: World) -> None:
    world.recon_ok = True
    with make_client(tmp_path, world) as c:
        _jobs(c).recon3d_available()
        assert c.get("/api/health").json()["recon3d_available"] is True
        st = run_inspection(c)
        rec = st["recon3d"]
        assert rec["status"] == "done" and rec["error"] is None and rec["seconds"] == 1.5
        assert rec["ply"] == f"/api/inspections/{st['id']}/recon3d/points.ply"
        assert rec["pins"][0]["part"] == "front_bumper" and "_ply_path" not in rec
        r = c.get(rec["ply"])
        assert r.status_code == 200 and r.content.startswith(b"ply") and r.content.endswith(b"WEB")


def test_recon3d_build_failure_is_reported(tmp_path: Path, world: World) -> None:
    world.recon_ok, world.recon_build_ok = True, False
    with make_client(tmp_path, world) as c:
        st = run_inspection(c)
        assert st["status"] == "done"
        assert st["recon3d"]["status"] == "error" and st["recon3d"]["error"] == "stub DA3 crashed"
        assert c.get(f"/api/inspections/{st['id']}/recon3d/points.ply").status_code == 404


def test_shops_ranked_cached_and_attributed(client: TestClient, world: World, tmp_path: Path) -> None:
    r = client.get("/api/shops", params={"lat": 41.3111, "lon": 69.2797, "radius_m": 5000})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "OpenStreetMap contributors (ODbL)"
    assert body["attribution_url"] == "https://www.openstreetmap.org/copyright"
    assert body["fetched_utc"].endswith("Z") and body["cached"] is False
    assert [s["match"] for s in body["shops"]] == ["body_repair_tag", "name_keyword", "car_repair"]
    assert [s["name"] for s in body["shops"]] == [None, "Kuzov Usta", "Avto Servis"]
    assert all(set(s) >= {"name", "lat", "lon", "tags", "distance_m", "match"} for s in body["shops"])
    assert body["shops"][2]["distance_m"] < 100
    assert world.overpass_calls == [OVERPASS_ENDPOINTS[0]]
    assert list((tmp_path / "data" / "cache").glob("shops_41.311_69.280_5000.json"))

    again = client.get("/api/shops", params={"lat": 41.3112, "lon": 69.2798}).json()
    assert again["cached"] is True and len(world.overpass_calls) == 1 and len(again["shops"]) == 3


def test_shops_fallback_and_503(client: TestClient, world: World) -> None:
    world.overpass_fail = {OVERPASS_ENDPOINTS[0]}
    r = client.get("/api/shops", params={"lat": 40.0, "lon": 70.0})
    assert r.status_code == 200 and r.json()["endpoint"] == OVERPASS_ENDPOINTS[1]
    world.overpass_fail = set(OVERPASS_ENDPOINTS)
    r = client.get("/api/shops", params={"lat": 39.0, "lon": 66.0})
    assert r.status_code == 503 and "unreachable" in r.json()["detail"]
    assert client.get("/api/shops", params={"lat": 91, "lon": 0}).status_code == 422


def test_static_web_mounted_after_api(tmp_path: Path, world: World) -> None:
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "index.html").write_text("<!doctype html><title>AvtoPassport</title>")
    with make_client(tmp_path, world) as c:
        r = c.get("/")
        assert r.status_code == 200 and "AvtoPassport" in r.text
        assert c.get("/api/health").json()["ok"] is True


def test_default_backends_import_the_agreed_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The lazy loaders import exactly the module paths / names of the agreed contract."""
    seen: dict[str, Any] = {}

    def fake(name: str, **attrs: Any) -> None:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)

    @dataclass
    class InspectionConfig:
        device: str = "cpu"

    class Inspector:
        def __init__(self, config: InspectionConfig) -> None:
            seen["config"] = config

    class PassportStore:
        def __init__(self, path: Path) -> None:
            seen["db"] = path

    fake("avtopassport.pipeline.run", InspectionConfig=InspectionConfig, Inspector=Inspector)
    fake("avtopassport.passport.store", PassportStore=PassportStore)
    fake("avtopassport.passport.diff", diff=lambda p, c: {"new": c, "prev": p})
    fake("avtopassport.passport.report", build_json=lambda r, diff, decode, extra: {"r": r, "extra": extra},
         build_pdf=lambda report, out_path, images: Path(out_path))
    fake("avtopassport.recon3d.build", available=lambda: (False, "no env"),
         build=lambda frames_dir, out_dir, device, masks_dir: {"ok": True, "device": device, "masks": masks_dir})
    fake("avtopassport.recon3d.pins", project_pins=lambda out_dir, pins2d, frame_size: [frame_size])
    fake("avtopassport.tryon.recolor", load_palette=lambda: ["p"], recolor=lambda img, mask, hex_colour: hex_colour)

    b = Backends()
    assert isinstance(b.make_inspector("cpu"), Inspector) and seen["config"].device == "cpu"
    assert isinstance(b.open_store(tmp_path / "x.sqlite"), PassportStore)
    assert b.diff(None, [1]) == {"new": [1], "prev": None}
    assert b.build_json("R", diff=None, decode=None, extra={"a": 1}) == {"r": "R", "extra": {"a": 1}}
    assert b.build_pdf({}, tmp_path / "r.pdf", []) == tmp_path / "r.pdf"
    assert b.recon3d_available() == (False, "no env")
    assert b.recon3d_build(tmp_path, tmp_path, "cpu", None) == {"ok": True, "device": "cpu", "masks": None}
    assert b.project_pins(tmp_path, [], [1, 2]) == [[1, 2]]
    assert b.load_palette() == ["p"] and b.recolor(None, None, "#FFFFFF") == "#FFFFFF"

    # a missing 3D module is reported as unavailable, not raised
    monkeypatch.setitem(sys.modules, "avtopassport.recon3d.build", None)
    jm = JobManager(Settings(data_dir=tmp_path, device="cpu", warmup=False))
    ok, msg = jm.recon3d_available()
    assert ok is False and "not available" in msg


def test_real_passport_and_tryon_modules_end_to_end(tmp_path: Path) -> None:
    """Stub pipeline + 3D only; the real passport.store / diff / report and tryon.recolor do the rest."""
    pytest.importorskip("avtopassport.passport.store")
    pytest.importorskip("avtopassport.passport.diff")
    pytest.importorskip("avtopassport.passport.report")
    pytest.importorskip("avtopassport.tryon.recolor")
    w = World()
    stubbed = w.backends()
    real = Backends()
    backends = Backends(make_inspector=stubbed.make_inspector, recon3d_available=stubbed.recon3d_available,
                        recon3d_build=stubbed.recon3d_build, project_pins=stubbed.project_pins,
                        open_store=real.open_store, diff=real.diff, build_json=real.build_json,
                        build_pdf=real.build_pdf, load_palette=real.load_palette, recolor=real.recolor)
    s = Settings(data_dir=tmp_path / "data", device="cpu", warmup=False)
    app = create_app(settings=s, backends=backends, shops=ShopFinder(s.data_dir / "cache", fetch=w.fetch),
                     web_dir=tmp_path / "web")
    with TestClient(app) as c:
        first = run_inspection(c)
        assert first["status"] == "done" and first["warnings"] == [], first["warnings"]
        assert first["diff"]["first_inspection"] is True
        second = run_inspection(c)
        assert second["warnings"] == [] and second["diff"]["first_inspection"] is False
        assert [(f["part"], f["damage"]) for f in second["diff"]["new"]] == [("left_front_door", "dent")]
        assert second["report"]["hash"]["frames_sha256"] == "ab" * 32
        pdf = c.get(second["media"]["pdf"])
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
        h = c.get(f"/api/vin/{VIN}/history").json()
        assert [i["n_findings"] for i in h["inspections"]] == [2, 1]
        assert [(f["part"], f["damage"]) for f in h["diff_latest"]["new"]] == [("left_front_door", "dent")]
        pal = c.get("/api/palette").json()
        assert pal and all("hex" in p for p in pal)
        assert "White" in {p["name"] for p in pal} and "Black" not in {p["name"] for p in pal}  # tryon: false
        r = c.get(f"/api/inspections/{second['id']}/tryon", params={"colour": pal[0]["hex"].lstrip("#")})
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert (tmp_path / "data" / "passports.sqlite").is_file()


# --------------------------------------------------------------------------------------------- 3D view + HQ splats

def _client_with(tmp_path: Path, world: World, settings: dict[str, Any] | None = None,
                 **backend_overrides: Any) -> TestClient:
    s = Settings(data_dir=tmp_path / "data", device="cpu", warmup=False, **(settings or {}))
    backends = dataclasses.replace(world.backends(), **backend_overrides)
    app = create_app(settings=s, backends=backends,
                     shops=ShopFinder(s.data_dir / "cache", fetch=world.fetch), web_dir=tmp_path / "web")
    return TestClient(app)


CAMERAS_VIEW = {"center": [0.1, 0.2, 0.3], "up": [0.0, -1.0, 0.0], "forward0": [1.0, 0.0, 0.0],
                "walkaround_radius": 2.5}


def test_read_view_from_cameras_json(tmp_path: Path) -> None:
    assert read_view(tmp_path) is None  # no cameras.json
    (tmp_path / "cameras.json").write_text(json.dumps({**CAMERAS_VIEW, "cameras": []}))
    assert read_view(tmp_path) == {"center": [0.1, 0.2, 0.3], "up": [0.0, -1.0, 0.0], "forward0": [1.0, 0.0, 0.0],
                                   "radius": 2.5}
    (tmp_path / "cameras.json").write_text(json.dumps({**CAMERAS_VIEW, "walkaround_radius": 0}))
    assert read_view(tmp_path) is None
    (tmp_path / "cameras.json").write_text(json.dumps({**CAMERAS_VIEW, "up": [0, 1]}))
    assert read_view(tmp_path) is None


def test_recon3d_done_exposes_view(tmp_path: Path, world: World) -> None:
    world.recon_ok = True
    stub = world.backends().recon3d_build

    def build_with_cameras(frames_dir: Path, out_dir: Path, device: str, masks_dir: Path | None) -> dict[str, Any]:
        res = stub(frames_dir, out_dir, device, masks_dir)
        (Path(out_dir) / "cameras.json").write_text(json.dumps(CAMERAS_VIEW))
        return res

    with _client_with(tmp_path, world, recon3d_build=build_with_cameras) as c:
        st = run_inspection(c)
        assert st["recon3d"]["status"] == "done"
        assert st["recon3d"]["view"] == {"center": [0.1, 0.2, 0.3], "up": [0.0, -1.0, 0.0],
                                         "forward0": [1.0, 0.0, 0.0], "radius": 2.5}
        assert st["recon3d_hq"]["status"] == "disabled"  # AVTO_HQ3D not set: nothing for the web page to show
        assert c.get(f"/api/inspections/{st['id']}/recon3d/splat.ply").status_code == 404


def test_hq_stage_done_serves_splat(tmp_path: Path, world: World) -> None:
    world.recon_ok = True
    calls: list[tuple[Path, int, Any, Any]] = []

    def hq_build(frames_dir: Path, out_dir: Path, steps: int, pins2d: Any, masks_dir: Path | None) -> dict[str, Any]:
        calls.append((Path(frames_dir), steps, pins2d, masks_dir))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "splat.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\nSPLAT")
        (Path(out_dir) / "splat_car.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\nCAR")
        report = {"ok": True, "steps": steps, "n_images": 6, "n_registered": 6, "mean_reprojection_error_px": 0.5,
                  "psnr": None, "seconds": {"sfm": 1.0, "brush": 2.0, "total": 3.0},
                  "view": {"center": [0, 0, 0], "up": [0, 1, 0], "forward0": [1, 0, 0], "radius": 3.0},
                  "pins": [{"xyz": [0.0, 0.5, 1.0], "damage": "scratch", "part": "front_bumper", "score": 0.8}],
                  "crop": {"ok": True, "n_splats": 10, "n_splats_car": 7}}
        (Path(out_dir) / "hq_report.json").write_text(json.dumps(report))
        return {"ok": True, "error": None, "splat": str(Path(out_dir) / "splat_car.ply"),
                "splat_full": str(Path(out_dir) / "splat.ply"), "report": report, "seconds_wall": 3.0,
                "out_dir": str(out_dir)}

    with _client_with(tmp_path, world, {"enable_hq3d": True, "hq3d_steps": 123},
                      hq_available=lambda: (True, "stub GPU"), hq_build=hq_build) as c:
        assert c.get("/api/health").json()["hq3d_enabled"] is True
        st = run_inspection(c)
        hq = st["recon3d_hq"]
        assert hq["status"] == "done" and hq["error"] is None and hq["seconds"] == 3.0
        assert hq["splat"] == f"/api/inspections/{st['id']}/recon3d/splat.ply" and "_splat_path" not in hq
        assert hq["report"]["n_registered"] == 6 and hq["view"]["radius"] == 3.0 and hq["pins"][0]["part"] == "front_bumper"
        assert hq["previous_run"] is None and hq["n_frames"] == 6  # first HQ build on this server
        assert calls and calls[0][1] == 123 and len(calls[0][2]) == 1
        assert calls[0][0] == _jobs(c).frames_dir(st["id"])  # all of the inspection's blurred frames
        assert calls[0][3] == _jobs(c).masks_dir(st["id"])  # car masks for the crop / car PSNR
        assert hq["cropped"] is True and "_splat_full_path" not in hq
        r = c.get(hq["splat"])
        assert r.status_code == 200 and r.content.endswith(b"CAR")  # the car crop is the served model
        r = c.get(hq["splat_full"])
        assert hq["splat_full"].endswith("/recon3d/splat_full.ply") and r.content.endswith(b"SPLAT")
        # the next inspection reports the measured duration of this one while it waits
        last = _jobs(c).last_hq_run()
        assert last is not None and last["seconds_total"] == 3.0 and last["steps"] == 123


def test_hq_stage_unavailable_or_failing(tmp_path: Path, world: World) -> None:
    world.recon_ok = True
    with _client_with(tmp_path / "a", world, {"enable_hq3d": True},
                      hq_available=lambda: (False, "no Vulkan GPU")) as c:
        st = run_inspection(c)
        assert st["recon3d_hq"]["status"] == "unavailable" and st["recon3d_hq"]["error"] == "no Vulkan GPU"

    def failing(frames_dir: Path, out_dir: Path, steps: int, pins2d: Any, masks_dir: Any) -> dict[str, Any]:
        return {"ok": False, "error": "Brush failed (exit 1)", "splat": None, "report": {"ok": False},
                "seconds_wall": 0.2, "out_dir": str(out_dir)}

    with _client_with(tmp_path / "b", world, {"enable_hq3d": True},
                      hq_available=lambda: (True, "stub GPU"), hq_build=failing) as c:
        st = run_inspection(c)
        assert st["recon3d_hq"]["status"] == "error" and "Brush failed" in st["recon3d_hq"]["error"]
        assert c.get(f"/api/inspections/{st['id']}/recon3d/splat.ply").status_code == 404
        assert c.get(f"/api/inspections/{st['id']}/recon3d/splat_full.ply").status_code == 404


def test_hq_interrupted_by_restart_is_not_rerun(tmp_path: Path, world: World) -> None:
    s = Settings(data_dir=tmp_path / "data", device="cpu", warmup=False, enable_hq3d=True)
    jm = JobManager(s, world.backends())
    job_id = jm.new_job_id()
    jm.job_dir(job_id).mkdir(parents=True)
    state = {"id": job_id, "status": "done", "recon3d": {"status": "done"},
             "recon3d_hq": {"status": "running", "splat": None}}
    (jm.job_dir(job_id) / "state.json").write_text(json.dumps(state))
    jm.start()
    try:
        hq = jm.get(job_id)["recon3d_hq"]
        assert hq["status"] == "error" and "restart" in hq["error"]
        assert jm.queue_sizes()["recon3d_hq"] == 0
    finally:
        jm.stop()


def test_inspection_overrides_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AVTO_FPS", raising=False)
    monkeypatch.delenv("AVTO_MAX_FRAMES", raising=False)
    assert inspection_overrides() == {}
    monkeypatch.setenv("AVTO_FPS", "4")
    monkeypatch.setenv("AVTO_MAX_FRAMES", "160")
    assert inspection_overrides() == {"fps": 4.0, "max_frames": 160}
    monkeypatch.setenv("AVTO_FPS", "fast")
    monkeypatch.setenv("AVTO_MAX_FRAMES", "-3")
    assert inspection_overrides() == {}


# --------------------------------------------------------------------------------------------- reuse of 3D builds

def _hq_stub(calls: list[Any]) -> Any:
    def hq_build(frames_dir: Path, out_dir: Path, steps: int, pins2d: Any, masks_dir: Path | None) -> dict[str, Any]:
        calls.append(steps)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "splat.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\nSPLAT")
        (Path(out_dir) / "splat_car.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\nCAR")
        report = {"ok": True, "steps": steps, "n_images": 6, "n_registered": 6, "seconds": {"total": 3.0},
                  "view": {"center": [0, 0, 0], "up": [0, 1, 0], "forward0": [1, 0, 0], "radius": 3.0},
                  "pins": [{"xyz": [0.0, 0.5, 1.0], "damage": "scratch", "part": "front_bumper", "score": 0.8}],
                  "crop": {"ok": True, "n_splats": 10, "n_splats_car": 7}}
        (Path(out_dir) / "hq_report.json").write_text(json.dumps(report))
        return {"ok": True, "error": None, "splat": str(Path(out_dir) / "splat_car.ply"),
                "splat_full": str(Path(out_dir) / "splat.ply"), "report": report, "seconds_wall": 3.0,
                "out_dir": str(out_dir)}
    return hq_build


def _reuse_client(tmp_path: Path, world: World, hq_calls: list[Any], **settings: Any) -> TestClient:
    world.recon_ok = True
    return _client_with(tmp_path, world, {"enable_hq3d": True, **settings},
                        hq_available=lambda: (True, "stub GPU"), hq_build=_hq_stub(hq_calls))


def test_same_frames_reuse_3d_and_hq_without_building(tmp_path: Path, world: World) -> None:
    hq_calls: list[Any] = []
    with _reuse_client(tmp_path, world, hq_calls) as c:
        first = run_inspection(c)
        assert first["recon3d"]["status"] == "done" and first["recon3d_hq"]["status"] == "done"
        assert "reused" not in first["recon3d"] and world.recon_builds == 1 and len(hq_calls) == 1
        index = json.loads((tmp_path / "data" / "cache" / "recon3d_index.json").read_text())["entries"]
        assert sorted(e["kind"] for e in index.values()) == ["recon3d", "recon3d_hq"]
        assert all(e["job_id"] == first["id"] and "ab" * 32 in k for k, e in index.items())

        second = run_inspection(c)
        rec, hq = second["recon3d"], second["recon3d_hq"]
        assert world.recon_builds == 1 and len(hq_calls) == 1  # nothing rebuilt, nothing queued
        assert _jobs(c).queue_sizes() == {"inspections": 0, "recon3d": 0, "recon3d_hq": 0}
        for r, secs in ((rec, 1.5), (hq, 3.0)):
            assert r["status"] == "done" and r["reused"] is True and r["error"] is None
            assert r["reused_from_job"] == first["id"] and r["reused_from_created_utc"] == first["created_utc"]
            assert r["original_seconds"] == secs and r["seconds"] == secs  # the measured time of the original
            assert r["reuse_files"]["linked"] + r["reuse_files"]["copied"] >= 1
        assert rec["original_device"] == "cpu" and hq["original_device"]
        assert rec["ply"] == f"/api/inspections/{second['id']}/recon3d/points.ply" and "_ply_path" not in rec
        assert rec["pins"][0]["part"] == "front_bumper"  # re-projected for the new inspection's boxes
        assert hq["pins"] == first["recon3d_hq"]["pins"] and hq["pins_note"] is None  # identical 2D boxes
        assert hq["cropped"] is True and hq["report"]["n_registered"] == 6
        assert c.get(rec["ply"]).content.endswith(b"WEB")
        assert c.get(hq["splat"]).content.endswith(b"CAR") and c.get(hq["splat_full"]).content.endswith(b"SPLAT")
        job2 = tmp_path / "data" / "jobs" / second["id"]
        assert not (job2 / "upload").exists()  # the raw upload is still deleted
        assert (job2 / "recon3d_reused" / "points_web.ply").is_file() and not (job2 / "recon3d").exists()
        # the reused copies are not counted as a new measured HQ run, nor indexed
        assert _jobs(c).last_hq_run()["seconds_total"] == 3.0
        index2 = json.loads((tmp_path / "data" / "cache" / "recon3d_index.json").read_text())["entries"]
        assert index2 == index


@pytest.mark.parametrize("change", [{"frames_sha256": "cd" * 32}, {"fps": 4.0}])
def test_different_frames_or_config_rebuilds(tmp_path: Path, world: World, change: dict[str, Any]) -> None:
    hq_calls: list[Any] = []
    with _reuse_client(tmp_path, world, hq_calls) as c:
        run_inspection(c)
        for k, v in change.items():
            setattr(world.inspector, k, v)
        second = run_inspection(c)
        assert world.recon_builds == 2 and len(hq_calls) == 2
        assert "reused" not in second["recon3d"] and "reused" not in second["recon3d_hq"]
        assert len(json.loads((tmp_path / "data" / "cache" / "recon3d_index.json").read_text())["entries"]) == 4


def test_different_hq_steps_rebuilds_hq_only(tmp_path: Path, world: World) -> None:
    hq_calls: list[Any] = []
    with _reuse_client(tmp_path / "a", world, hq_calls) as c:
        first = run_inspection(c)
    world2 = World()
    # same frames and frame config, other Brush steps: the cloud is reused, the splats are rebuilt
    with _reuse_client(tmp_path / "a", world2, hq_calls, hq3d_steps=5000) as c:
        second = run_inspection(c)
        assert second["recon3d"]["reused_from_job"] == first["id"] and world2.recon_builds == 0
        assert "reused" not in second["recon3d_hq"] and hq_calls == [30_000, 5000]


def test_missing_source_files_fall_back_to_a_build(tmp_path: Path, world: World) -> None:
    hq_calls: list[Any] = []
    with _reuse_client(tmp_path, world, hq_calls) as c:
        first = run_inspection(c)
        job1 = tmp_path / "data" / "jobs" / first["id"]
        (job1 / "recon3d" / "depth.npz").unlink()
        (job1 / "recon3d_hq" / "splat_car.ply").unlink()
        second = run_inspection(c)
        assert world.recon_builds == 2 and len(hq_calls) == 2
        assert second["recon3d"]["status"] == "done" and "reused" not in second["recon3d"]
        assert second["recon3d_hq"]["status"] == "done" and "reused" not in second["recon3d_hq"]
        job2 = tmp_path / "data" / "jobs" / second["id"]
        assert not (job2 / "recon3d_reused").exists() and not (job2 / "recon3d_hq_reused").exists()
        third = run_inspection(c)  # the index now points at the second (newest) builds
        assert third["recon3d"]["reused_from_job"] == second["id"]
        assert third["recon3d_hq"]["reused_from_job"] == second["id"]


def test_reuse_index_survives_restart_and_is_backfilled(tmp_path: Path, world: World) -> None:
    hq_calls: list[Any] = []
    with _reuse_client(tmp_path, world, hq_calls) as c:
        first = run_inspection(c)
    index_path = tmp_path / "data" / "cache" / "recon3d_index.json"
    assert index_path.is_file()
    index_path.unlink()  # e.g. jobs from before the index existed: rebuilt from the job states at startup
    world2 = World()
    with _reuse_client(tmp_path, world2, hq_calls) as c:
        assert len(json.loads(index_path.read_text())["entries"]) == 2
        second = run_inspection(c)
        assert second["recon3d"]["reused_from_job"] == first["id"] and world2.recon_builds == 0
        assert second["recon3d_hq"]["reused_from_job"] == first["id"] and len(hq_calls) == 1


def test_reuse_disabled_always_builds(tmp_path: Path, world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVTO_REUSE_3D", "0")
    assert Settings.from_env(tmp_path).reuse_3d is False
    monkeypatch.delenv("AVTO_REUSE_3D")
    assert Settings.from_env(tmp_path).reuse_3d is True
    hq_calls: list[Any] = []
    with _reuse_client(tmp_path, world, hq_calls, reuse_3d=False) as c:
        run_inspection(c)
        second = run_inspection(c)
        assert world.recon_builds == 2 and len(hq_calls) == 2 and "reused" not in second["recon3d_hq"]


def test_attach_stored_hq_to_a_finished_job_after_the_fact(tmp_path: Path, world: World) -> None:
    """scripts/reuse_3d.py: a finished job whose HQ build was interrupted gets the stored build of the same input."""
    hq_calls: list[Any] = []
    with _reuse_client(tmp_path, world, hq_calls) as c:
        first = run_inspection(c)
    with _reuse_client(tmp_path, World(), hq_calls, reuse_3d=False) as c:
        second = run_inspection(c)
    job2 = tmp_path / "data" / "jobs" / second["id"] / "state.json"
    st = json.loads(job2.read_text())
    st["recon3d_hq"] = {**st["recon3d_hq"], "status": "error", "error": "interrupted (Brush killed)"}
    job2.write_text(json.dumps(st))

    import importlib.util
    spec = importlib.util.spec_from_file_location("reuse_3d_script", Path(__file__).parents[1] / "scripts" / "reuse_3d.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert mod.main(["--data-dir", str(tmp_path / "data"), "--job", second["id"]]) == 0
    hq = json.loads(job2.read_text())["recon3d_hq"]
    assert hq["status"] == "done" and hq["reused"] is True and hq["reused_from_job"] == first["id"]
    assert hq["replaced_error"] == "interrupted (Brush killed)" and hq["original_seconds"] == 3.0
    assert json.loads(job2.read_text())["recon3d"].get("reused") is None  # its own finished cloud is kept
    with _reuse_client(tmp_path, World(), hq_calls) as c:
        assert c.get(f"/api/inspections/{second['id']}/recon3d/splat.ply").content.endswith(b"CAR")
