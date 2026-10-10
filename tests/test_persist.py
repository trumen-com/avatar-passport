"""Opt-in HF snapshot persistence (api/persist.py). Everything is faked: no Hugging Face call, no token, no network."""

from __future__ import annotations

import dataclasses
import io
import json
import sqlite3
import tarfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_api import VIN, World, run_inspection  # stub pipeline + upload helper (same tests/ folder)

from avtopassport.api import jobs as jobs_mod
from avtopassport.api import persist
from avtopassport.api.app import create_app
from avtopassport.api.jobs import JobManager, Settings
from avtopassport.api.persist import ARCHIVE_IN_REPO, PersistSettings, Persister
from avtopassport.api.shops import ShopFinder
from avtopassport.passport.store import PassportStore

REPO = "someone/avtopassport-data"
JOB_A = "a" * 32
JOB_RUNNING = "b" * 32


# --------------------------------------------------------------------------------------------- fakes

class EntryNotFoundError(Exception):
    """Same class name as huggingface_hub's: persist classifies by name, so no import is needed."""


@dataclasses.dataclass
class FakeHub:
    """In-memory dataset repo: create_repo / repo_info / upload_file + an hf_hub_download stand-in."""

    files: dict[str, bytes] = dataclasses.field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = dataclasses.field(default_factory=list)
    private: bool = True
    fail: Exception | None = None

    def api(self, token: str | None) -> FakeHub:
        self.calls.append(("api", {"token": token}))
        return self

    def create_repo(self, repo_id: str, **kw: Any) -> None:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kw}))
        if self.fail:
            raise self.fail

    def repo_info(self, repo_id: str, **kw: Any) -> Any:
        return type("Info", (), {"private": self.private})()

    def upload_file(self, **kw: Any) -> None:
        self.calls.append(("upload_file", kw))
        if self.fail:
            raise self.fail
        self.files[kw["path_in_repo"]] = Path(kw["path_or_fileobj"]).read_bytes()

    def download(self, **kw: Any) -> str:
        self.calls.append(("download", kw))
        if self.fail:
            raise self.fail
        if kw["filename"] not in self.files:
            raise EntryNotFoundError(kw["filename"])
        out = Path(kw["cache_dir"]) / "dl" / kw["filename"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.files[kw["filename"]])
        return str(out)

    @property
    def uploads(self) -> int:
        return sum(1 for name, _ in self.calls if name == "upload_file")

    def archive(self) -> tarfile.TarFile:
        return tarfile.open(fileobj=io.BytesIO(self.files[ARCHIVE_IN_REPO]), mode="r:gz")


def make_persister(data_dir: Path, hub: FakeHub, **settings: Any) -> Persister:
    s = PersistSettings(repo_id=REPO, token="hf_fake", **settings)
    return Persister(data_dir, s, api_factory=hub.api, download=hub.download)


def seed_data(data_dir: Path, n: int = 2) -> None:
    """A real PassportStore with n inspections + one finished and one running job with every kind of file."""
    with PassportStore(data_dir / "passports.sqlite") as store:
        for i in range(n):
            store.add_inspection({"vin": VIN, "created_utc": f"2026-10-11T0{i}:00:00Z", "frames_sha256": "ab" * 32,
                                  "findings": [{"part": "front_bumper", "damage": "scratch"}]},
                                 media={"job_id": JOB_A})
    job = data_dir / "jobs" / JOB_A
    for sub in ("frames", "masks", "recon3d", "upload", "spin"):
        (job / sub).mkdir(parents=True, exist_ok=True)
    (job / "frames" / "f_00000.jpg").write_bytes(b"\xff\xd8 blurred frame")
    (job / "spin" / "f_00000.jpg").write_bytes(b"\xff\xd8 blurred frame")
    (job / "masks" / "f_00000.png").write_bytes(b"\x89PNG mask")
    (job / "recon3d" / "points.ply").write_bytes(b"ply\nend_header\n")
    (job / "upload" / "video.mp4").write_bytes(b"raw video")
    (job / "report.pdf").write_bytes(b"%PDF-1.4 stub")
    (job / "report.json").write_text(json.dumps({"vin": VIN}), encoding="utf-8")
    state = {"id": JOB_A, "status": "done", "inspection_id": 1, "warnings": [], "_pdf": "report.pdf",
             "media": {"frames": [f"/api/inspections/{JOB_A}/frames/f_00000.jpg"], "spin": [], "masks": [],
                       "pdf": f"/api/inspections/{JOB_A}/report.pdf"},
             "recon3d": {"status": "done", "ply": "x", "_ply_path": str(job / "recon3d" / "points.ply")},
             "recon3d_hq": {"status": "not_started"}}
    (job / "state.json").write_text(json.dumps(state), encoding="utf-8")
    running = data_dir / "jobs" / JOB_RUNNING
    running.mkdir(parents=True)
    (running / "state.json").write_text(json.dumps({"id": JOB_RUNNING, "status": "running"}), encoding="utf-8")


def count_rows(db: Path) -> int:
    conn = sqlite3.connect(db)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0])
    finally:
        conn.close()


def app_client(data_dir: Path, world: World, persister: Persister | None = None, *,
               real_store: bool = False) -> TestClient:
    backends = world.backends()
    if real_store:
        backends = dataclasses.replace(backends, open_store=jobs_mod._open_store)
    s = Settings(data_dir=data_dir, device="cpu", warmup=False)
    return TestClient(create_app(settings=s, backends=backends, persister=persister, web_dir=data_dir / "no-web",
                                 shops=ShopFinder(data_dir / "cache", fetch=world.fetch)))


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("AVTO_SNAPSHOT_REPO", "HF_TOKEN", "AVTO_SNAPSHOT_EVERY_MIN"):
        monkeypatch.delenv(name, raising=False)
    yield


# --------------------------------------------------------------------------------------------- tests

def test_disabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> Any:
        raise AssertionError("must not touch Hugging Face when disabled")

    monkeypatch.setattr(persist, "_hf_api", boom)
    monkeypatch.setattr(persist, "_hf_download", boom)
    seed_data(tmp_path)
    p = Persister.from_env(tmp_path)
    assert p.enabled is False
    assert p.restore() == "disabled" and p.snapshot_now() == "disabled"
    p.notify("x")
    p.start()
    assert p._thread is None
    p.stop()
    assert p.status() == {"enabled": False, "last_snapshot_utc": None, "last_error": None, "last_restore": None,
                          "every_min": None}
    with app_client(tmp_path / "app", World()) as c:
        assert c.get("/api/health").json()["persistence"]["enabled"] is False


def test_repo_without_token_stays_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVTO_SNAPSHOT_REPO", REPO)
    monkeypatch.setenv("AVTO_SNAPSHOT_EVERY_MIN", "nonsense")
    p = Persister.from_env(tmp_path)
    assert p.enabled is False and p.settings.every_min == 5.0
    assert "HF_TOKEN" in (p.status()["last_error"] or "")


def test_snapshot_archive_contents(tmp_path: Path) -> None:
    seed_data(tmp_path, n=3)
    hub = FakeHub()
    p = make_persister(tmp_path, hub)
    assert p.snapshot_now() == "uploaded"
    create = next(kw for name, kw in hub.calls if name == "create_repo")
    assert create == {"repo_id": REPO, "repo_type": "dataset", "private": True, "exist_ok": True}
    upload = next(kw for name, kw in hub.calls if name == "upload_file")
    assert upload["repo_type"] == "dataset" and upload["repo_id"] == REPO and upload["path_in_repo"] == ARCHIVE_IN_REPO
    with hub.archive() as tar:
        names = set(tar.getnames())
        assert names == {"manifest.json", "passports.sqlite", f"jobs/{JOB_A}/state.json",
                         f"jobs/{JOB_A}/report.json", f"jobs/{JOB_A}/report.pdf"}
        manifest = json.loads(tar.extractfile("manifest.json").read())  # type: ignore[union-attr]
        for name, meta in manifest["files"].items():
            data = tar.extractfile(name).read()  # type: ignore[union-attr]
            assert persist._sha256(data) == meta["sha256"] and len(data) == meta["bytes"]
        db_copy = tmp_path / "out" / "passports.sqlite"
        db_copy.parent.mkdir()
        db_copy.write_bytes(tar.extractfile("passports.sqlite").read())  # type: ignore[union-attr]
    assert not any(part in n for n in names for part in ("frames", "masks", "recon3d", "upload", "spin", JOB_RUNNING))
    assert manifest["counts"] == {"inspections": 3, "jobs": 1, "states": 1, "reports_json": 1, "pdfs": 1}
    assert manifest["created_utc"].endswith("Z") and "frames" in manifest["not_stored"]
    with PassportStore(db_copy) as store:  # the DB in the archive is restorable
        assert len(store.history(VIN)) == 3
    assert p.status()["last_snapshot_utc"] == manifest["created_utc"] and p.status()["last_error"] is None


def test_refuses_public_repo(tmp_path: Path) -> None:
    seed_data(tmp_path)
    hub = FakeHub(private=False)
    p = make_persister(tmp_path, hub)
    assert p.snapshot_now() == "error"
    assert hub.uploads == 0 and "not private" in (p.status()["last_error"] or "")


def test_backup_is_consistent_while_a_writer_holds_a_transaction(tmp_path: Path) -> None:
    db = tmp_path / "passports.sqlite"
    with PassportStore(db) as store:
        store.add_inspection({"vin": VIN, "findings": []})
    writer = sqlite3.connect(db, timeout=10, check_same_thread=False)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO inspections (vin, created_utc) VALUES ('UNCOMMITTED', 'x')")
    copy = tmp_path / "copy.sqlite"
    persist.backup_db(db, copy)
    assert persist._db_ok(copy) == (True, 1)  # committed data only, never a torn file
    writer.commit()
    persist.backup_db(db, copy)
    assert persist._db_ok(copy) == (True, 2)

    stop = threading.Event()

    def keep_writing() -> None:  # bounded: at most 300 small transactions of 20 rows
        for _ in range(300):
            if stop.is_set():
                return
            with writer:
                writer.executemany("INSERT INTO inspections (vin, result_json) VALUES (?, ?)", [(VIN, "x")] * 20)

    t = threading.Thread(target=keep_writing)
    t.start()
    try:
        counts = []
        for _ in range(5):
            persist.backup_db(db, copy)
            ok, n = persist._db_ok(copy)
            assert ok and n % 20 == 2  # whole transactions only
            counts.append(n)
    finally:
        stop.set()
        t.join()
        writer.close()
    assert counts == sorted(counts)


def test_restore_into_empty_dir(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    seed_data(src, n=2)
    hub = FakeHub()
    assert make_persister(src, hub).snapshot_now() == "uploaded"
    p = make_persister(dst, hub)
    msg = p.restore()
    assert msg.startswith("restored snapshot") and "2 inspections" in msg
    assert count_rows(dst / "passports.sqlite") == 2
    assert (dst / "jobs" / JOB_A / "report.pdf").read_bytes() == b"%PDF-1.4 stub"
    assert not (dst / "jobs" / JOB_RUNNING).exists() and not (dst / "jobs" / JOB_A / "frames").exists()
    st = json.loads((dst / "jobs" / JOB_A / "state.json").read_text(encoding="utf-8"))
    assert st["media"]["frames"] == [] and st["media"]["pdf"].endswith("report.pdf")  # no dangling frame URLs
    assert st["recon3d"]["status"] == "unavailable" and "_ply_path" not in st["recon3d"]
    assert persist.RESTORED_NOTE in st["warnings"]
    jm = JobManager(Settings(data_dir=dst, warmup=False, enable_recon3d=False))
    assert jm.get(JOB_A)["status"] == "done" and jm.resolve_pdf(JOB_A) is not None
    assert p.status()["last_error"] is None and p.status()["last_restore"] == msg


def test_restore_never_overwrites_a_local_db(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    seed_data(src, n=3)
    hub = FakeHub()
    make_persister(src, hub).snapshot_now()
    with PassportStore(dst / "passports.sqlite") as store:
        store.add_inspection({"vin": "LOCALNEWER", "findings": []})
    before = (dst / "passports.sqlite").read_bytes()
    msg = make_persister(dst, hub).restore()
    assert "not restored" in msg and not any(name == "download" for name, _ in hub.calls)
    assert (dst / "passports.sqlite").read_bytes() == before
    # extract_snapshot itself never overwrites anything either (DB or job files)
    archive = tmp_path / "snap.tar.gz"
    archive.write_bytes(hub.files[ARCHIVE_IN_REPO])
    (dst / "jobs" / JOB_A).mkdir(parents=True)
    (dst / "jobs" / JOB_A / "state.json").write_text("{\"local\": true}", encoding="utf-8")
    res = persist.extract_snapshot(archive, dst)
    assert res["db"] is False and (dst / "passports.sqlite").read_bytes() == before
    assert (dst / "jobs" / JOB_A / "state.json").read_text(encoding="utf-8") == "{\"local\": true}"
    assert (dst / "jobs" / JOB_A / "report.pdf").is_file()


def test_network_error_app_still_starts(tmp_path: Path) -> None:
    hub = FakeHub(fail=ConnectionError("network unreachable"))
    p = make_persister(tmp_path / "data", hub)
    with app_client(tmp_path / "data", World(), p, real_store=True) as c:
        h = c.get("/api/health").json()
        assert h["ok"] is True and h["persistence"]["enabled"] is True
        assert "restore failed" in h["persistence"]["last_error"]
        assert run_inspection(c)["status"] == "done"  # the app works without its snapshot
    assert "snapshot failed" in (p.status()["last_error"] or "")  # the shutdown snapshot failed too, quietly
    assert p._thread is None


def test_missing_snapshot_is_not_an_error(tmp_path: Path) -> None:
    p = make_persister(tmp_path, FakeHub())
    assert "no snapshot" in p.restore() and p.status()["last_error"] is None


def test_env_wiring_with_monkeypatched_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = FakeHub()
    monkeypatch.setattr(persist, "_hf_api", hub.api)
    monkeypatch.setattr(persist, "_hf_download", hub.download)
    monkeypatch.setenv("AVTO_SNAPSHOT_REPO", REPO)
    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    monkeypatch.setenv("AVTO_SNAPSHOT_EVERY_MIN", "60")
    world = World()
    with app_client(tmp_path / "a", world, real_store=True) as c:
        assert c.get("/api/health").json()["persistence"]["every_min"] == 60.0
        assert run_inspection(c)["status"] == "done"
    assert hub.uploads == 1 and ("api", {"token": "hf_fake"}) in hub.calls  # final snapshot at shutdown
    # restart on an empty disk: restore, then a repeat inspection of the same VIN shows the diff
    with app_client(tmp_path / "b", world, real_store=True) as c:
        assert "restored snapshot" in c.get("/api/health").json()["persistence"]["last_restore"]
        assert len(c.get(f"/api/vin/{VIN}/history").json()["inspections"]) == 1
        st = run_inspection(c)
        assert st["diff"]["previous_inspection_id"] == 1
        assert [(f["part"], f["damage"]) for f in st["diff"]["new"]] == [("left_front_door", "dent")]


def test_skip_when_unchanged_and_debounce(tmp_path: Path) -> None:
    seed_data(tmp_path, n=1)
    hub = FakeHub()
    p = make_persister(tmp_path, hub, every_min=60.0, debounce_s=0.3)
    assert p.snapshot_now() == "uploaded"
    assert p.snapshot_now() == "unchanged" and hub.uploads == 1
    p.start()
    try:
        with PassportStore(tmp_path / "passports.sqlite") as store:
            for _ in range(5):  # five inspections in quick succession -> one snapshot
                store.add_inspection({"vin": VIN, "findings": []})
                p.notify(JOB_A)
        deadline = time.monotonic() + 5
        while hub.uploads < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.6)
        assert hub.uploads == 2
    finally:
        p.stop()
    assert hub.uploads == 2  # the final snapshot found nothing new
    with hub.archive() as tar:
        db = tmp_path / "check.sqlite"
        db.write_bytes(tar.extractfile("passports.sqlite").read())  # type: ignore[union-attr]
    assert count_rows(db) == 6


def test_job_manager_calls_the_hook_after_storing(tmp_path: Path) -> None:
    seen: list[str] = []
    world = World()
    with app_client(tmp_path, world) as c:
        c.app.state.jobs.on_stored = seen.append  # type: ignore[attr-defined]
        st = run_inspection(c)
        no_vin = run_inspection(c, vin_text=None)
        world.inspector.fail = True
        assert run_inspection(c)["status"] == "error"  # nothing stored -> no snapshot trigger
    assert seen == [st["id"], no_vin["id"]]
