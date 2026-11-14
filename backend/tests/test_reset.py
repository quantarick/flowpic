"""The reset feature: what it counts, what it deletes, what it keeps, and when it refuses."""

from __future__ import annotations

import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.dependencies import get_task_manager
from app.models import ProjectConfig, TaskRecord
from app.routers import reset as reset_router
from app.services import reset, task_db
from app.services.task_manager import TaskManager


def make_project(data_dir, name: str, photos: int, videos: int = 0) -> None:
    project = data_dir / name
    (project / "images").mkdir(parents=True)
    (project / "meta.json").write_text("{}")
    for i in range(photos):
        (project / "images" / f"photo{i}.jpg").write_bytes(b"x" * 100)
    (project / "music.mp3").write_bytes(b"m" * 500)
    (project / "captions").mkdir()
    (project / "captions" / "photo0.json").write_text("{}")
    (project / "cache").mkdir()
    (project / "cache" / "matches.json").write_text("[]")
    (project / "output" / "crops").mkdir(parents=True)
    (project / "output" / "crops" / "000_photo0_crop.png").write_bytes(b"c" * 50)
    for i in range(videos):
        (project / "output" / f"task{i}.mp4").write_bytes(b"v" * 1000)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A data directory with two projects, some history, and files a reset must keep."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(task_db, "_DB_PATH", tmp_path / "tasks.db")
    task_db._init_db()

    make_project(tmp_path, "aaaa", photos=3, videos=2)
    make_project(tmp_path, "bbbb", photos=2)
    for task_id, project_id in (("t1", "aaaa"), ("t2", "aaaa"), ("t3", "bbbb")):
        task_db.save_task(TaskRecord(task_id=task_id, project_id=project_id, image_count=2, config=ProjectConfig()))
    task_db.record_published_images("aaaa", ["000_photo0_crop.png"])

    # Not projects: no meta.json. A reset leaves these alone.
    (tmp_path / "xhs_cookies.json").write_text("{}")
    (tmp_path / "xhs_style_profile.json").write_text("{}")
    (tmp_path / "xhs_debug").mkdir()
    (tmp_path / "xhs_debug" / "01.png").write_bytes(b"p")
    return tmp_path


def client_with(tm) -> TestClient:
    app = FastAPI()
    app.include_router(reset_router.router, prefix="/api")
    app.dependency_overrides[get_task_manager] = lambda: tm
    return TestClient(app)


class IdleManager:
    def __init__(self, running: bool = False):
        self.running = running
        self.forgot = False

    def has_running(self) -> bool:
        return self.running

    def forget_finished(self) -> None:
        self.forgot = True


def test_preview_counts_without_deleting(store):
    assert reset.preview() == {
        "projects": 2,
        "photos": 5,
        "songs": 2,
        "videos": 2,
        "tasks": 3,
        "published": 1,
        "bytes": 3 * 100 + 2 * 100 + 2 * 500 + 2 * 1000 + 2 * 50 + 2 * (2 + 2 + 2),
    }
    assert (store / "aaaa" / "images" / "photo0.jpg").exists()


def test_reset_deletes_projects_and_history(store):
    result = reset.reset_all()

    assert result["projects_deleted"] == 2 and result["tasks_deleted"] == 3
    assert result["published_deleted"] == 1
    assert result["failed"] == [] and result["bytes_freed"] > 0
    assert not (store / "aaaa").exists() and not (store / "bbbb").exists()
    assert task_db.list_tasks() == []
    assert task_db.get_published_image_hashes() == set()
    assert reset.preview() == {
        "projects": 0, "photos": 0, "songs": 0, "videos": 0, "tasks": 0, "published": 0, "bytes": 0,
    }


def test_reset_keeps_the_login_and_the_style_profile(store):
    reset.reset_all()

    assert (store / "xhs_cookies.json").exists()
    assert (store / "xhs_style_profile.json").exists()
    assert (store / "xhs_debug" / "01.png").exists()
    assert (store / "tasks.db").exists()  # emptied, not removed


def test_reset_on_an_empty_store_is_harmless(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(task_db, "_DB_PATH", tmp_path / "tasks.db")
    task_db._init_db()

    assert reset.reset_all() == {
        "projects_deleted": 0, "tasks_deleted": 0, "published_deleted": 0, "bytes_freed": 0, "failed": [],
    }


def test_routes_preview_then_reset(store):
    tm = IdleManager()
    client = client_with(tm)

    assert client.get("/api/reset/preview").json()["photos"] == 5
    response = client.post("/api/reset")

    assert response.status_code == 200 and response.json()["projects_deleted"] == 2
    assert tm.forgot is True
    assert client.get("/api/reset/preview").json()["projects"] == 0


def test_reset_refuses_while_a_task_runs(store):
    response = client_with(IdleManager(running=True)).post("/api/reset")

    assert response.status_code == 409
    assert "still running" in response.json()["detail"]
    assert (store / "aaaa" / "images" / "photo0.jpg").exists()
    assert len(task_db.list_tasks()) == 3
    assert len(task_db.get_published_image_hashes()) == 1


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://192.168.1.20:5173",
        "http://10.0.0.7:5173",
        "http://my-laptop:5173",
        "http://my-laptop.local:5173",
        "http://[::1]:5173",
    ],
)
def test_reset_is_allowed_from_this_machine_or_the_lan(store, origin):
    response = client_with(IdleManager()).post("/api/reset", headers={"Origin": origin})
    assert response.status_code == 200


@pytest.mark.parametrize("origin", ["https://evil.example.com", "http://8.8.8.8", "null"])
def test_reset_refuses_a_page_on_the_public_internet(store, origin):
    response = client_with(IdleManager()).post("/api/reset", headers={"Origin": origin})

    assert response.status_code == 403
    assert (store / "aaaa").exists()


def test_task_manager_knows_when_work_is_running(store):
    tm = TaskManager(max_workers=1)
    started, release = threading.Event(), threading.Event()

    def work(progress_callback, cancel_check):
        started.set()
        release.wait(timeout=10)
        return "out.mp4"

    task_db.save_task(TaskRecord(task_id="live", project_id="aaaa", image_count=1, config=ProjectConfig()))
    try:
        assert tm.has_running() is False
        tm.submit(task_id="live", fn=work)
        assert started.wait(timeout=10)
        assert tm.has_running() is True

        release.set()
        tm._futures["live"].result(timeout=10)
        assert tm.has_running() is False

        tm.forget_finished()
        assert tm.get_status("live") is None
    finally:
        release.set()
        tm.shutdown()
