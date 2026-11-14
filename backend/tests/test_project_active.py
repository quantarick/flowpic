"""Which project the page reopens after a reload, and what a restart does to running tasks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.models import ProjectConfig, TaskRecord, TaskStatus
from app.routers import project as project_router
from app.services import task_db

T0 = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(task_db, "_DB_PATH", tmp_path / "tasks.db")
    task_db._init_db()
    return tmp_path


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(project_router.router, prefix="/api/project")
    return TestClient(app)


def make_project(store, name: str, photos: int = 2, created: str = "2026-11-14T08:00:00+00:00") -> None:
    folder = store / name
    (folder / "images").mkdir(parents=True)
    (folder / "meta.json").write_text('{"created_at": "%s", "config": {}}' % created)
    for i in range(photos):
        (folder / "images" / f"p{i}.jpg").write_bytes(b"x")


def add_task(task_id: str, project_id: str, status: TaskStatus, minutes: int = 0) -> None:
    task_db.save_task(TaskRecord(
        task_id=task_id, project_id=project_id, image_count=2, config=ProjectConfig(),
        created_at=T0 + timedelta(minutes=minutes),
    ))
    if status is not TaskStatus.PENDING:
        task_db.update_task_status(task_id, status)


def active(client) -> str | None:
    body = client.get("/api/project/active").json()
    return body["project_id"] if body else None


def test_a_project_that_was_never_run_is_reopened(store, client):
    make_project(store, "fresh")
    assert active(client) == "fresh"


def test_a_running_project_is_reopened(store, client):
    make_project(store, "busy")
    add_task("t1", "busy", TaskStatus.RENDERING)
    assert active(client) == "busy"


@pytest.mark.parametrize("status", [TaskStatus.FAILED, TaskStatus.CANCELLED])
def test_a_project_whose_run_failed_or_was_cancelled_is_reopened(store, client, status):
    # Before this fix the page came back empty: the photos were on disk but
    # nothing loaded them, so a setting could not be changed before trying again.
    make_project(store, "stuck")
    add_task("t1", "stuck", status)
    assert active(client) == "stuck"


def test_a_project_whose_run_completed_is_not_reopened(store, client):
    make_project(store, "finished")
    add_task("t1", "finished", TaskStatus.DONE)
    assert active(client) is None


def test_only_the_latest_run_decides(store, client):
    make_project(store, "retried_ok")
    add_task("a1", "retried_ok", TaskStatus.FAILED, minutes=0)
    add_task("a2", "retried_ok", TaskStatus.DONE, minutes=5)

    make_project(store, "broke_later")
    add_task("b1", "broke_later", TaskStatus.DONE, minutes=0)
    add_task("b2", "broke_later", TaskStatus.FAILED, minutes=5)

    assert active(client) == "broke_later"


def test_an_empty_project_is_skipped(store, client):
    make_project(store, "empty", photos=0)
    assert active(client) is None


def test_restart_marks_interrupted_tasks_as_failed(store, client):
    make_project(store, "cut_off")
    add_task("live", "cut_off", TaskStatus.MATCHING, minutes=0)
    add_task("queued", "cut_off", TaskStatus.PENDING, minutes=1)
    add_task("old_ok", "cut_off", TaskStatus.DONE, minutes=-60)

    assert task_db.fail_unfinished("Interrupted: the server stopped") == 2

    by_id = {t.task_id: t for t in task_db.list_tasks()}
    assert by_id["live"].status is TaskStatus.FAILED
    assert by_id["live"].error_message == "Interrupted: the server stopped"
    assert by_id["live"].finished_at is not None
    assert by_id["queued"].status is TaskStatus.FAILED
    assert by_id["old_ok"].status is TaskStatus.DONE and by_id["old_ok"].error_message is None
    assert task_db.fail_unfinished("again") == 0

    # The project is still the one to reopen: its latest run did not complete.
    assert active(client) == "cut_off"
