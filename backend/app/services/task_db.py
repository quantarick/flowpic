"""SQLite-backed task history for persisting completed tasks across restarts."""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import settings
from app.models import ProjectConfig, TaskRecord, TaskStatus

logger = logging.getLogger(__name__)

_DB_PATH = settings.data_dir / "tasks.db"
_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            finished_at TEXT,
            output_path TEXT,
            image_count INTEGER DEFAULT 0,
            duration_seconds REAL,
            config_json TEXT,
            error_message TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_tasks_created
        ON tasks (created_at DESC)
    """)
    # Migrate: add error_message column if missing
    try:
        conn.execute("SELECT error_message FROM tasks LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE tasks ADD COLUMN error_message TEXT")
    conn.commit()
    conn.close()
    logger.info(f"Task DB initialized at {_DB_PATH}")


# Initialize on import
_init_db()


def save_task(record: TaskRecord):
    """Insert or update a task record."""
    config_json = record.config.model_dump_json() if record.config else None
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                """INSERT INTO tasks
                   (task_id, project_id, status, created_at, finished_at,
                    output_path, image_count, duration_seconds, config_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(task_id) DO UPDATE SET
                    status=excluded.status,
                    finished_at=excluded.finished_at,
                    output_path=excluded.output_path,
                    image_count=excluded.image_count,
                    duration_seconds=excluded.duration_seconds,
                    config_json=excluded.config_json
                """,
                (
                    record.task_id,
                    record.project_id,
                    record.status.value,
                    record.created_at.isoformat(),
                    record.finished_at.isoformat() if record.finished_at else None,
                    record.output_path,
                    record.image_count,
                    record.duration_seconds,
                    config_json,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def update_task_status(
    task_id: str,
    status: TaskStatus,
    output_path: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    error_message: Optional[str] = None,
):
    """Update status and optional fields for an existing task."""
    with _lock:
        conn = _get_conn()
        try:
            finished = datetime.now(timezone.utc).isoformat() if status in (
                TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED
            ) else None
            conn.execute(
                """UPDATE tasks SET status=?, finished_at=?,
                   output_path=COALESCE(?, output_path),
                   duration_seconds=COALESCE(?, duration_seconds),
                   error_message=COALESCE(?, error_message)
                   WHERE task_id=?""",
                (status.value, finished, output_path, duration_seconds,
                 error_message, task_id),
            )
            conn.commit()
        finally:
            conn.close()


def list_tasks(limit: int = 50, offset: int = 0) -> list[TaskRecord]:
    """List tasks ordered by most recent first."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [_row_to_record(r) for r in rows]
    finally:
        conn.close()


def get_task(task_id: str) -> Optional[TaskRecord]:
    """Get a single task by ID."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        return _row_to_record(row) if row else None
    finally:
        conn.close()


def _row_to_record(row: sqlite3.Row) -> TaskRecord:
    config = None
    if row["config_json"]:
        config = ProjectConfig(**json.loads(row["config_json"]))
    return TaskRecord(
        task_id=row["task_id"],
        project_id=row["project_id"],
        status=TaskStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        output_path=row["output_path"],
        image_count=row["image_count"],
        duration_seconds=row["duration_seconds"],
        config=config,
        error_message=row["error_message"] if "error_message" in row.keys() else None,
    )
