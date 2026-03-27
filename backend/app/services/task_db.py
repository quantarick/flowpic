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
    # Migrate: add task_type column if missing
    try:
        conn.execute("SELECT task_type FROM tasks LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE tasks ADD COLUMN task_type TEXT DEFAULT 'video'")
    # Published images tracking table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS published_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            crop_filename TEXT NOT NULL,
            image_hash TEXT NOT NULL,
            crop_mode TEXT NOT NULL,
            published_at TEXT NOT NULL,
            post_url TEXT,
            note_id TEXT,
            UNIQUE(image_hash, crop_mode)
        )
    """)
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
                    output_path, image_count, duration_seconds, config_json, task_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    record.task_type,
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


def list_tasks(limit: int = 50, offset: int = 0, task_type: Optional[str] = None) -> list[TaskRecord]:
    """List tasks ordered by most recent first, optionally filtered by task_type."""
    conn = _get_conn()
    try:
        if task_type:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE task_type=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (task_type, limit, offset),
            ).fetchall()
        else:
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


def _parse_crop_filename(filename: str) -> tuple[str, str]:
    """Extract image_hash and crop_mode from a crop filename.

    Expected format: {index}_{hash}_{mode}.png
    e.g. "001_0f64301fd3c6_crop.png" -> ("0f64301fd3c6", "crop")
    """
    stem = Path(filename).stem  # "001_0f64301fd3c6_crop"
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        prefix, mode = parts
        # Extract hash: skip the index prefix
        hash_parts = prefix.split("_", 1)
        image_hash = hash_parts[1] if len(hash_parts) == 2 else prefix
        return image_hash, mode
    return stem, "unknown"


def record_published_images(
    project_id: str,
    crop_filenames: list[str],
    post_url: str | None = None,
    note_id: str | None = None,
):
    """Record crop images that were successfully published."""
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _get_conn()
        try:
            for fname in crop_filenames:
                image_hash, crop_mode = _parse_crop_filename(fname)
                conn.execute(
                    """INSERT OR IGNORE INTO published_images
                       (project_id, crop_filename, image_hash, crop_mode,
                        published_at, post_url, note_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (project_id, fname, image_hash, crop_mode, now, post_url, note_id),
                )
            conn.commit()
            logger.info("Recorded %d published images for project %s", len(crop_filenames), project_id)
        finally:
            conn.close()


def get_published_image_hashes() -> set[str]:
    """Return set of '{hash}_{mode}' strings for all published images."""
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT image_hash, crop_mode FROM published_images").fetchall()
        return {f"{r['image_hash']}_{r['crop_mode']}" for r in rows}
    finally:
        conn.close()


def get_published_images_for_project(project_id: str) -> list[dict]:
    """Return published image records for a specific project."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT crop_filename, image_hash, crop_mode, published_at, post_url, note_id "
            "FROM published_images WHERE project_id = ? ORDER BY published_at DESC",
            (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def filter_unpublished_crops(crop_filenames: list[str]) -> list[str]:
    """Return only filenames that haven't been published yet."""
    published = get_published_image_hashes()
    result = []
    for fname in crop_filenames:
        image_hash, crop_mode = _parse_crop_filename(fname)
        key = f"{image_hash}_{crop_mode}"
        if key not in published:
            result.append(fname)
    return result


def _row_to_record(row: sqlite3.Row) -> TaskRecord:
    config = None
    if row["config_json"]:
        config = ProjectConfig(**json.loads(row["config_json"]))
    return TaskRecord(
        task_id=row["task_id"],
        project_id=row["project_id"],
        status=TaskStatus(row["status"]),
        task_type=row["task_type"] if "task_type" in row.keys() else "video",
        created_at=datetime.fromisoformat(row["created_at"]),
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        output_path=row["output_path"],
        image_count=row["image_count"],
        duration_seconds=row["duration_seconds"],
        config=config,
        error_message=row["error_message"] if "error_message" in row.keys() else None,
    )
