"""Reset: delete every project and all history.

A project is a folder in the data directory that holds a `meta.json`. Deleting
it removes its photos, music, captions, stage caches, crops and rendered
videos. The task history and the record of published photos are emptied too.

Anything else in the data directory is left alone. That keeps the Xiaohongshu
login and the style profile, which are account settings and not project data.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def _project_dirs() -> list[Path]:
    data_dir = settings.data_dir
    if not data_dir.exists():
        return []
    return sorted(d for d in data_dir.iterdir() if d.is_dir() and (d / "meta.json").exists())


def _count_files(folder: Path, pattern: str = "*") -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.glob(pattern) if f.is_file())


def _tree_size(folder: Path) -> int:
    total = 0
    for f in folder.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def preview() -> dict:
    """What a reset would delete, as counts and a size in bytes."""
    from app.services.task_db import count_history

    projects = _project_dirs()
    history = count_history()
    return {
        "projects": len(projects),
        "photos": sum(_count_files(d / "images") for d in projects),
        "songs": sum(_count_files(d, "music.*") for d in projects),
        "videos": sum(_count_files(d / "output", "*.mp4") for d in projects),
        "tasks": history["tasks"],
        "published": history["published"],
        "bytes": sum(_tree_size(d) for d in projects),
    }


def reset_all() -> dict:
    """Delete every project folder, every task record and the published-photo record.

    A file that is open elsewhere cannot be deleted on Windows. Such a project
    is reported in `failed` and whatever could be removed from it is removed.
    """
    from app.services.task_db import clear_history

    before = preview()
    failed: list[str] = []

    for project in _project_dirs():
        shutil.rmtree(project, ignore_errors=True)
        if project.exists():
            failed.append(project.name)
            logger.warning("Reset could not fully delete project %s (a file is in use)", project.name)

    cleared = clear_history()
    left = sum(_tree_size(settings.data_dir / name) for name in failed)

    result = {
        "projects_deleted": before["projects"] - len(failed),
        "tasks_deleted": cleared["tasks"],
        "published_deleted": cleared["published"],
        "bytes_freed": max(0, before["bytes"] - left),
        "failed": failed,
    }
    logger.info("Reset done: %s", result)
    return result
