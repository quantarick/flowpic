import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models import ProjectConfig, ProjectCreateResponse, ProjectInfo

router = APIRouter()


def _project_dir(project_id: str) -> Path:
    p = settings.data_dir / project_id
    if not p.exists():
        raise HTTPException(404, "Project not found")
    return p


def _load_project_info(proj_dir: Path) -> dict:
    meta_path = proj_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def _save_project_meta(proj_dir: Path, meta: dict):
    (proj_dir / "meta.json").write_text(json.dumps(meta, default=str))


@router.post("/create", response_model=ProjectCreateResponse)
async def create_project():
    project_id = uuid.uuid4().hex[:16]
    proj_dir = settings.data_dir / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "images").mkdir(exist_ok=True)

    meta = {
        "project_id": project_id,
        "config": ProjectConfig().model_dump(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_project_meta(proj_dir, meta)

    return ProjectCreateResponse(project_id=project_id)


@router.get("/active", response_model=Optional[ProjectInfo])
async def get_active_project():
    """Find the most recent project that has uploads but no completed task,
    i.e. either never started or still running."""
    from app.services.task_db import list_tasks

    data_dir = settings.data_dir
    if not data_dir.exists():
        return None

    # Build set of project_ids that have a terminal task
    all_tasks = list_tasks(limit=500)
    terminal = {"done", "failed", "cancelled"}
    # Projects with at least one active (non-terminal) task
    active_project_ids = {
        t.project_id for t in all_tasks if t.status.value not in terminal
    }
    # Projects where ALL tasks are terminal
    finished_project_ids = set()
    project_tasks: dict[str, list] = {}
    for t in all_tasks:
        project_tasks.setdefault(t.project_id, []).append(t)
    for pid, tasks in project_tasks.items():
        if all(t.status.value in terminal for t in tasks):
            finished_project_ids.add(pid)

    # Scan project directories, find candidates
    candidates = []
    for d in data_dir.iterdir():
        if not d.is_dir() or d.name == "__pycache__":
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        images_dir = d / "images"
        image_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
        if image_count == 0:
            continue  # empty project, skip

        pid = d.name
        has_active_task = pid in active_project_ids
        has_no_task = pid not in project_tasks
        if has_active_task or has_no_task:
            meta = json.loads(meta_path.read_text())
            created = meta.get("created_at", "")
            candidates.append((created, pid, d, meta, image_count))

    if not candidates:
        return None

    # Return most recent candidate
    candidates.sort(reverse=True)
    _, pid, proj_dir, meta, _ = candidates[0]
    images_dir = proj_dir / "images"
    images = sorted(f.name for f in images_dir.glob("*")) if images_dir.exists() else []
    music_files = list(proj_dir.glob("music.*"))
    music = music_files[0].name if music_files else None

    return ProjectInfo(
        project_id=pid,
        images=images,
        music=music,
        config=ProjectConfig(**meta.get("config", {})),
        created_at=meta.get("created_at", datetime.now(timezone.utc).isoformat()),
    )


@router.get("/{project_id}", response_model=ProjectInfo)
async def get_project(project_id: str):
    proj_dir = _project_dir(project_id)
    meta = _load_project_info(proj_dir)

    images_dir = proj_dir / "images"
    images = sorted(f.name for f in images_dir.glob("*")) if images_dir.exists() else []

    music_files = list(proj_dir.glob("music.*"))
    music = music_files[0].name if music_files else None

    return ProjectInfo(
        project_id=project_id,
        images=images,
        music=music,
        config=ProjectConfig(**meta.get("config", {})),
        created_at=meta.get("created_at", datetime.now(timezone.utc).isoformat()),
    )


@router.put("/{project_id}/config", response_model=ProjectConfig)
async def update_config(project_id: str, config: ProjectConfig):
    proj_dir = _project_dir(project_id)
    meta = _load_project_info(proj_dir)
    meta["config"] = config.model_dump()
    _save_project_meta(proj_dir, meta)
    return config
