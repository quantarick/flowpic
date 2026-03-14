import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

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
