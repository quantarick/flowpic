import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from app.config import settings
from app.dependencies import get_task_manager
from app.models import GenerateResponse, ProjectConfig, TaskStatus
from app.services.task_manager import TaskManager

router = APIRouter()
ws_router = APIRouter()


@router.post("/generate/{project_id}", response_model=GenerateResponse)
async def generate_video(
    project_id: str,
    tm: TaskManager = Depends(get_task_manager),
):
    proj_dir = settings.data_dir / project_id
    if not proj_dir.exists():
        raise HTTPException(404, "Project not found")

    images_dir = proj_dir / "images"
    images = sorted(images_dir.glob("*")) if images_dir.exists() else []
    if len(images) < 2:
        raise HTTPException(400, "Need at least 2 images")

    music_files = list(proj_dir.glob("music.*"))
    if not music_files:
        raise HTTPException(400, "No music file uploaded")

    meta_path = proj_dir / "meta.json"
    config = ProjectConfig()
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        config = ProjectConfig(**meta.get("config", {}))

    task_id = uuid.uuid4().hex[:16]

    from app.workers.video_worker import run_pipeline
    tm.submit(
        task_id=task_id,
        fn=run_pipeline,
        project_id=project_id,
        task_id_arg=task_id,
        config=config,
    )

    return GenerateResponse(task_id=task_id)


@ws_router.websocket("/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    tm = get_task_manager()
    await websocket.accept()

    try:
        async for msg in tm.subscribe(task_id):
            await websocket.send_json(msg)
            if msg.get("status") in ("done", "failed", "cancelled"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        tm.unsubscribe(task_id)
