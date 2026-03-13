import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect

from app.config import settings
from app.dependencies import get_task_manager
from app.models import GenerateResponse, ProjectConfig, TaskRecord, TaskStatus
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

    # Persist task record
    from app.services.task_db import save_task
    save_task(TaskRecord(
        task_id=task_id,
        project_id=project_id,
        image_count=len(images),
        config=config,
    ))

    from app.workers.video_worker import run_pipeline
    tm.submit(
        task_id=task_id,
        fn=run_pipeline,
        project_id=project_id,
        task_id_arg=task_id,
        config=config,
    )

    return GenerateResponse(task_id=task_id)


@router.get("/tasks")
async def list_tasks(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    """List task history, most recent first."""
    from app.services.task_db import list_tasks as db_list
    tasks = db_list(limit=limit, offset=offset)
    return [t.model_dump() for t in tasks]


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get a single task by ID."""
    from app.services.task_db import get_task as db_get
    task = db_get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task.model_dump()


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    tm: TaskManager = Depends(get_task_manager),
):
    """Cancel a running or stale task."""
    from app.services.task_db import get_task as db_get, update_task_status
    task = db_get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED):
        raise HTTPException(400, "Task already finished")
    # Try to cancel via TaskManager (works for queued/running tasks)
    tm.cancel(task_id)
    # Also directly update DB for stale tasks (e.g. pending from crashed server)
    if tm.get_status(task_id) is None:
        update_task_status(task_id, TaskStatus.CANCELLED,
                           error_message="Cancelled (stale task)")
    return {"status": "cancelling"}


@router.post("/tasks/{task_id}/retry", response_model=GenerateResponse)
async def retry_task(
    task_id: str,
    tm: TaskManager = Depends(get_task_manager),
):
    """Retry a failed/cancelled task by re-submitting the same project."""
    from app.services.task_db import get_task as db_get
    old_task = db_get(task_id)
    if not old_task:
        raise HTTPException(404, "Task not found")
    if old_task.status not in (TaskStatus.FAILED, TaskStatus.CANCELLED):
        raise HTTPException(400, "Only failed or cancelled tasks can be retried")

    proj_dir = settings.data_dir / old_task.project_id
    if not proj_dir.exists():
        raise HTTPException(404, "Project not found (files may have been cleaned up)")

    images_dir = proj_dir / "images"
    images = sorted(images_dir.glob("*")) if images_dir.exists() else []
    if len(images) < 2:
        raise HTTPException(400, "Need at least 2 images (project files may have been cleaned up)")

    music_files = list(proj_dir.glob("music.*"))
    if not music_files:
        raise HTTPException(400, "No music file (project files may have been cleaned up)")

    config = old_task.config or ProjectConfig()
    new_task_id = uuid.uuid4().hex[:16]

    from app.services.task_db import save_task
    save_task(TaskRecord(
        task_id=new_task_id,
        project_id=old_task.project_id,
        image_count=len(images),
        config=config,
    ))

    from app.workers.video_worker import run_pipeline
    tm.submit(
        task_id=new_task_id,
        fn=run_pipeline,
        project_id=old_task.project_id,
        task_id_arg=new_task_id,
        config=config,
    )

    return GenerateResponse(task_id=new_task_id)


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
