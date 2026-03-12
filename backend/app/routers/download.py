from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings

router = APIRouter()


@router.get("/download/{task_id}")
async def download_video(task_id: str):
    # Search for output video in all project directories
    for proj_dir in settings.data_dir.iterdir():
        if not proj_dir.is_dir():
            continue
        output = proj_dir / "output" / f"{task_id}.mp4"
        if output.exists():
            return FileResponse(
                path=str(output),
                media_type="video/mp4",
                filename=f"flowpic_{task_id}.mp4",
            )

    raise HTTPException(404, "Video not found")
