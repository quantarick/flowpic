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


@router.get("/crops/{project_id}")
async def list_crops(project_id: str):
    """List cropped preview images for a project."""
    crops_dir = settings.data_dir / project_id / "output" / "crops"
    if not crops_dir.exists():
        raise HTTPException(404, "No crops found")
    files = sorted(f.name for f in crops_dir.iterdir() if f.suffix.lower() in (".jpg", ".png"))
    return {"project_id": project_id, "crops": files}


@router.get("/crops/{project_id}/{filename}")
async def get_crop(project_id: str, filename: str):
    """Serve a single cropped image."""
    crop_path = settings.data_dir / project_id / "output" / "crops" / filename
    if not crop_path.exists():
        raise HTTPException(404, "Crop not found")
    media = "image/png" if crop_path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path=str(crop_path), media_type=media)
