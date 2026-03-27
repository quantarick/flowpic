import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import settings
from app.models import ProjectConfig

router = APIRouter()


class RegenerateCropRequest(BaseModel):
    crop_filename: str  # e.g. "003_47baa1edc3a1_crop.png"
    feedback: str       # user's correction text


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


@router.get("/images/{project_id}/{stem}")
async def get_original_image(project_id: str, stem: str):
    """Serve an original source image by filename stem."""
    images_dir = settings.data_dir / project_id / "images"
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            media = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp",
            }[ext]
            return FileResponse(path=str(candidate), media_type=media)
    raise HTTPException(404, "Original image not found")


@router.post("/crops/{project_id}/regenerate")
async def regenerate_crop(project_id: str, req: RegenerateCropRequest):
    """Re-caption a single image with user feedback and regenerate its crop."""
    proj_dir = settings.data_dir / project_id
    images_dir = proj_dir / "images"
    crops_dir = proj_dir / "output" / "crops"

    if not proj_dir.exists():
        raise HTTPException(404, "Project not found")

    # Parse image hash/stem from crop filename: "003_<stem>_<mode>.png"
    parts = req.crop_filename.rsplit(".", 1)[0]  # strip extension
    tokens = parts.split("_", 1)  # split off the index prefix
    if len(tokens) < 2:
        raise HTTPException(400, "Invalid crop filename format")
    rest = tokens[1]  # "<stem>_<mode>"
    # The mode is the last token after the last underscore
    last_underscore = rest.rfind("_")
    if last_underscore < 0:
        raise HTTPException(400, "Invalid crop filename format")
    image_stem = rest[:last_underscore]

    # Find source image by stem
    source_image = None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = images_dir / f"{image_stem}{ext}"
        if candidate.exists():
            source_image = candidate
            break
    if source_image is None:
        raise HTTPException(404, f"Source image not found for stem '{image_stem}'")

    # Delete old caption cache
    cache_file = proj_dir / "captions" / f"{image_stem}.json"
    if cache_file.exists():
        cache_file.unlink()

    # Delete old crop file(s) matching this stem
    if crops_dir.exists():
        for old_crop in crops_dir.glob(f"*_{image_stem}_*.png"):
            old_crop.unlink()

    # Load project config for vision model
    meta_path = proj_dir / "meta.json"
    config = ProjectConfig()
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        config = ProjectConfig(**meta.get("config", {}))

    # Re-caption with feedback
    from app.services.image_captioner import ImageCaptioner
    captioner = ImageCaptioner(model=config.vision_model)
    caption = captioner.caption_image(source_image, feedback=req.feedback)

    # Re-crop
    import cv2
    from app.services.smart_crop import smart_fit, get_output_resolution
    out_w, out_h = get_output_resolution(config.aspect_ratio, config.quality)

    img = cv2.imread(str(source_image))  # cv2 imported above
    if img is None:
        raise HTTPException(500, "Failed to read source image")

    subject_box = None
    if caption.subject_x1 is not None:
        subject_box = (caption.subject_x1, caption.subject_y1,
                       caption.subject_x2, caption.subject_y2)

    result = smart_fit(
        img, out_w, out_h,
        face_regions=caption.face_regions,
        focus_x=caption.focus_x, focus_y=caption.focus_y,
        scale_factor=1.0, fit_mode=caption.fit_mode,
        subject_box=subject_box,
        horizon_y=caption.horizon_y,
        people_centers=caption.people_centers,
    )

    # Determine index prefix from original crop filename
    idx_prefix = tokens[0]  # e.g. "003"
    new_crop_name = f"{idx_prefix}_{image_stem}_{caption.fit_mode}.png"
    crops_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crops_dir / new_crop_name), result.canvas)

    return {"crop_filename": new_crop_name}
