import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.config import settings
from app.models import MusicUploadResponse, UploadResponse

router = APIRouter()


def _project_dir(project_id: str) -> Path:
    p = settings.data_dir / project_id
    if not p.exists():
        raise HTTPException(404, "Project not found")
    return p


@router.post("/images", response_model=UploadResponse)
async def upload_images(
    project_id: str = Query(...),
    files: list[UploadFile] = File(...),
):
    proj_dir = _project_dir(project_id)
    images_dir = proj_dir / "images"
    images_dir.mkdir(exist_ok=True)

    existing = list(images_dir.glob("*"))
    if len(existing) + len(files) > settings.max_images:
        raise HTTPException(
            400,
            f"Too many images. Max {settings.max_images}, "
            f"already have {len(existing)}, tried to add {len(files)}",
        )

    saved: list[str] = []
    for f in files:
        if f.content_type not in settings.allowed_image_types:
            raise HTTPException(400, f"Unsupported image type: {f.content_type}")
        if f.size and f.size > settings.max_image_size_mb * 1024 * 1024:
            raise HTTPException(400, f"Image too large: {f.filename}")

        ext = Path(f.filename or "image.jpg").suffix
        name = f"{uuid.uuid4().hex[:12]}{ext}"
        dest = images_dir / name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(name)

    return UploadResponse(filenames=saved, count=len(saved))


@router.post("/music", response_model=MusicUploadResponse)
async def upload_music(
    project_id: str = Query(...),
    file: UploadFile = File(...),
):
    proj_dir = _project_dir(project_id)

    allowed_extensions = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".mp4", ".ogg", ".wma"}
    ext = Path(file.filename or "audio.mp3").suffix.lower()
    if file.content_type not in settings.allowed_audio_types and ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported audio type: {file.content_type} ({ext})")
    if file.size and file.size > settings.max_music_size_mb * 1024 * 1024:
        raise HTTPException(400, "Audio file too large")

    ext = Path(file.filename or "audio.mp3").suffix
    name = f"music{ext}"
    dest = proj_dir / name

    # Remove any existing music file
    for old in proj_dir.glob("music.*"):
        old.unlink()

    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # Optionally get duration via librosa (lazy)
    duration = None
    try:
        import librosa
        y, sr = librosa.load(str(dest), sr=None, duration=settings.max_music_duration_sec + 1)
        duration = len(y) / sr
        if duration > settings.max_music_duration_sec:
            dest.unlink()
            raise HTTPException(400, f"Audio too long: {duration:.0f}s (max {settings.max_music_duration_sec}s)")
    except HTTPException:
        raise
    except Exception:
        pass  # Duration check is best-effort

    return MusicUploadResponse(filename=name, duration_seconds=duration)
