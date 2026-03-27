"""XHS (Xiaohongshu) publishing endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.models import XhsCookieStatus, XhsPublishRequest, XhsPublishResult, XhsStyleProfile
from app.services.xhs_publisher import (
    clear_cookies,
    publish_note,
    save_cookies,
    validate_cookies,
)

router = APIRouter()


class CookieBody(BaseModel):
    cookie: str


@router.post("/xhs/cookies", response_model=XhsCookieStatus)
async def post_cookies(body: CookieBody):
    return save_cookies(body.cookie)


@router.get("/xhs/cookies", response_model=XhsCookieStatus)
async def get_cookies():
    return validate_cookies()


@router.delete("/xhs/cookies")
async def delete_cookies():
    clear_cookies()
    return {"ok": True}


# --- Style Profile (must be before /xhs/publish/{project_id} to avoid catch-all) ---

@router.post("/xhs/style-profile", response_model=XhsStyleProfile)
async def scan_style_profile(force: bool = False):
    from app.services.xhs_scraper import scrape_user_style
    try:
        return scrape_user_style(force=force)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Style scan failed: {e}")


@router.get("/xhs/style-profile", response_model=XhsStyleProfile)
async def get_style_profile():
    from app.services.xhs_scraper import get_cached_style_profile
    profile = get_cached_style_profile()
    if not profile:
        raise HTTPException(404, "No style profile cached. Run a scan first.")
    return profile


@router.delete("/xhs/style-profile")
async def delete_style_profile():
    from app.services.xhs_scraper import clear_style_profile
    clear_style_profile()
    return {"ok": True}


# --- Published Images ---

@router.get("/xhs/published-images/{project_id}")
async def get_published_images(project_id: str):
    from app.services.task_db import get_published_images_for_project
    proj_dir = settings.data_dir / project_id
    if not proj_dir.exists():
        raise HTTPException(404, "Project not found")
    images = get_published_images_for_project(project_id)
    return {"project_id": project_id, "published_images": images}


# --- Publish ---

@router.post("/xhs/publish/{project_id}", response_model=XhsPublishResult)
async def xhs_publish(project_id: str, body: XhsPublishRequest):
    import logging
    logging.getLogger(__name__).info("XHS publish request: %d images: %s", len(body.image_filenames), body.image_filenames[:3])
    proj_dir = settings.data_dir / project_id
    if not proj_dir.exists():
        raise HTTPException(404, "Project not found")
    return publish_note(
        project_dir=proj_dir,
        title=body.title,
        description=body.description,
        hashtags=body.hashtags,
        image_filenames=body.image_filenames,
        project_id=project_id,
    )
