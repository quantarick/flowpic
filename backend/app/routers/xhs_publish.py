"""XHS (Xiaohongshu) publishing endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.models import XhsCookieStatus, XhsPublishRequest, XhsPublishResult
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
    )
