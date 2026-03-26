"""Copywriting generation endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.config import settings
from app.models import CopywritingResult
from app.services.copywriting_generator import generate_copywriting, get_cached_copywriting

router = APIRouter()


class CopywritingRequest(BaseModel):
    hint: Optional[str] = ""


@router.post("/copywriting/{project_id}", response_model=CopywritingResult)
async def create_copywriting(project_id: str, body: CopywritingRequest = None):
    proj_dir = settings.data_dir / project_id
    if not proj_dir.exists():
        raise HTTPException(404, "Project not found")
    hint = (body.hint or "") if body else ""
    import logging
    logging.getLogger(__name__).info("Copywriting hint: %r", hint)
    try:
        return generate_copywriting(proj_dir, hint=hint)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Copywriting generation failed: {e}")


@router.get("/copywriting/{project_id}", response_model=CopywritingResult)
async def read_copywriting(project_id: str):
    proj_dir = settings.data_dir / project_id
    if not proj_dir.exists():
        raise HTTPException(404, "Project not found")
    result = get_cached_copywriting(proj_dir)
    if result is None:
        raise HTTPException(404, "No copywriting generated yet")
    return result
