"""LLM discovery: which providers and models the frontend may offer."""

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.llm import LLMNotConfigured, vision_catalog

router = APIRouter()


@router.get("/models")
async def list_vision_models():
    """Vision-capable providers, their models, and the server default.

    The response carries provider names, model names and a `configured` flag.
    It never carries an API key.
    """
    try:
        return await run_in_threadpool(vision_catalog)
    except LLMNotConfigured as e:
        raise HTTPException(500, str(e))
