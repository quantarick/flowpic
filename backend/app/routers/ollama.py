"""Ollama model discovery. Kept for older clients; the frontend uses /api/llm/models."""

import logging

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.llm import LLMError
from app.llm.ollama import OllamaProvider

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/models")
async def list_vision_models():
    """Return Ollama models that support image input, plus the Ollama default."""
    default = settings.ollama_model
    try:
        models = [m.to_dict() for m in await run_in_threadpool(OllamaProvider().list_models)]
    except LLMError as e:
        logger.warning(f"Failed to query Ollama models: {e}")
        # Return just the configured default so the UI still works
        return {"default": default, "models": [{"name": default, "size": None}]}

    # Ensure the default is always listed even if not currently pulled
    default_base = default.split(":")[0]
    if not any(m["name"].split(":")[0] == default_base for m in models):
        models.insert(0, {"name": default, "size": None, "parameter_size": None})

    return {"default": default, "models": models}
