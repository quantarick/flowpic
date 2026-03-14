"""Ollama model discovery — lists available vision models for the frontend dropdown."""

import logging

import httpx
from fastapi import APIRouter

from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Models known to support vision (image input).
# Ollama doesn't expose a "capabilities" field, so we maintain a allowlist.
VISION_CAPABLE = {
    "moondream",
    "llava",
    "llava-phi3",
    "llava-llama3",
    "bakllava",
    "gemma3",
    "qwen2.5vl",
    "qwen2-vl",
    "qwen3-vl",
    "minicpm-v",
    "llama3.2-vision",
    "granite3.2-vision",
    "deepseek-ocr",
    "smolvlm",
}


def _is_vision_model(name: str) -> bool:
    """Check if a model name (possibly with tag) is vision-capable."""
    base = name.split(":")[0].lower()
    return base in VISION_CAPABLE


@router.get("/models")
async def list_vision_models():
    """Return Ollama models that support image input, plus the server default."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            all_models = resp.json().get("models", [])
    except Exception as e:
        logger.warning(f"Failed to query Ollama models: {e}")
        # Return just the configured default so the UI still works
        return {
            "default": settings.ollama_model,
            "models": [{"name": settings.ollama_model, "size": None}],
        }

    vision_models = []
    for m in all_models:
        name = m.get("name", "")
        if _is_vision_model(name):
            vision_models.append({
                "name": name,
                "size": m.get("size"),
                "parameter_size": m.get("details", {}).get("parameter_size"),
            })

    # Ensure the server default is always listed even if not currently pulled
    default_base = settings.ollama_model.split(":")[0]
    default_present = any(
        m["name"].split(":")[0] == default_base
        for m in vision_models
    )
    if not default_present:
        vision_models.insert(0, {
            "name": settings.ollama_model,
            "size": None,
            "parameter_size": None,
        })

    return {
        "default": settings.ollama_model,
        "models": vision_models,
    }
