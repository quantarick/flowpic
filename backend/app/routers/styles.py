"""Video style discovery: the style plugins the frontend may offer."""

from fastapi import APIRouter

from app.styles import DEFAULT_STYLE, list_styles

router = APIRouter()


@router.get("/styles")
async def list_video_styles():
    """Every registered video style, with its labels and descriptions per language."""
    return {"default": DEFAULT_STYLE, "styles": [style.to_dict() for style in list_styles()]}
