"""Cinematic location title card and subtitle renderer."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

# Font search paths for subtitles (serif)
_FONT_SEARCH_PATHS = [
    Path(__file__).parent.parent / "assets" / "fonts" / "Playfair-Display-Bold.ttf",
    Path("C:/Windows/Fonts/georgiab.ttf"),
    Path("C:/Windows/Fonts/georgia.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Georgia Bold.ttf"),
]

# Mood-based font pools for title cards
# Each mood maps to a list of candidate fonts (first match wins)
_ASSETS = Path(__file__).parent.parent / "assets" / "fonts"

_MOOD_FONTS: dict[str, list[Path]] = {
    # Soft, romantic, peaceful (high valence, low arousal)
    "elegant": [
        _ASSETS / "GreatVibes-Regular.ttf",
        Path("C:/Windows/Fonts/Gabriola.ttf"),
        Path("C:/Windows/Fonts/SCRIPTBL.TTF"),       # Script MT Bold
        Path("C:/Windows/Fonts/KUNSTLER.TTF"),        # Kunstler Script
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Snell Roundhand.ttf"),
    ],
    # Dramatic, cinematic, melancholic (low valence)
    "cinematic": [
        _ASSETS / "Playfair-Display-Bold.ttf",
        Path("C:/Windows/Fonts/pala.ttf"),            # Palatino Linotype
        Path("C:/Windows/Fonts/georgiab.ttf"),        # Georgia Bold
        Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Georgia Bold.ttf"),
    ],
    # Energetic, upbeat, bold (high arousal, high valence)
    "bold": [
        Path("C:/Windows/Fonts/impact.ttf"),          # Impact
        Path("C:/Windows/Fonts/bahnschrift.ttf"),     # Bahnschrift
        Path("C:/Windows/Fonts/ariblk.ttf"),          # Arial Black
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Black.ttf"),
    ],
    # Clean, modern, neutral
    "modern": [
        Path("C:/Windows/Fonts/segoeuil.ttf"),        # Segoe UI Light
        Path("C:/Windows/Fonts/calibri.ttf"),          # Calibri
        Path("C:/Windows/Fonts/segoeui.ttf"),          # Segoe UI
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ],
}

_cached_font_path: Optional[Path] = None
_cached_mood_fonts: dict[str, Optional[str]] = {}


def _find_font() -> Optional[str]:
    """Find the best available subtitle font, caching the result."""
    global _cached_font_path
    if _cached_font_path is not None:
        return str(_cached_font_path) if _cached_font_path.exists() else None

    for path in _FONT_SEARCH_PATHS:
        if path.exists():
            _cached_font_path = path
            logger.info(f"Subtitle font: {path}")
            return str(path)

    logger.warning("No suitable subtitle font found, using PIL default")
    return None


def select_title_font(valence: float, arousal: float) -> Optional[str]:
    """Select a title card font based on music mood.

    Args:
        valence: 1-10 scale (low = sad/dark, high = happy/bright)
        arousal: 1-10 scale (low = calm, high = energetic)

    Returns:
        Font file path string, or None for PIL default.
    """
    # Classify mood into font style
    if valence >= 5.5 and arousal < 5.0:
        mood = "elegant"      # peaceful, romantic, warm
    elif valence < 5.0:
        mood = "cinematic"    # dramatic, melancholic, dark
    elif valence >= 5.5 and arousal >= 6.0:
        mood = "bold"         # energetic, joyful, intense
    else:
        mood = "modern"       # neutral, moderate

    # Check cache
    if mood in _cached_mood_fonts:
        return _cached_mood_fonts[mood]

    # Find first available font for this mood
    for path in _MOOD_FONTS[mood]:
        if path.exists():
            _cached_mood_fonts[mood] = str(path)
            logger.info(f"Title font for mood '{mood}' (v={valence:.1f}, a={arousal:.1f}): {path.name}")
            return str(path)

    # Fallback: try any available font from any mood
    for pool in _MOOD_FONTS.values():
        for path in pool:
            if path.exists():
                _cached_mood_fonts[mood] = str(path)
                logger.info(f"Title font fallback for mood '{mood}': {path.name}")
                return str(path)

    _cached_mood_fonts[mood] = None
    logger.warning(f"No title font found for mood '{mood}'")
    return None


def compute_font_size(frame_height: int) -> int:
    """Compute font size as 3.5% of frame height, clamped [24, 72]."""
    size = int(frame_height * 0.035)
    return max(24, min(72, size))


def compute_title_font_size(frame_height: int) -> int:
    """Compute title card font size as 7% of frame height, clamped [40, 128]."""
    size = int(frame_height * 0.07)
    return max(40, min(128, size))


def generate_title_card(
    canvas: np.ndarray,
    text: str,
    font_size: int,
    valence: float = 5.0,
    arousal: float = 5.0,
) -> np.ndarray:
    """Generate a cinematic title card image with blurred/darkened background and centered text.

    Args:
        canvas: RGB numpy array (H, W, 3), uint8 — the source image at output resolution.
        text: Location text to render (e.g. "Shibuya, Tokyo").
        font_size: Font size for the title text.
        valence: Music mood valence (1-10) for font selection.
        arousal: Music mood arousal (1-10) for font selection.

    Returns:
        RGB numpy array (H, W, 3), uint8 — the title card image.
    """
    h, w = canvas.shape[:2]

    # Heavy Gaussian blur using PIL (avoids OpenCV C++ exceptions)
    pil_img = Image.fromarray(canvas)
    blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=30))

    # Darken by multiplying with 0.35
    darkened = (np.array(blurred_pil).astype(np.float32) * 0.35).astype(np.uint8)

    # Select mood-appropriate font
    font_path = select_title_font(valence, arousal)
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Create RGBA overlay for text
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Center both axes
    x = (w - text_w) // 2
    y = (h - text_h) // 2

    # Drop shadow (3px offset for larger text)
    draw.text((x + 3, y + 3), text, font=font, fill=(0, 0, 0, 200))

    # Main text — white
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    # Composite text onto darkened background
    bg = Image.fromarray(darkened).convert("RGBA")
    composited = Image.alpha_composite(bg, overlay)
    return np.array(composited.convert("RGB"))


def compute_subtitle_alpha(
    t: float,
    duration: float,
    fade_in: float = 0.5,
    fade_out: float = 0.8,
) -> float:
    """Compute per-frame subtitle opacity with fade-in and fade-out.

    Returns alpha in [0.0, 1.0].
    """
    if t < 0 or t > duration:
        return 0.0
    if t < fade_in:
        return t / fade_in
    if t > duration - fade_out:
        return (duration - t) / fade_out
    return 1.0


def render_subtitle_overlay(
    frame: np.ndarray,
    text: str,
    alpha: float,
    font_size: int = 0,
    y_position: float = 0.85,
) -> np.ndarray:
    """Render a cinematic subtitle overlay on a video frame.

    Args:
        frame: RGB numpy array (H, W, 3), uint8
        text: Subtitle text to render
        alpha: Opacity [0.0, 1.0]
        font_size: Override font size (0 = auto from frame height)
        y_position: Vertical position as fraction of frame height (0.85 = lower third)

    Returns:
        New RGB numpy array with subtitle composited.
    """
    if alpha <= 0.0 or not text:
        return frame

    h, w = frame.shape[:2]
    if font_size <= 0:
        font_size = compute_font_size(h)

    # Load font
    font_path = _find_font()
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Create RGBA overlay for text
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Center horizontally, position at y_position vertically
    x = (w - text_w) // 2
    y = int(h * y_position) - text_h // 2

    shadow_offset = max(2, font_size // 20)
    shadow_alpha = int(180 * alpha)
    text_alpha = int(255 * alpha)

    # Draw drop shadow
    draw.text(
        (x + shadow_offset, y + shadow_offset),
        text,
        font=font,
        fill=(0, 0, 0, shadow_alpha),
    )

    # Draw main text
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255, text_alpha),
    )

    # Composite overlay onto frame
    frame_img = Image.fromarray(frame).convert("RGBA")
    composited = Image.alpha_composite(frame_img, overlay)
    return np.array(composited.convert("RGB"))
