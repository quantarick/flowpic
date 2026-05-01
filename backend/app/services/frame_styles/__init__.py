"""Frame styles package — pluggable decorative borders for frame fit mode."""

from .base import BaseFrameStyle, FrameLayout, get_frame_style, register_style  # noqa: F401

# Import styles to trigger registration
from . import film, clean, polaroid, shadow  # noqa: F401
