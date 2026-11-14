"""Video styles for FlowPic.

    from app.styles import get_style

    style = get_style(config.video_style)
    canvas = style.grade(canvas)

To add a style, add a module to this package with a `VideoStyle` subclass
decorated with `@register`. See `art_film.py` for a complete example.
"""

import importlib
import pkgutil

from app.styles.base import (
    DEFAULT_STYLE,
    FrameFn,
    VideoStyle,
    get_style,
    list_styles,
    register,
)

# Modules in this package that are not styles.
_NOT_STYLES = {"base", "effects"}


def _discover() -> None:
    """Import every style module so its @register decorator runs."""
    for module in pkgutil.iter_modules(__path__):
        if module.name in _NOT_STYLES or module.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{module.name}")


_discover()

__all__ = ["DEFAULT_STYLE", "FrameFn", "VideoStyle", "get_style", "list_styles", "register"]
