"""Video style plugins: how pictures look in the rendered video.

A style is one class in one module of this package. It registers itself with
`@register`, and the package imports every module at startup, so adding a
style means adding a file. Nothing else needs to change: the API lists what
is registered and the frontend builds its dropdown from that list.

A style can shape a render in three ways:

- `grade(image)` runs once per picture and sets its colour and tone.
- `letterbox`, `vignette` and `grain` describe screen-fixed finishing that is
  applied to every output frame, so it stays put while the picture moves.
- `frame_finisher(...)` can be overridden for finishing those three do not cover.
"""

from __future__ import annotations

from abc import ABC
from typing import Callable, ClassVar

import numpy as np

# Takes a rendered frame and its time in seconds, returns the finished frame.
FrameFn = Callable[[np.ndarray, float], np.ndarray]

# Letterbox bars only make sense on a wide frame. Below this width/height
# ratio (4:3, square, portrait) a style's letterbox is ignored.
_MIN_ASPECT_FOR_LETTERBOX = 1.6


class VideoStyle(ABC):
    """Base class for a video style. Subclass it, set the fields, add @register."""

    name: ClassVar[str]                       # stable id stored in project config
    label: ClassVar[dict[str, str]]           # display name per language: {"en": ..., "zh": ...}
    description: ClassVar[dict[str, str]] = {}

    # Width / height of the picture area on wide outputs, e.g. 2.39 for
    # cinema bars. None fills the whole frame.
    letterbox: ClassVar[float | None] = None
    # Corner darkening, 0 (off) to 1 (corners black).
    vignette: ClassVar[float] = 0.0
    # Animated film grain as noise sigma in 8-bit levels. 0 is off.
    grain: ClassVar[float] = 0.0

    def grade(self, image: np.ndarray) -> np.ndarray:
        """Return the picture with this style's look.

        `image` is RGB uint8 with shape (H, W, 3). Return the same shape and
        dtype. Do not modify `image` in place.
        """
        return image

    def picture_size(self, out_w: int, out_h: int) -> tuple[int, int]:
        """Size of the area pictures are fitted into, inside an out_w x out_h frame."""
        aspect = out_w / out_h
        if self.letterbox and _MIN_ASPECT_FOR_LETTERBOX <= aspect < self.letterbox:
            pic_h = int(round(out_w / self.letterbox))
            return out_w, pic_h - pic_h % 2
        return out_w, out_h

    def frame_finisher(self, out_w: int, out_h: int, fps: int) -> FrameFn | None:
        """Build the per-frame finishing step for one render, or None for no work.

        The function receives frames of `picture_size` and must return frames
        of exactly (out_h, out_w, 3) without modifying its input.
        """
        pic_w, pic_h = self.picture_size(out_w, out_h)
        if (pic_w, pic_h) == (out_w, out_h) and not self.vignette and not self.grain:
            return None

        from app.styles.effects import FrameFinisher
        return FrameFinisher(
            out_w, out_h, pic_w, pic_h, fps,
            vignette=self.vignette, grain=self.grain,
        )

    def to_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "description": self.description}


# --- Registry ---

DEFAULT_STYLE = "natural"

_registry: dict[str, VideoStyle] = {}


def register(cls: type[VideoStyle]) -> type[VideoStyle]:
    """Class decorator that makes a style selectable."""
    name = getattr(cls, "name", "")
    if not name or not name.replace("_", "").isalnum() or name != name.lower():
        raise ValueError(f"{cls.__name__}.name must be lowercase letters, digits and underscores")
    if not getattr(cls, "label", None) or "en" not in cls.label:
        raise ValueError(f"{cls.__name__}.label needs at least an 'en' entry")
    if name in _registry and type(_registry[name]) is not cls:
        raise ValueError(f"Video style '{name}' is already registered")
    _registry[name] = cls()
    return cls


def get_style(name: str | None) -> VideoStyle:
    """Look up a style by name. None or an empty name means the default."""
    key = name or DEFAULT_STYLE
    try:
        return _registry[key]
    except KeyError:
        available = ", ".join(sorted(_registry))
        raise ValueError(f"Unknown video style '{name}'. Available: {available}") from None


def list_styles() -> list[VideoStyle]:
    """Every registered style, the default first, then by name."""
    return sorted(_registry.values(), key=lambda s: (s.name != DEFAULT_STYLE, s.name))
