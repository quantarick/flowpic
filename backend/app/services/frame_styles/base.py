"""Base class and registry for frame styles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameLayout:
    """Describes how the photo sits inside the framed canvas."""
    canvas_w: int
    canvas_h: int
    image_x: int       # photo left edge on canvas
    image_y: int       # photo top edge on canvas
    image_w: int       # photo width on canvas
    image_h: int       # photo height on canvas
    scale: float        # scale applied to original image


class BaseFrameStyle(ABC):
    """Abstract base for all frame styles."""

    @abstractmethod
    def compute_layout(
        self,
        img_w: int,
        img_h: int,
        target_w: int,
        target_h: int,
    ) -> FrameLayout:
        """Compute how the image fits inside the target canvas with frame decoration."""

    @abstractmethod
    def render(
        self,
        image: np.ndarray,
        target_w: int,
        target_h: int,
        layout: FrameLayout,
        metadata: dict | None = None,
    ) -> np.ndarray:
        """Render the framed image onto a canvas of exactly (target_w, target_h)."""


# --- Registry ---

_registry: dict[str, type[BaseFrameStyle]] = {}


def register_style(name: str):
    """Decorator to register a frame style class."""
    def decorator(cls: type[BaseFrameStyle]):
        _registry[name] = cls
        return cls
    return decorator


def get_frame_style(name: str) -> BaseFrameStyle:
    """Instantiate a registered frame style by name."""
    if name not in _registry:
        raise ValueError(f"Unknown frame style: {name!r}. Available: {list(_registry)}")
    return _registry[name]()
