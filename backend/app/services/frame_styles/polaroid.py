"""Polaroid / instant film frame style.

White background, ~4% sides/top, ~18% thick bottom area (write-on strip).
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseFrameStyle, FrameLayout, register_style


@register_style("polaroid")
class PolaroidFrameStyle(BaseFrameStyle):

    def compute_layout(
        self, img_w: int, img_h: int, target_w: int, target_h: int,
    ) -> FrameLayout:
        pad_side_frac = 0.04
        pad_top_frac = 0.04
        pad_bottom_frac = 0.18

        pad_side = int(target_w * pad_side_frac)
        pad_top = int(target_h * pad_top_frac)
        pad_bottom = int(target_h * pad_bottom_frac)

        avail_w = target_w - 2 * pad_side
        avail_h = target_h - pad_top - pad_bottom

        scale = min(avail_w / img_w, avail_h / img_h)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)

        image_x = pad_side + (avail_w - scaled_w) // 2
        image_y = pad_top + (avail_h - scaled_h) // 2

        return FrameLayout(
            canvas_w=target_w, canvas_h=target_h,
            image_x=image_x, image_y=image_y,
            image_w=scaled_w, image_h=scaled_h,
            scale=scale,
        )

    def render(
        self,
        image: np.ndarray,
        target_w: int,
        target_h: int,
        layout: FrameLayout,
        metadata: dict | None = None,
    ) -> np.ndarray:
        resized = cv2.resize(image, (layout.image_w, layout.image_h),
                             interpolation=cv2.INTER_LANCZOS4)

        # White canvas
        canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)

        # Place photo
        canvas[layout.image_y:layout.image_y + layout.image_h,
               layout.image_x:layout.image_x + layout.image_w] = resized

        # Subtle shadow beneath the photo (thin dark line at bottom/right)
        shadow_color = (220, 220, 220)
        # Bottom shadow
        sy = layout.image_y + layout.image_h
        if sy + 2 < target_h:
            canvas[sy:sy + 2, layout.image_x + 2:layout.image_x + layout.image_w + 2] = shadow_color
        # Right shadow
        sx = layout.image_x + layout.image_w
        if sx + 2 < target_w:
            canvas[layout.image_y + 2:layout.image_y + layout.image_h + 2, sx:sx + 2] = shadow_color

        return canvas
