"""Drop shadow on dark background frame style.

Dark background (30,30,30), ~8% uniform padding, soft gaussian blur shadow
beneath the image.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseFrameStyle, FrameLayout, register_style


@register_style("shadow")
class ShadowFrameStyle(BaseFrameStyle):

    PAD_FRAC = 0.08
    BG_COLOR = (30, 30, 30)
    SHADOW_OFFSET = 6
    SHADOW_BLUR = 21  # must be odd

    def compute_layout(
        self, img_w: int, img_h: int, target_w: int, target_h: int,
    ) -> FrameLayout:
        pad_x = int(target_w * self.PAD_FRAC)
        pad_y = int(target_h * self.PAD_FRAC)
        avail_w = target_w - 2 * pad_x
        avail_h = target_h - 2 * pad_y

        scale = min(avail_w / img_w, avail_h / img_h)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)

        image_x = pad_x + (avail_w - scaled_w) // 2
        image_y = pad_y + (avail_h - scaled_h) // 2

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

        # Dark canvas
        canvas = np.full((target_h, target_w, 3), self.BG_COLOR, dtype=np.uint8)

        # Draw shadow: a white rectangle shifted down-right, then blur, then darken
        shadow_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        sx = layout.image_x + self.SHADOW_OFFSET
        sy = layout.image_y + self.SHADOW_OFFSET
        ex = min(sx + layout.image_w, target_w)
        ey = min(sy + layout.image_h, target_h)
        shadow_mask[sy:ey, sx:ex] = 180

        blur_size = self.SHADOW_BLUR | 1  # ensure odd
        shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_size, blur_size), 0)

        # Apply shadow: darken canvas where shadow mask is > 0
        shadow_3ch = shadow_mask[:, :, np.newaxis].astype(np.float32) / 255.0
        canvas = (canvas.astype(np.float32) * (1.0 - shadow_3ch * 0.6)).astype(np.uint8)

        # Place photo on top
        canvas[layout.image_y:layout.image_y + layout.image_h,
               layout.image_x:layout.image_x + layout.image_w] = resized

        return canvas
