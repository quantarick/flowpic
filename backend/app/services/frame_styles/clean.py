"""Clean minimal border frame style.

White background with uniform ~6% padding and a thin 1px inner border line.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseFrameStyle, FrameLayout, register_style


@register_style("clean")
class CleanFrameStyle(BaseFrameStyle):

    PAD_FRAC = 0.06  # 6% uniform padding

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

        # White canvas
        canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)

        # Place photo
        canvas[layout.image_y:layout.image_y + layout.image_h,
               layout.image_x:layout.image_x + layout.image_w] = resized

        # Thin 1px border line around photo (2px gap)
        gap = 2
        x1 = layout.image_x - gap - 1
        y1 = layout.image_y - gap - 1
        x2 = layout.image_x + layout.image_w + gap
        y2 = layout.image_y + layout.image_h + gap
        border_color = (200, 200, 200)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 1)

        return canvas
