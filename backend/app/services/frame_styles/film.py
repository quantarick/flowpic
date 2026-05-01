"""Hasselblad analog film rebate frame style.

Adapted from Phantrace/hasselblad_frame.py — white background with thin frame
line, film edge ticks, camera text, date, and frame number.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .base import BaseFrameStyle, FrameLayout, register_style


@register_style("film")
class FilmFrameStyle(BaseFrameStyle):

    def compute_layout(
        self, img_w: int, img_h: int, target_w: int, target_h: int,
    ) -> FrameLayout:
        # Padding proportions (relative to inner photo area)
        pad_side_frac = 0.04
        pad_top_frac = 0.14
        pad_bottom_frac = 0.20

        # The photo area is target minus padding.
        # photo_w = target_w / (1 + 2*pad_side_frac)
        photo_w = target_w / (1 + 2 * pad_side_frac)
        photo_h = target_h / (1 + pad_top_frac + pad_bottom_frac)

        # Scale image to fit inside photo area (contain)
        scale = min(photo_w / img_w, photo_h / img_h)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)

        # Center photo in the photo area
        pad_side = int(target_w * pad_side_frac / (1 + 2 * pad_side_frac))
        pad_top = int(target_h * pad_top_frac / (1 + pad_top_frac + pad_bottom_frac))

        # Actual placement: center within available area
        avail_w = target_w - 2 * pad_side
        avail_h = target_h - pad_top - int(target_h * pad_bottom_frac / (1 + pad_top_frac + pad_bottom_frac))
        image_x = pad_side + (avail_w - scaled_w) // 2
        image_y = pad_top + (avail_h - scaled_h) // 2

        return FrameLayout(
            canvas_w=target_w,
            canvas_h=target_h,
            image_x=image_x,
            image_y=image_y,
            image_w=scaled_w,
            image_h=scaled_h,
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
        meta = metadata or {}
        frame_number = meta.get("frame_number", 16)
        camera_text = meta.get("camera_text", "HASSELBLAD 500C/M")
        date_str = meta.get("date_str", "")

        # Resize image to layout dimensions
        resized = cv2.resize(image, (layout.image_w, layout.image_h),
                             interpolation=cv2.INTER_LANCZOS4)

        # Convert to PIL for text rendering (BGR → RGB)
        pil_photo = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        # Create white canvas
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        canvas.paste(pil_photo, (layout.image_x, layout.image_y))

        draw = ImageDraw.Draw(canvas)
        ref = min(layout.image_w, layout.image_h)

        dark = (25, 25, 25)
        text_color = (50, 50, 50)
        light_gray = (180, 180, 180)

        # Frame line around photo
        gap = max(3, int(ref * 0.004))
        line_w = max(2, int(ref * 0.002))
        fx1 = layout.image_x - gap - line_w
        fy1 = layout.image_y - gap - line_w
        fx2 = layout.image_x + layout.image_w + gap + line_w
        fy2 = layout.image_y + layout.image_h + gap + line_w
        draw.rectangle([fx1, fy1, fx2, fy2], outline=dark, width=line_w)

        # Fonts
        font_size_lg = max(16, int(ref * 0.028))
        font_size_sm = max(12, int(ref * 0.020))
        try:
            font_lg = ImageFont.truetype("consola.ttf", font_size_lg)
            font_sm = ImageFont.truetype("consola.ttf", font_size_sm)
        except OSError:
            font_lg = ImageFont.load_default()
            font_sm = font_lg

        # Film edge marks
        dash_len = int(ref * 0.008)
        for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
            dx = int(fx1 + (fx2 - fx1) * frac)
            draw.line([(dx, fy1 - dash_len), (dx, fy1)], fill=light_gray, width=1)
            draw.line([(dx, fy2), (dx, fy2 + dash_len)], fill=light_gray, width=1)

        # Bottom text
        text_y = fy2 + int(ref * 0.035)
        draw.text((fx1 + int(ref * 0.005), text_y), camera_text, fill=text_color, font=font_lg)

        if date_str:
            date_bbox = draw.textbbox((0, 0), date_str, font=font_lg)
            date_w = date_bbox[2] - date_bbox[0]
            draw.text((fx2 - date_w - int(ref * 0.005), text_y), date_str, fill=text_color, font=font_lg)

        # Frame number
        sub_text = f"{frame_number}A"
        sub_bbox = draw.textbbox((0, 0), sub_text, font=font_sm)
        sub_w = sub_bbox[2] - sub_bbox[0]
        draw.text(
            (fx2 - sub_w - int(ref * 0.005), fy2 + int(ref * 0.006)),
            sub_text, fill=light_gray, font=font_sm,
        )

        # Convert back to BGR numpy
        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
