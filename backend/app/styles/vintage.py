"""Vintage film: warm, faded colour with soft contrast."""

import numpy as np

from app.styles import effects as fx
from app.styles.base import VideoStyle, register


@register
class Vintage(VideoStyle):
    name = "vintage"
    label = {"en": "Vintage film", "zh": "复古胶片"}
    description = {
        "en": "Warm, faded colour with soft contrast, like an old colour print.",
        "zh": "偏暖褪色、柔和反差，像一张老彩色照片。",
    }

    vignette = 0.22
    grain = 7.0

    def grade(self, image: np.ndarray) -> np.ndarray:
        out = fx.saturation(image, 0.85)
        out = fx.tone_curve(out, contrast=0.90, lift=0.08, gain=0.96)
        return fx.split_tone(out, shadows=(10, 2, -8), highlights=(12, 6, -14))
