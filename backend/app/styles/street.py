"""Street photo: high-contrast black and white with heavy grain."""

import numpy as np

from app.styles import effects as fx
from app.styles.base import VideoStyle, register


@register
class Street(VideoStyle):
    name = "street"
    label = {"en": "Street photo", "zh": "街头摄影"}
    description = {
        "en": "High-contrast black and white with heavy grain, like pushed 35mm film.",
        "zh": "高反差黑白与粗颗粒，像迫冲的 35mm 胶片。",
    }

    vignette = 0.35
    grain = 11.0

    def grade(self, image: np.ndarray) -> np.ndarray:
        # Extra red weight darkens skies, as a red filter does on black-and-white film.
        out = fx.monochrome(image, weights=(0.45, 0.45, 0.10))
        return fx.tone_curve(out, contrast=1.35, lift=0.02, gain=0.98)
