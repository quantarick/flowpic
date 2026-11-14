"""Art film: muted colour, faded blacks, teal shadows, cinema bars and fine grain."""

import numpy as np

from app.styles import effects as fx
from app.styles.base import VideoStyle, register


@register
class ArtFilm(VideoStyle):
    name = "art_film"
    label = {"en": "Art film", "zh": "文艺片"}
    description = {
        "en": "Muted colour, faded blacks and teal shadows, with 2.39:1 bars and fine grain.",
        "zh": "低饱和、灰调暗部与青色阴影，配 2.39:1 宽银幕遮幅和细腻颗粒。",
    }

    letterbox = 2.39
    vignette = 0.28
    grain = 5.0

    def grade(self, image: np.ndarray) -> np.ndarray:
        out = fx.saturation(image, 0.72)
        out = fx.tone_curve(out, contrast=0.92, lift=0.06, gain=0.97)
        return fx.split_tone(out, shadows=(-6, 2, 10), highlights=(8, 3, -8))
