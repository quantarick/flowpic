"""Building blocks for video styles.

Colour helpers take and return RGB uint8 arrays of shape (H, W, 3) and never
modify their input. `FrameFinisher` does the screen-fixed work on every frame.
"""

from __future__ import annotations

import cv2
import numpy as np


def saturation(image: np.ndarray, amount: float) -> np.ndarray:
    """Scale colour intensity. 1 leaves the picture alone, 0 removes all colour."""
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)[..., None]
    out = grey + (image.astype(np.float32) - grey) * amount
    return np.clip(out, 0, 255).astype(np.uint8)


def monochrome(image: np.ndarray, weights: tuple[float, float, float] = (0.299, 0.587, 0.114)) -> np.ndarray:
    """Black and white. `weights` are the R, G, B contributions and should sum to 1.

    More red weight darkens blue skies, like a red filter on black-and-white film.
    """
    grey = image.astype(np.float32) @ np.asarray(weights, dtype=np.float32)
    grey8 = np.clip(grey, 0, 255).astype(np.uint8)
    return cv2.merge([grey8, grey8, grey8])


def tone_curve(image: np.ndarray, *, contrast: float = 1.0, lift: float = 0.0, gain: float = 1.0) -> np.ndarray:
    """Apply an S-curve, then set the black and white points.

    `contrast` is the slope at mid-grey: above 1 is punchier, below 1 is
    flatter. `lift` (0..1) is where pure black lands, `gain` where pure white
    lands. lift=0.06 gives the faded blacks of old film stock.
    """
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    low = 0.5 * np.power(2.0 * x, contrast)
    high = 1.0 - 0.5 * np.power(2.0 * (1.0 - x), contrast)
    y = np.where(x < 0.5, low, high)
    y = lift + y * (gain - lift)
    lut = np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)
    return cv2.LUT(image, lut)


def split_tone(
    image: np.ndarray,
    shadows: tuple[float, float, float],
    highlights: tuple[float, float, float],
) -> np.ndarray:
    """Tint dark and bright areas separately.

    Each tint is an (R, G, B) offset in 8-bit levels: (-6, 2, 10) pushes
    shadows toward teal, (8, 3, -8) warms the highlights.
    """
    lum = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    dark = ((1.0 - lum) ** 2)[..., None]
    bright = (lum ** 2)[..., None]
    out = (
        image.astype(np.float32)
        + dark * np.asarray(shadows, dtype=np.float32)
        + bright * np.asarray(highlights, dtype=np.float32)
    )
    return np.clip(out, 0, 255).astype(np.uint8)


class FrameFinisher:
    """Screen-fixed finishing for every output frame: vignette, grain, letterbox bars.

    Everything costly is built once here. Each call then does a few
    saturating OpenCV operations and returns a new array.
    """

    GRAIN_TILES = 4  # distinct noise patterns, cycled one per frame

    def __init__(
        self,
        out_w: int,
        out_h: int,
        pic_w: int,
        pic_h: int,
        fps: int,
        *,
        vignette: float = 0.0,
        grain: float = 0.0,
        seed: int = 7,
    ) -> None:
        self._pic_w, self._pic_h = pic_w, pic_h
        self._fps = max(1, fps)
        self._top = (out_h - pic_h) // 2
        self._bottom = out_h - pic_h - self._top
        self._side = (out_w - pic_w) // 2
        self._side_right = out_w - pic_w - self._side
        self._vignette = self._vignette_mask(pic_w, pic_h, vignette) if vignette > 0 else None
        self._grain = self._grain_tiles(pic_w, pic_h, grain, seed) if grain > 0 else []

    @staticmethod
    def _vignette_mask(w: int, h: int, strength: float) -> np.ndarray:
        ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        radius = np.sqrt((xs ** 2 + ys ** 2) / 2.0)  # 0 at the centre, 1 in the corners
        falloff = np.clip((radius - 0.45) / 0.55, 0.0, 1.0)
        falloff = falloff * falloff * (3.0 - 2.0 * falloff)
        mask = np.round((1.0 - min(strength, 1.0) * falloff) * 255.0).astype(np.uint8)
        return cv2.merge([mask, mask, mask])

    # Grain is drawn as if the frame were this many pixels on its shorter side,
    # then scaled up, so a grain is the same size on screen at every quality
    # setting. Per-pixel noise at 4K would be too fine to see. Note this does
    # not make files smaller: grain costs the encoder about the same either way.
    GRAIN_REFERENCE_SIDE = 720

    def _grain_tiles(self, w: int, h: int, sigma: float, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(seed)
        cell = max(1.0, min(w, h) / self.GRAIN_REFERENCE_SIDE)
        grain_w, grain_h = max(1, round(w / cell)), max(1, round(h / cell))
        tiles = []
        for _ in range(self.GRAIN_TILES):
            noise = rng.normal(0.0, sigma, (grain_h, grain_w)).astype(np.float32)
            if (grain_w, grain_h) != (w, h):
                noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
            # Stored as two unsigned planes so a frame needs only add + subtract.
            brighter = np.clip(noise, 0, 255).astype(np.uint8)
            darker = np.clip(-noise, 0, 255).astype(np.uint8)
            tiles.append((brighter, darker))
        return tiles

    def __call__(self, frame: np.ndarray, t: float) -> np.ndarray:
        if frame.shape[0] != self._pic_h or frame.shape[1] != self._pic_w:
            frame = cv2.resize(frame, (self._pic_w, self._pic_h), interpolation=cv2.INTER_AREA)

        out = frame
        if self._vignette is not None:
            out = cv2.multiply(out, self._vignette, scale=1.0 / 255.0)
        if self._grain:
            brighter, darker = self._grain[int(t * self._fps) % len(self._grain)]
            out = cv2.add(out, cv2.merge([brighter, brighter, brighter]))
            out = cv2.subtract(out, cv2.merge([darker, darker, darker]))
        if self._top or self._bottom or self._side or self._side_right:
            out = cv2.copyMakeBorder(
                out, self._top, self._bottom, self._side, self._side_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
        return out if out is not frame else frame.copy()
