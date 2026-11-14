"""Vertical output sizes and title cards on narrow frames."""

from __future__ import annotations

import numpy as np
import pytest

from app.models import AspectRatio, Quality
from app.services.smart_crop import RESOLUTIONS, get_output_resolution
from app.services.subtitle_renderer import compute_title_font_size, generate_title_card

# H.264 level 5.2 allows 36,864 macroblocks of 16x16 pixels per frame.
LEVEL_5_2_PIXELS = 36_864 * 256
NVENC_MAX_SIDE = 4096

VERTICAL = {
    AspectRatio.RATIO_9_16: 9 / 16,
    AspectRatio.RATIO_9_19_5: 9 / 19.5,
    AspectRatio.RATIO_3_4: 3 / 4,
}


def test_every_ratio_has_every_quality():
    for aspect in AspectRatio:
        for quality in Quality:
            assert (aspect.value, quality.value) in RESOLUTIONS, (aspect.value, quality.value)


@pytest.mark.parametrize("aspect", list(VERTICAL))
@pytest.mark.parametrize("quality", list(Quality))
def test_vertical_sizes_are_playable_on_a_phone(aspect, quality):
    w, h = get_output_resolution(aspect, quality)

    assert h > w
    assert w % 2 == 0 and h % 2 == 0                        # H.264 with 4:2:0 needs even sides
    assert abs(w / h - VERTICAL[aspect]) / VERTICAL[aspect] < 0.005
    assert w * h <= LEVEL_5_2_PIXELS                        # within what phones decode
    assert max(w, h) <= NVENC_MAX_SIDE                      # the GPU encoder can still be used


def test_iphone_sizes():
    sizes = [get_output_resolution(AspectRatio.RATIO_9_19_5, q) for q in Quality]
    assert sizes == [(720, 1560), (1080, 2340), (1440, 3120), (1772, 3840)]


@pytest.mark.parametrize("size", [(720, 1560), (1080, 2340), (720, 960), (1280, 720)])
def test_title_text_stays_inside_the_frame(size):
    w, h = size
    canvas = np.full((h, w, 3), 90, np.uint8)

    card = generate_title_card(canvas, "Te Whanganui-a-Tara, Wellington", compute_title_font_size(h))

    assert card.shape == (h, w, 3)
    # The background is darkened to at most 35% brightness; the text is white.
    text_columns = np.where((card > 200).all(axis=2).any(axis=0))[0]
    assert text_columns.size > 0
    margin = int(w * 0.05)
    assert text_columns.min() >= margin and text_columns.max() <= w - margin
