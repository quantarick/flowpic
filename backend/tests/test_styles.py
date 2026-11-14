"""Video style plugins: registry, grading, and per-frame finishing."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models import ProjectConfig
from app.styles import DEFAULT_STYLE, VideoStyle, get_style, list_styles, register
from app.styles import base as styles_base
from app.styles import effects as fx


@pytest.fixture
def picture():
    """A colourful gradient picture, RGB uint8."""
    h, w = 90, 160
    ys, xs = np.mgrid[0:h, 0:w]
    return np.dstack([xs * 255 // (w - 1), ys * 255 // (h - 1), np.full((h, w), 128)]).astype(np.uint8)


# --- Registry ---

def test_built_in_styles_are_discovered():
    names = [style.name for style in list_styles()]
    assert names[0] == DEFAULT_STYLE == "natural"
    assert {"natural", "art_film", "street", "vintage"} <= set(names)


def test_every_style_has_labels_in_both_languages():
    for style in list_styles():
        assert style.label.get("en") and style.label.get("zh"), style.name
        assert style.description.get("en") and style.description.get("zh"), style.name


def test_unknown_style_lists_the_choices():
    with pytest.raises(ValueError, match="Unknown video style 'noir'.*art_film"):
        get_style("noir")


@pytest.mark.parametrize("name", [None, ""])
def test_missing_name_means_the_default(name):
    assert get_style(name).name == "natural"


def test_project_config_defaults_to_natural():
    assert ProjectConfig().video_style == "natural"
    assert ProjectConfig(**{"aspect_ratio": "16:9"}).video_style == "natural"  # an old saved project


def test_a_new_style_is_one_decorated_class(monkeypatch):
    monkeypatch.setattr(styles_base, "_registry", dict(styles_base._registry))

    @register
    class Sepia(VideoStyle):
        name = "sepia_test"
        label = {"en": "Sepia", "zh": "棕褐"}

        def grade(self, image):
            return fx.split_tone(fx.monochrome(image), shadows=(12, 4, -10), highlights=(14, 6, -12))

    assert get_style("sepia_test").to_dict()["label"]["en"] == "Sepia"
    assert "sepia_test" in [style.name for style in list_styles()]


@pytest.mark.parametrize(
    "fields",
    [
        {"name": "Bad Name", "label": {"en": "x"}},
        {"name": "ok_name", "label": {"zh": "只有中文"}},
    ],
)
def test_register_rejects_malformed_styles(monkeypatch, fields):
    monkeypatch.setattr(styles_base, "_registry", dict(styles_base._registry))
    with pytest.raises(ValueError):
        register(type("Broken", (VideoStyle,), fields))


def test_register_rejects_a_duplicate_name(monkeypatch):
    monkeypatch.setattr(styles_base, "_registry", dict(styles_base._registry))
    with pytest.raises(ValueError, match="already registered"):
        register(type("Impostor", (VideoStyle,), {"name": "street", "label": {"en": "x"}}))


# --- Natural is a strict no-op ---

def test_natural_changes_nothing(picture):
    natural = get_style("natural")
    assert natural.grade(picture) is picture
    assert natural.picture_size(1280, 720) == (1280, 720)
    assert natural.frame_finisher(1280, 720, 30) is None


# --- Grading ---

@pytest.mark.parametrize("name", ["art_film", "street", "vintage"])
def test_grade_keeps_shape_and_leaves_input_alone(picture, name):
    before = picture.copy()
    graded = get_style(name).grade(picture)

    assert graded.shape == picture.shape and graded.dtype == np.uint8
    assert np.array_equal(picture, before)
    assert not np.array_equal(graded, picture)


def test_street_is_black_and_white(picture):
    graded = get_style("street").grade(picture)
    assert np.array_equal(graded[..., 0], graded[..., 1])
    assert np.array_equal(graded[..., 1], graded[..., 2])


def test_art_film_mutes_colour_and_lifts_blacks(picture):
    def colourfulness(img):
        return float(np.mean(img.max(axis=2).astype(int) - img.min(axis=2).astype(int)))

    graded = get_style("art_film").grade(picture)
    assert colourfulness(graded) < colourfulness(picture)

    black = np.zeros((8, 8, 3), np.uint8)
    assert get_style("art_film").grade(black).min() > 0


def test_tone_curve_identity_and_endpoints():
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    assert np.array_equal(fx.tone_curve(ramp), ramp)

    punchy = fx.tone_curve(ramp, contrast=1.5)
    assert punchy[0, 0, 0] == 0 and punchy[-1, -1, 0] == 255
    assert punchy[4, 0, 0] < ramp[4, 0, 0] and punchy[12, 0, 0] > ramp[12, 0, 0]


# --- Picture area and letterbox ---

@pytest.mark.parametrize(
    "out, expected",
    [
        ((1280, 720), (1280, 536)),    # 16:9 gets 2.39:1 bars
        ((1920, 1080), (1920, 802)),
        ((2560, 1080), (2560, 1070)),  # 21:9 is already close to 2.39:1
        ((1440, 1080), (1440, 1080)),  # 4:3 is left alone
        ((1080, 1080), (1080, 1080)),  # square is left alone
        ((720, 1280), (720, 1280)),    # portrait is left alone
    ],
)
def test_art_film_picture_area(out, expected):
    assert get_style("art_film").picture_size(*out) == expected


# --- Per-frame finishing ---

def test_finisher_pads_to_the_output_frame_with_black_bars():
    art = get_style("art_film")
    finish = art.frame_finisher(1280, 720, 30)
    frame = np.full((536, 1280, 3), 200, np.uint8)

    out = finish(frame, 0.0)

    assert out.shape == (720, 1280, 3) and out.dtype == np.uint8
    assert not out[:92].any() and not out[-92:].any()      # bars are pure black
    assert out[92:628].mean() > 150                        # picture is still there
    assert out[360, 640].mean() > out[100, 8].mean()       # corners are darker than the centre


def test_finisher_never_modifies_its_input():
    finish = get_style("street").frame_finisher(320, 180, 30)
    frame = np.full((180, 320, 3), 120, np.uint8)
    before = frame.copy()

    out = finish(frame, 0.0)

    assert np.array_equal(frame, before)
    assert out is not frame and out.shape == (180, 320, 3)


def test_grain_moves_between_frames_and_repeats_exactly():
    finish = get_style("street").frame_finisher(320, 180, 30)
    frame = np.full((180, 320, 3), 120, np.uint8)

    first, second = finish(frame, 0.0), finish(frame, 1 / 30)
    assert not np.array_equal(first, second)                    # grain is animated
    assert np.array_equal(first, finish(frame, 0.0))            # and deterministic

    again = get_style("street").frame_finisher(320, 180, 30)
    assert np.array_equal(first, again(frame, 0.0))             # across renders too


def test_finisher_accepts_a_frame_of_the_wrong_size():
    finish = get_style("art_film").frame_finisher(1280, 720, 30)
    assert finish(np.full((580, 1380, 3), 90, np.uint8), 0.0).shape == (720, 1280, 3)


# --- API ---

def test_styles_route_lists_plugins():
    from app.routers import styles as styles_router

    app = FastAPI()
    app.include_router(styles_router.router, prefix="/api")
    body = TestClient(app).get("/api/styles").json()

    assert body["default"] == "natural"
    assert body["styles"][0]["name"] == "natural"
    street = next(s for s in body["styles"] if s["name"] == "street")
    assert street["label"] == {"en": "Street photo", "zh": "街头摄影"}
    assert set(street) == {"name", "label", "description"}
