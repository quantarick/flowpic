"""The services that call an LLM, driven by a fake provider."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app.config import settings
from app.llm import (
    ImageInput,
    LLMError,
    LLMNotConfigured,
    LLMQuotaExceeded,
    LLMTimeout,
    parse_json_object,
)
from app.models import (
    AudioFeatures,
    AudioSegment,
    LyricsResult,
    ScrapedPost,
    SegmentLyrics,
    TranscribedWord,
)
from tests.conftest import FakeLLM

CAPTION_REPLY = """TOP: sky
CAPTION: A red barn on a green hill.
ELEMENT: architecture
SUBJECT: 40%, 60%
BOUNDS: 10%, 20%, 90%, 95%
PERSON: NO
PEOPLE: 0
HORIZON: 30%, treeline, yes
CROP_HINT: keep the barn"""


@pytest.fixture
def photo(tmp_path):
    """A flat grey JPEG inside a project-shaped folder.

    It is larger than OpenCV's 64x128 HOG window on purpose: the body
    detector crashes the process on images smaller than that window.
    """
    images = tmp_path / "images"
    images.mkdir()
    path = images / "barn.jpg"
    cv2.imwrite(str(path), np.full((240, 320, 3), 128, np.uint8))
    return path


# --- parse_json_object ---

@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```{"a": 1}```',
        'Here you go:\n{"a": 1}\nHope that helps.',
    ],
)
def test_parse_json_object_finds_the_object(text):
    assert parse_json_object(text) == {"a": 1}


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", "{broken"])
def test_parse_json_object_returns_none_otherwise(text):
    assert parse_json_object(text) is None


# --- ImageCaptioner ---

def test_caption_is_parsed_and_cached(photo):
    from app.services.image_captioner import ImageCaptioner

    llm = FakeLLM(CAPTION_REPLY)
    caption = ImageCaptioner(llm=llm).caption_image(photo)

    assert caption.caption == "A red barn on a green hill."
    assert caption.element_type == "architecture"
    assert (caption.subject_x1, caption.subject_y2) == (0.10, 0.95)
    assert caption.horizon_type == "treeline" and caption.horizon_valid is True
    assert (caption.img_width, caption.img_height) == (320, 240)

    # The image reached the provider as the file's own bytes.
    assert llm.calls[0]["images"] == [ImageInput(photo.read_bytes(), "image/jpeg")]

    cache = photo.parent.parent / "captions" / "barn.json"
    assert json.loads(cache.read_text(encoding="utf-8"))["caption"] == "A red barn on a green hill."

    # A second call is served from the cache.
    ImageCaptioner(llm=llm).caption_image(photo)
    assert len(llm.calls) == 1


def test_failed_caption_is_a_placeholder_and_not_cached(photo):
    from app.services.image_captioner import ImageCaptioner

    caption = ImageCaptioner(llm=FakeLLM(error=LLMError("HTTP 500"))).caption_image(photo)

    assert caption.caption == "An image. (Caption unavailable: HTTP 500)"
    assert not (photo.parent.parent / "captions" / "barn.json").exists()


def test_caption_timeout_uses_the_default_caption(photo):
    from app.services.image_captioner import ImageCaptioner

    caption = ImageCaptioner(llm=FakeLLM(error=LLMTimeout("slow"))).caption_image(photo)
    assert caption.caption == "A photograph with various visual elements."


def test_unusable_provider_stops_captioning(photo):
    from app.services.image_captioner import ImageCaptioner

    captioner = ImageCaptioner(llm=FakeLLM(error=LLMNotConfigured("FLOWPIC_GEMINI_API_KEY not configured")))
    with pytest.raises(LLMNotConfigured):
        captioner.caption_images([photo])


def test_cached_captions_load_without_building_a_provider(photo, monkeypatch):
    from app.services import image_captioner
    from app.services.image_captioner import ImageCaptioner

    ImageCaptioner(llm=FakeLLM(CAPTION_REPLY)).caption_image(photo)

    def forbidden(*args, **kwargs):
        raise AssertionError("no provider should be built to read the cache")

    monkeypatch.setattr(image_captioner, "get_llm", forbidden)
    cached = ImageCaptioner(model="anything", provider="ollama").load_cached_captions([photo])
    assert cached is not None and cached[0].caption == "A red barn on a green hill."


def test_captioner_passes_project_overrides(monkeypatch):
    from app.services import image_captioner
    from app.services.image_captioner import ImageCaptioner

    seen = {}

    def fake_get_llm(role, *, provider=None, model=None):
        seen.update(role=role.value, provider=provider, model=model)
        return FakeLLM("x")

    monkeypatch.setattr(image_captioner, "get_llm", fake_get_llm)
    ImageCaptioner(model="moondream", provider="ollama")._get_llm()
    assert seen == {"role": "vision", "provider": "ollama", "model": "moondream"}


# --- CropReviewer ---

def test_crop_review_parses_the_verdict():
    from app.services.crop_reviewer import CropReviewer

    canvas = np.full((40, 60, 3), 200, np.uint8)
    llm = FakeLLM("COMPLETE: NO\nCUT_OFF: bottom")
    reviewer = CropReviewer(llm=llm)

    assert reviewer._ask_vision(reviewer._encode_canvas(canvas)) == {"complete": False, "cut_off": "bottom"}
    assert llm.calls[0]["images"][0].mime_type == "image/jpeg"


def test_crop_review_failure_counts_as_fine():
    from app.services.crop_reviewer import CropReviewer

    reviewer = CropReviewer(llm=FakeLLM(error=LLMError("HTTP 500")))
    assert reviewer._ask_vision(ImageInput(b"x")) == {"complete": True, "cut_off": "none"}


def test_crop_review_stops_on_unusable_provider():
    from app.services.crop_reviewer import CropReviewer

    reviewer = CropReviewer(llm=FakeLLM(error=LLMNotConfigured("no key")))
    with pytest.raises(LLMNotConfigured):
        reviewer._ask_vision(ImageInput(b"x"))


# --- LyricEmotionAnalyzer ---

LYRIC = SegmentLyrics(segment_index=3, start=0.0, end=4.0, text="hold me close tonight", word_count=4)


def test_lyric_emotion_is_parsed():
    from app.services.lyric_emotion_analyzer import LyricEmotionAnalyzer

    llm = FakeLLM('```json\n{"theme": "longing", "mood_keywords": ["tender"], "mood_description": "Soft."}\n```')
    emotion = LyricEmotionAnalyzer(llm=llm)._analyze_segment(LYRIC)

    assert (emotion.segment_index, emotion.theme, emotion.mood_keywords) == (3, "longing", ["tender"])
    assert "hold me close tonight" in llm.calls[0]["prompt"]
    assert llm.calls[0]["images"] == []


@pytest.mark.parametrize("llm", [FakeLLM("not json"), FakeLLM(error=LLMError("HTTP 500"))])
def test_lyric_failure_skips_the_segment(llm):
    from app.services.lyric_emotion_analyzer import LyricEmotionAnalyzer

    assert LyricEmotionAnalyzer(llm=llm)._analyze_segment(LYRIC) is None


def test_lyric_analysis_stops_on_unusable_provider():
    from app.services.lyric_emotion_analyzer import LyricEmotionAnalyzer

    segment = AudioSegment(
        start=0.0, end=4.0, beat_count=4, rms_energy=0.1, spectral_centroid=1500.0, chroma=[0.0] * 12
    )
    features = AudioFeatures(
        beat_times=[0.0, 1.0, 2.0, 3.0], onset_times=[], tempo=60.0, duration=4.0, segments=[segment]
    )
    lyrics = LyricsResult(
        text="hold me close",
        words=[
            TranscribedWord(word=w, start=0.5 + i, end=1.0 + i, probability=0.9)
            for i, w in enumerate(["hold", "me", "close"])
        ],
        language="en",
        has_vocals=True,
    )
    analyzer = LyricEmotionAnalyzer(llm=FakeLLM(error=LLMNotConfigured("no key")))

    with pytest.raises(LLMNotConfigured):
        analyzer.analyze(lyrics, features)


# --- Copywriting and XHS style analysis ---

@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project folder with one cached caption, isolated from the real data dir."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    proj = tmp_path / "proj"
    (proj / "captions").mkdir(parents=True)
    (proj / "captions" / "a.json").write_text(
        json.dumps({"caption": "A red barn.", "place_name": "Matakana"}), encoding="utf-8"
    )
    return proj


def test_copywriting_uses_the_copywriting_llm(project, monkeypatch):
    from app.services import copywriting_generator

    llm = FakeLLM('{"title": "Barn day", "description": "Went north.", "hashtags": ["#barn"], "cover_index": 0}')
    roles = []
    monkeypatch.setattr(copywriting_generator, "get_llm", lambda role: roles.append(role.value) or llm)

    result = copywriting_generator.generate_copywriting(project, hint="keep it short")

    assert roles == ["copywriting"]
    assert (result.title, result.hashtags) == ("Barn day", ["#barn"])
    call = llm.calls[0]
    assert "A red barn. (location: Matakana)" in call["prompt"]
    assert "keep it short" in call["system"]
    assert call["max_tokens"] == 2048
    assert json.loads((project / "cache" / "copywriting.json").read_text(encoding="utf-8"))["title"] == "Barn day"


def test_copywriting_repairs_unescaped_quotes(project, monkeypatch):
    from app.services import copywriting_generator

    llm = FakeLLM('{"title": "The "best" barn", "description": "ok", "hashtags": [], "cover_index": 0}')
    monkeypatch.setattr(copywriting_generator, "get_llm", lambda role: llm)

    assert "best" in copywriting_generator.generate_copywriting(project).title


def test_copywriting_without_a_key_is_a_value_error(project):
    from app.services.copywriting_generator import generate_copywriting

    # The default provider is Gemini and the test settings hold no key.
    # ValueError is what the router turns into HTTP 400.
    with pytest.raises(ValueError, match="FLOWPIC_GEMINI_API_KEY not configured"):
        generate_copywriting(project)


def test_copywriting_rejects_a_non_json_reply(project, monkeypatch):
    from app.services import copywriting_generator

    monkeypatch.setattr(copywriting_generator, "get_llm", lambda role: FakeLLM("sorry, no"))
    with pytest.raises(ValueError, match="fake returned non-JSON response"):
        copywriting_generator.generate_copywriting(project)


def test_style_analysis_uses_the_copywriting_llm(monkeypatch):
    from app.services import xhs_scraper

    llm = FakeLLM('{"tone": "playful", "sample_phrases": ["so good"], "overall_summary": "light"}')
    monkeypatch.setattr(xhs_scraper, "get_llm", lambda role: llm)
    post = ScrapedPost(title="Day out", description="Loved it", hashtags=["#trip"])

    profile = xhs_scraper._analyze_style([post])

    assert (profile.tone, profile.sample_phrases, profile.overall_summary) == ("playful", ["so good"], "light")
    assert "Day out" in llm.calls[0]["prompt"] and llm.calls[0]["system"]


# --- Captioning many photos ---

@pytest.fixture
def photos(tmp_path):
    """Twelve small photos in a project-shaped folder."""
    images = tmp_path / "images"
    images.mkdir()
    paths = []
    for i in range(12):
        path = images / f"photo{i:02d}.jpg"
        cv2.imwrite(str(path), np.full((240, 320, 3), 60 + 10 * i, np.uint8))
        paths.append(path)
    return paths


def test_hosted_provider_captions_several_photos_at_once(photos):
    from app.services.image_captioner import ImageCaptioner

    llm = FakeLLM(CAPTION_REPLY, delay=0.15, parallel_requests=4)
    seen = []
    captions = ImageCaptioner(llm=llm).caption_images(photos, progress_callback=lambda done, total: seen.append((done, total)))

    assert 2 <= llm.max_active <= 4
    assert [c.filename for c in captions] == [p.name for p in photos]   # input order is kept
    assert seen[-1] == (12, 12) and len(seen) == 12
    assert len(list((photos[0].parent.parent / "captions").glob("*.json"))) == 12


def test_local_provider_captions_one_photo_at_a_time(photos):
    from app.services.image_captioner import ImageCaptioner

    llm = FakeLLM(CAPTION_REPLY, delay=0.02, parallel_requests=1)
    ImageCaptioner(llm=llm).caption_images(photos)

    assert llm.max_active == 1 and len(llm.calls) == 12


def test_caption_parallel_setting_overrides_the_provider(photos, monkeypatch):
    from app.services.image_captioner import ImageCaptioner

    monkeypatch.setattr(settings, "caption_parallel", 2)
    llm = FakeLLM(CAPTION_REPLY, delay=0.1, parallel_requests=4)
    ImageCaptioner(llm=llm).caption_images(photos)

    assert llm.max_active == 2


def test_spent_quota_stops_the_photos_still_waiting(photos):
    from app.services.image_captioner import ImageCaptioner

    llm = FakeLLM(error=LLMQuotaExceeded("20 requests per day"), delay=0.05, parallel_requests=4)

    with pytest.raises(LLMQuotaExceeded):
        ImageCaptioner(llm=llm).caption_images(photos)

    # Only the photos already in flight reached the provider, not all twelve.
    assert len(llm.calls) <= 4


def test_caption_reply_length_is_capped(photo, monkeypatch):
    from app.services.image_captioner import ImageCaptioner

    llm = FakeLLM(CAPTION_REPLY)
    ImageCaptioner(llm=llm).caption_image(photo)
    assert llm.calls[0]["max_tokens"] == 450

    monkeypatch.setattr(settings, "caption_max_tokens", 0)
    ImageCaptioner(llm=llm).caption_image(photo, feedback="the barn")
    assert llm.calls[1]["max_tokens"] is None
