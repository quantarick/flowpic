"""End-to-end render through VideoGenerator, with drawn pictures and a generated tone.

This is the real rendering path (fit, Ken Burns, title card, crossfade, style
finishing, FFmpeg encode). Only the analysis stages are replaced by hand-built
inputs, so it needs no model, no Ollama and no network.
"""

from __future__ import annotations

import wave

import cv2
import numpy as np
import pytest

pytest.importorskip("moviepy")

from app.models import (  # noqa: E402
    AspectRatio,
    AudioFeatures,
    AudioSegment,
    FitStrategy,
    ImageCaption,
    LocationGroup,
    MatchResult,
    Quality,
    SegmentEmotion,
)

FPS = 15
CLIP_SECONDS = 1.0


@pytest.fixture
def render_inputs(tmp_path):
    images = tmp_path / "images"
    images.mkdir()

    names = []
    # One landscape picture (crop path) and one portrait picture (pan path).
    for i, (w, h) in enumerate([(1200, 800), (800, 1200)]):
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (200 - 60 * i, 150, 90 + 60 * i)
        cv2.circle(img, (w // 2, h // 2), min(w, h) // 5, (40, 40, 200), -1)
        name = f"picture{i}.jpg"
        cv2.imwrite(str(images / name), img)
        names.append(name)

    total = CLIP_SECONDS * len(names)
    audio = tmp_path / "music.wav"
    with wave.open(str(audio), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(22050)
        t = np.arange(int(22050 * total)) / 22050
        f.writeframes((np.sin(2 * np.pi * 220 * t) * 6000).astype(np.int16).tobytes())

    segments = [
        AudioSegment(
            start=CLIP_SECONDS * i, end=CLIP_SECONDS * (i + 1), beat_count=4,
            rms_energy=0.2, spectral_centroid=1500.0, chroma=[0.0] * 12,
        )
        for i in range(len(names))
    ]
    return {
        "matches": [MatchResult(segment_index=i, image_filename=n, similarity_score=0.5) for i, n in enumerate(names)],
        "segment_emotions": [
            SegmentEmotion(segment_index=i, start=s.start, end=s.end, valence=6.0, arousal=4.0, mood_description="calm")
            for i, s in enumerate(segments)
        ],
        "image_captions": [ImageCaption(filename=n, caption="A red circle.") for n in names],
        "audio_features": AudioFeatures(
            beat_times=[0.5 * i for i in range(int(total * 2))], onset_times=[],
            tempo=120.0, duration=total, segments=segments,
        ),
        "images_dir": images,
        "audio_path": audio,
        # A location group puts a title card on the first clip.
        "location_groups": [LocationGroup(place_name="Matakana", start_clip_index=0, end_clip_index=1)],
    }


def read_frames(path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


@pytest.mark.parametrize(
    "style, fit, bars",
    [
        ("natural", FitStrategy.CROP, False),
        ("art_film", FitStrategy.CROP, True),
        ("street", FitStrategy.FRAME, False),
    ],
)
def test_render_writes_a_playable_video(render_inputs, tmp_path, style, fit, bars):
    from app.services.video_generator import VideoGenerator

    output = tmp_path / "out" / f"{style}.mp4"
    VideoGenerator(
        aspect_ratio=AspectRatio.RATIO_16_9, quality=Quality.SD, fps=FPS,
        fit_strategy=fit, video_style=style,
    ).generate(output_path=output, **render_inputs)

    frames = read_frames(output)
    assert len(frames) >= FPS  # at least a second of video survived the crossfade
    assert all(frame.shape == (720, 1280, 3) for frame in frames)

    last = frames[-1]
    top_band = last[:60]  # inside the 92px letterbox bar of a 2.39:1 picture
    if bars:
        assert top_band.mean() < 8
        assert last[300:420].mean() > 30  # the picture itself is not black
    else:
        assert top_band.mean() > 30


def test_unknown_style_fails_before_any_rendering():
    from app.services.video_generator import VideoGenerator

    with pytest.raises(ValueError, match="Unknown video style 'noir'"):
        VideoGenerator(aspect_ratio=AspectRatio.RATIO_16_9, quality=Quality.SD, video_style="noir")


def flat_inputs(tmp_path, pictures: list[np.ndarray], seconds: float = 1.0) -> dict:
    """Render inputs for the given pictures, one per segment of `seconds`, with a beat every half second."""
    images = tmp_path / "images"
    images.mkdir()
    names = []
    for i, img in enumerate(pictures):
        name = f"picture{i}.png"
        cv2.imwrite(str(images / name), img)
        names.append(name)

    total = seconds * len(names)
    audio = tmp_path / "music.wav"
    with wave.open(str(audio), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(22050)
        t = np.arange(int(22050 * total)) / 22050
        f.writeframes((np.sin(2 * np.pi * 220 * t) * 6000).astype(np.int16).tobytes())

    segments = [
        AudioSegment(start=seconds * i, end=seconds * (i + 1), beat_count=2,
                     rms_energy=0.2, spectral_centroid=1500.0, chroma=[0.0] * 12)
        for i in range(len(names))
    ]
    return {
        "matches": [MatchResult(segment_index=i, image_filename=n, similarity_score=0.5) for i, n in enumerate(names)],
        "segment_emotions": [
            SegmentEmotion(segment_index=i, start=s.start, end=s.end, valence=6.0, arousal=4.0, mood_description="calm")
            for i, s in enumerate(segments)
        ],
        "image_captions": [ImageCaption(filename=n, caption="A picture.") for n in names],
        "audio_features": AudioFeatures(
            beat_times=[0.5 * i for i in range(int(total * 2))], onset_times=[],
            tempo=120.0, duration=total, segments=segments,
        ),
        "images_dir": images,
        "audio_path": audio,
    }


def test_picture_changes_land_on_the_segment_boundaries(tmp_path):
    """Each crossfade is half-way through exactly where the next segment starts."""
    from app.services.video_generator import VideoGenerator

    fps = 30
    levels = [40, 100, 160, 220]                     # four flat grey pictures
    inputs = flat_inputs(tmp_path, [np.full((720, 1280, 3), v, np.uint8) for v in levels])
    output = tmp_path / "out" / "timing.mp4"
    VideoGenerator(aspect_ratio=AspectRatio.RATIO_16_9, quality=Quality.SD, fps=fps).generate(
        output_path=output, **inputs
    )

    brightness = np.array([frame.mean() for frame in read_frames(output)])
    assert abs(len(brightness) - 4 * fps) <= 1       # the video is as long as the music

    plateau = [brightness[int((i + 0.5) * fps)] for i in range(4)]
    for i in (1, 2, 3):
        half_way = (plateau[i - 1] + plateau[i]) / 2
        after = int(np.argmax(brightness >= half_way))                 # first frame past the middle of the fade
        a, b = brightness[after - 1], brightness[after]
        crossing = (after - 1 + (half_way - a) / (b - a)) / fps        # seconds
        assert crossing == pytest.approx(float(i), abs=1.0 / fps), f"change {i} is at {crossing:.3f}s"


def test_remove_borders_takes_a_printed_frame_out_of_the_video(tmp_path):
    from app.services.video_generator import VideoGenerator

    rng = np.random.default_rng(0)
    pictures = []
    for i in range(2):
        y, x = np.mgrid[0:640, 0:1200]
        scene = np.stack([60 + 100 * x / 1200, 90 + 80 * y / 640, 140 - 70 * x / 1200], axis=2)
        scene = np.clip(scene + rng.normal(0, 10, scene.shape), 20, 200).astype(np.uint8)
        framed = np.full((720, 1280, 3), 255, np.uint8)             # a 40 px white mat all round
        framed[40:680, 40:1240] = scene
        pictures.append(framed)
    inputs = flat_inputs(tmp_path, pictures)

    def white_share(remove_borders: bool) -> float:
        output = tmp_path / "out" / f"borders_{remove_borders}.mp4"
        VideoGenerator(
            aspect_ratio=AspectRatio.RATIO_16_9, quality=Quality.SD, fps=FPS, remove_borders=remove_borders,
        ).generate(output_path=output, **inputs)
        return max(float((frame.min(axis=2) > 235).mean()) for frame in read_frames(output))

    assert white_share(False) > 0.02        # as before: the mat is in the picture
    assert white_share(True) < 0.001        # with the option: none of it is


def top_level_atoms(path) -> list[str]:
    """Names of the top-level boxes of an MP4 file, in file order."""
    atoms = []
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            size = int.from_bytes(header[:4], "big")
            atoms.append(header[4:8].decode("latin-1"))
            if size == 0:      # box runs to the end of the file
                break
            if size == 1:      # 64-bit size follows the header
                size = int.from_bytes(f.read(8), "big")
                f.seek(size - 16, 1)
            else:
                f.seek(size - 8, 1)
    return atoms


@pytest.mark.parametrize(
    "aspect, style, frame_shape",
    [
        (AspectRatio.RATIO_9_19_5, "natural", (1560, 720, 3)),   # iPhone full screen
        (AspectRatio.RATIO_9_19_5, "art_film", (1560, 720, 3)),  # no cinema bars on a tall frame
        (AspectRatio.RATIO_3_4, "natural", (960, 720, 3)),       # vertical photo
        (AspectRatio.RATIO_9_16, "street", (1280, 720, 3)),
    ],
)
def test_vertical_render(render_inputs, tmp_path, aspect, style, frame_shape):
    from app.services.video_generator import VideoGenerator

    output = tmp_path / "out" / "vertical.mp4"
    VideoGenerator(aspect_ratio=aspect, quality=Quality.SD, fps=FPS, video_style=style).generate(
        output_path=output, **render_inputs
    )

    frames = read_frames(output)
    assert len(frames) >= FPS
    assert all(frame.shape == frame_shape for frame in frames)
    assert frames[-1][:60].mean() > 20  # the top of the frame is picture, not a black bar

    # The index sits before the media data, so a phone can start playing at once.
    atoms = top_level_atoms(output)
    assert atoms.index("moov") < atoms.index("mdat"), atoms
