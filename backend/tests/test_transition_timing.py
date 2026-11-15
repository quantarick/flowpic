"""Picture changes land on the beat: clip lengths and start times."""

from __future__ import annotations

import pytest

pytest.importorskip("moviepy")

from app.core.transitions import clip_durations, clip_starts, fit_crossfade  # noqa: E402
from app.models import SegmentEmotion  # noqa: E402


def boundaries(segment_durations: list[float]) -> list[float]:
    """Music time at which each segment after the first starts."""
    out, t = [], 0.0
    for d in segment_durations[:-1]:
        t += d
        out.append(t)
    return out


def fade_middles(segment_durations: list[float], xfade: float) -> list[float]:
    """Video time of the middle of each crossfade."""
    starts = clip_starts(clip_durations(segment_durations, xfade), xfade)
    return [s + xfade / 2 for s in starts[1:]]


SONG = [11.42, 6.81, 4.46, 2.14, 4.46, 5.02, 3.9, 4.46, 2.23, 6.62]


@pytest.mark.parametrize("xfade", [0.3, 0.6, 0.05])
def test_every_fade_is_centred_on_its_segment_boundary(xfade):
    assert fade_middles(SONG, xfade) == pytest.approx(boundaries(SONG), abs=1e-9)


@pytest.mark.parametrize("xfade", [0.3, 0.6])
def test_the_video_is_as_long_as_the_music(xfade):
    durations = clip_durations(SONG, xfade)
    starts = clip_starts(durations, xfade)

    assert starts[-1] + durations[-1] == pytest.approx(sum(SONG), abs=1e-9)


def test_an_equal_share_of_the_fade_per_clip_would_drift():
    # What the renderer did before: the right total, with the first change half
    # a fade late and the last half a fade early. Kept here as the reason the
    # rule above exists.
    xfade, n = 0.3, len(SONG)
    even = [d + (n - 1) * xfade / n for d in SONG]
    middles = [s + xfade / 2 for s in clip_starts(even, xfade)[1:]]
    errors = [m - b for m, b in zip(middles, boundaries(SONG))]

    assert errors[0] == pytest.approx(xfade / 2 - xfade / n)
    assert errors[-1] == pytest.approx(-(xfade / 2 - xfade / n))


def test_one_clip_or_no_fade_changes_nothing():
    assert clip_durations([5.0], 0.3) == [5.0]
    assert clip_durations([2.0, 3.0], 0.0) == [2.0, 3.0]
    assert clip_durations([], 0.3) == []


def test_fade_is_shortened_when_a_segment_has_no_room_for_it():
    assert fit_crossfade([4.0, 4.0, 4.0], 0.6) == 0.6
    assert fit_crossfade([4.0, 0.5, 4.0], 0.6) == pytest.approx(0.2)
    assert fit_crossfade([4.0], 0.6) == 0.0
    assert fit_crossfade([], 0.6) == 0.0


def test_fitted_fade_survives_the_limit_inside_crossfade_clips():
    # crossfade_clips caps the fade at 40% of the shortest clip. Clips sized by
    # clip_durations with a fitted fade are never shorter than that needs.
    for durations in ([4.0, 0.5, 4.0], [0.4, 0.4], SONG):
        xfade = fit_crossfade(durations, 0.6)
        assert xfade <= min(clip_durations(durations, xfade)) * 0.4 + 1e-9


# --- the renderer's plan, made from the segments that will actually be shown ---

def seg(index: int, start: float, end: float) -> SegmentEmotion:
    return SegmentEmotion(
        segment_index=index, start=start, end=end, valence=5.0, arousal=5.0, mood_description="calm",
    )


def plan(segments, xfade=0.3):
    from app.services.video_generator import VideoGenerator

    return VideoGenerator._plan_timing(segments, xfade)


def test_plan_puts_each_change_at_the_start_of_the_next_segment():
    segments = [seg(0, 0.0, 4.0), seg(1, 4.0, 6.5), seg(2, 6.5, 12.0)]

    xfade, durations = plan(segments)
    starts = clip_starts(durations, xfade)

    assert xfade == 0.3
    assert [s + xfade / 2 for s in starts[1:]] == pytest.approx([4.0, 6.5])
    assert starts[-1] + durations[-1] == pytest.approx(12.0)


def test_a_segment_with_no_clip_does_not_move_the_changes_after_it():
    # Segment 1 (4.0 to 6.5) has no picture. The first clip stays up until 6.5.
    segments = [seg(0, 0.0, 4.0), seg(2, 6.5, 12.0), seg(3, 12.0, 15.0)]

    xfade, durations = plan(segments)
    starts = clip_starts(durations, xfade)

    assert [s + xfade / 2 for s in starts[1:]] == pytest.approx([6.5, 12.0])
    assert starts[-1] + durations[-1] == pytest.approx(15.0)


def test_first_clip_starts_with_the_music_even_if_its_segment_does_not():
    segments = [seg(1, 2.0, 5.0), seg(2, 5.0, 9.0)]

    xfade, durations = plan(segments)

    assert clip_starts(durations, xfade)[1] + xfade / 2 == pytest.approx(5.0)
    assert durations[0] == pytest.approx(5.0 + xfade / 2)


def test_segments_out_of_time_order_fall_back_to_their_own_lengths():
    segments = [seg(2, 6.0, 9.0), seg(0, 0.0, 4.0)]

    xfade, durations = plan(segments)

    assert durations == pytest.approx([3.0 + xfade / 2, 4.0 + xfade / 2])


def test_plan_for_nothing_and_for_one_clip():
    assert plan([]) == (0.0, [])
    assert plan([seg(0, 0.0, 7.0)]) == (0.0, [7.0])
