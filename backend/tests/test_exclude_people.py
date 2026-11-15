"""Leaving out photos that have people in them.

The detector's model is never loaded here: its `_run` is replaced by a script
of replies, and the project-level code gets a stand-in detector.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app.models import FaceRegion, ImageCaption, LocationGroup, MatchResult, ProjectConfig, SegmentEmotion
from app.services import person_detector
from app.services.person_detector import PersonDetector, seen_scale, split_by_people, tile_grid
from app.workers import video_worker

PERSON, BIRD = 1, 16


class Scripted(PersonDetector):
    """A detector whose model replies from a list, one reply per region looked at."""

    def __init__(self, replies, score: float = 0.9):
        super().__init__(score=score)
        self.replies = list(replies)
        self.regions: list[tuple[int, int]] = []     # (width, height) of each region looked at

    def _run(self, rgb):
        self.regions.append((rgb.shape[1], rgb.shape[0]))
        return self.replies.pop(0) if self.replies else []


PHOTO = np.zeros((3000, 2000, 3), np.uint8)   # portrait: the whole photo, then 2 x 3 tiles


# --- tiles ---

def test_portrait_photo_gets_two_by_three_overlapping_tiles_that_cover_it():
    tiles = tile_grid(2000, 3000)

    assert len(tiles) == 6
    assert min(t[0] for t in tiles) == 0 and min(t[1] for t in tiles) == 0
    assert max(t[2] for t in tiles) == 2000 and max(t[3] for t in tiles) == 3000
    first, second = tiles[0], tiles[1]
    assert second[0] < first[2]                      # neighbours overlap


def test_landscape_photo_gets_three_by_two_and_a_small_photo_none():
    assert len(tile_grid(3000, 2000)) == 6
    assert tile_grid(3000, 2000)[1][1] == 0          # second tile is still in the top row
    assert tile_grid(900, 600) == []


def test_seen_scale_follows_the_models_resize_rule():
    assert seen_scale(2000, 3000) == pytest.approx(0.4)        # short side to 800
    assert seen_scale(1000, 4000) == pytest.approx(1333 / 4000)  # long side capped at 1333
    assert seen_scale(400, 400) == pytest.approx(2.0)          # small regions are enlarged


# --- what counts as a person ---

def test_clear_person_in_the_whole_photo_is_found_without_looking_at_tiles():
    detector = Scripted([[(PERSON, 0.99, (800.0, 1000.0, 1100.0, 1900.0))]])

    people = detector.find_people(PHOTO)

    assert len(people) == 1 and people[0].score == 0.99
    assert detector.regions == [(2000, 3000)]


def test_person_seen_only_in_a_tile_is_reported_in_photo_coordinates():
    tiles = tile_grid(2000, 3000)
    replies = [[] for _ in range(1 + len(tiles))]
    replies[1 + 3] = [(PERSON, 0.95, (10.0, 20.0, 40.0, 80.0))]     # in the fourth tile
    detector = Scripted(replies)

    people = detector.find_people(PHOTO)

    x0, y0 = tiles[3][0], tiles[3][1]
    assert [(p.x0, p.y0, p.x1, p.y1) for p in people] == [(x0 + 10.0, y0 + 20.0, x0 + 40.0, y0 + 80.0)]
    assert len(detector.regions) == 1 + 4            # stopped at the tile that had someone


def test_unsure_detections_and_other_things_are_not_people():
    everywhere = [[
        (PERSON, 0.74, (800.0, 1000.0, 1100.0, 1900.0)),     # a branch that looks a bit like a person
        (BIRD, 0.99, (100.0, 100.0, 400.0, 400.0)),
    ]] * 7
    detector = Scripted(everywhere)

    assert detector.find_people(PHOTO) == []
    assert len(detector.regions) == 7                # looked everywhere before saying no


def test_a_dot_too_small_to_tell_is_ignored_however_sure_the_model_is():
    # 30 px tall in a region the model shrinks to 40%: 12 px as the model sees it.
    detector = Scripted([[(PERSON, 0.97, (500.0, 500.0, 512.0, 530.0))]])
    assert detector.find_people(PHOTO) == []

    # The same figure in a tile, where it is seen larger, counts.
    tiles = tile_grid(2000, 3000)
    tile_w, tile_h = tiles[0][2] - tiles[0][0], tiles[0][3] - tiles[0][1]
    assert 30 * seen_scale(tile_w, tile_h) >= 16
    detector = Scripted([[], [(PERSON, 0.97, (500.0, 500.0, 512.0, 530.0))]])
    assert len(detector.find_people(PHOTO)) == 1


def test_threshold_comes_from_the_setting_and_can_be_lowered():
    reply = [[(PERSON, 0.74, (800.0, 1000.0, 1100.0, 1900.0))]]
    assert Scripted(reply, score=0.7).find_people(PHOTO)

    from app.config import Settings
    assert Settings(_env_file=None).person_score == 0.9


def test_every_person_is_returned_when_not_stopping_at_the_first():
    replies = [[(PERSON, 0.99, (800.0, 1000.0, 1100.0, 1900.0))]] + [[(PERSON, 0.95, (10.0, 20.0, 60.0, 200.0))]] * 6
    assert len(Scripted(replies).find_people(PHOTO, stop_at_first=False)) == 7


def test_unreadable_file_is_an_error_for_one_photo(tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="broken.jpg"):
        Scripted([]).has_people(bad)


# --- splitting a project's photos ---

class Stand_in:
    """Says a photo has people when its name says so."""

    def __init__(self, score: float = 0.9):
        self.score = score
        self.asked: list[str] = []
        self.unloaded = 0

    def has_people(self, path):
        self.asked.append(path.name)
        if path.name.startswith("broken"):
            raise ValueError(f"Could not read image: {path.name}")
        return path.name.startswith("people")

    def unload(self):
        self.unloaded += 1


@pytest.fixture
def photos(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    paths = []
    for name in ("beach.jpg", "people_walking.jpg", "hill.jpg", "people_pier.jpg"):
        cv2.imwrite(str(images / name), np.full((120, 160, 3), 90, np.uint8))
        paths.append(images / name)
    return paths


def test_split_keeps_order_reports_progress_and_frees_the_model(photos):
    detector, seen = Stand_in(), []

    without, with_people = split_by_people(photos, detector, lambda done, total: seen.append((done, total)))

    assert [p.name for p in without] == ["beach.jpg", "hill.jpg"]
    assert [p.name for p in with_people] == ["people_walking.jpg", "people_pier.jpg"]
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert detector.unloaded == 1


def test_second_run_answers_from_the_cache(photos):
    split_by_people(photos, Stand_in())
    again = Stand_in()

    without, with_people = split_by_people(photos, again)

    assert again.asked == []
    assert len(without) == 2 and len(with_people) == 2
    cached = json.loads((photos[0].parent.parent / "people" / "people_walking.json").read_text(encoding="utf-8"))
    assert cached["has_people"] is True and cached["model"] == person_detector.MODEL_NAME


def test_cache_is_not_used_for_a_different_threshold_or_method(photos, monkeypatch):
    split_by_people(photos, Stand_in(score=0.9))

    stricter = Stand_in(score=0.95)
    split_by_people(photos, stricter)
    assert len(stricter.asked) == 4

    monkeypatch.setattr(person_detector, "CACHE_VERSION", person_detector.CACHE_VERSION + 1)
    newer = Stand_in(score=0.95)
    split_by_people(photos, newer)
    assert len(newer.asked) == 4


def test_unreadable_photo_is_kept_and_not_cached(photos):
    broken = photos[0].parent / "broken.jpg"
    broken.write_bytes(b"not an image")

    without, _ = split_by_people([*photos, broken], Stand_in())

    assert broken in without
    assert not (broken.parent.parent / "people" / "broken.json").exists()


def test_model_is_freed_even_when_the_job_is_cancelled_midway(photos):
    detector = Stand_in()

    def cancel(done, total):
        raise RuntimeError("Task cancelled")

    with pytest.raises(RuntimeError):
        split_by_people(photos, detector, cancel)
    assert detector.unloaded == 1


# --- the pipeline's side of it ---

def test_options_are_off_unless_asked_for():
    config = ProjectConfig()
    assert config.exclude_people is False and config.remove_borders is False
    assert ProjectConfig(**{"aspect_ratio": "9:16"}).exclude_people is False   # a project saved before the options


def test_pipeline_step_returns_the_photos_without_people(photos, monkeypatch):
    monkeypatch.setattr(person_detector, "PersonDetector", Stand_in)
    notes = []

    kept = video_worker._leave_out_people(photos, notes.append, lambda: None)

    assert [p.name for p in kept] == ["beach.jpg", "hill.jpg"]
    assert notes[0] == "0/4 photos checked"
    assert notes[-1] == "Left out 2 photos with people, 2 remaining"


def test_pipeline_step_stops_with_a_clear_message_when_too_few_photos_are_left(photos, monkeypatch):
    monkeypatch.setattr(person_detector, "PersonDetector", Stand_in)

    with pytest.raises(RuntimeError, match="Only 1 of 3 photos have no people"):
        video_worker._leave_out_people(photos[:2] + photos[3:], lambda detail: None, lambda: None)


def test_a_photo_that_passed_the_check_is_treated_as_scenery():
    caption = ImageCaption(
        filename="hill.jpg", caption="A hill.", has_person=True, fit_mode="full", element_type="person",
        face_regions=[FaceRegion(x=1, y=2, w=30, h=30)], people_centers=[[0.5, 0.5]], focus_x=0.3,
    )

    scenery = video_worker._as_scenery(caption)

    assert (scenery.has_person, scenery.face_regions, scenery.people_centers) == (False, [], None)
    assert (scenery.fit_mode, scenery.element_type) == ("crop", None)
    assert scenery.focus_x == 0.3 and scenery.caption == "A hill."
    assert caption.has_person is True and caption.fit_mode == "full"      # the original is untouched
    assert video_worker._as_scenery(ImageCaption(filename="a.jpg", caption="x", element_type="landscape")).element_type == "landscape"


def captions(*names: str) -> list[ImageCaption]:
    return [ImageCaption(filename=n, caption="x") for n in names]


def saved_match(tmp_path, offered: list[str], shown: list[str], record: bool = True):
    matches = [MatchResult(segment_index=i, image_filename=n, similarity_score=0.5) for i, n in enumerate(shown)]
    emotions = [
        SegmentEmotion(segment_index=i, start=float(i), end=float(i + 1), valence=5, arousal=5, mood_description="calm")
        for i in range(len(shown))
    ]
    video_worker._save_matches(tmp_path, captions(*offered), matches, [LocationGroup(place_name="x", start_clip_index=0, end_clip_index=0)], emotions)
    if not record:
        (tmp_path / "cache" / "match_inputs.json").unlink()


def test_cached_match_is_reused_for_the_same_photos(tmp_path):
    saved_match(tmp_path, offered=["a.jpg", "b.jpg", "c.jpg"], shown=["a.jpg", "b.jpg"])

    cached = video_worker._load_cached_matches(tmp_path, captions("c.jpg", "a.jpg", "b.jpg"))

    assert cached is not None
    matches, groups, emotions = cached
    assert [m.image_filename for m in matches] == ["a.jpg", "b.jpg"]
    assert len(groups) == 1 and len(emotions) == 2


def test_cached_match_is_dropped_when_photos_were_left_out_or_brought_back(tmp_path):
    saved_match(tmp_path, offered=["a.jpg", "b.jpg", "c.jpg"], shown=["a.jpg", "b.jpg"])

    assert video_worker._load_cached_matches(tmp_path, captions("a.jpg", "b.jpg")) is None            # c left out
    assert video_worker._load_cached_matches(tmp_path, captions("a.jpg", "b.jpg", "c.jpg", "d.jpg")) is None


def test_match_cached_before_the_photo_list_was_recorded(tmp_path):
    saved_match(tmp_path, offered=["a.jpg", "b.jpg", "c.jpg"], shown=["a.jpg", "b.jpg"], record=False)

    # Still fine when it shows nothing that this run leaves out...
    assert video_worker._load_cached_matches(tmp_path, captions("a.jpg", "b.jpg")) is not None
    # ...and not when it shows a photo that is now left out.
    assert video_worker._load_cached_matches(tmp_path, captions("a.jpg", "c.jpg")) is None


def test_no_cache_means_no_match(tmp_path):
    assert video_worker._load_cached_matches(tmp_path, captions("a.jpg")) is None


def test_saving_embeddings_does_not_write_this_runs_adjustments_into_the_caption_cache(tmp_path):
    caption_dir = tmp_path / "captions"
    caption_dir.mkdir()
    original = ImageCaption(
        filename="hill.jpg", caption="A hill.", has_person=True, fit_mode="full",
        face_regions=[FaceRegion(x=1, y=2, w=30, h=30)],
    )
    (caption_dir / "hill.json").write_text(original.model_dump_json(), encoding="utf-8")

    adjusted = video_worker._as_scenery(original)
    adjusted.clip_embedding = [0.25, 0.5]
    video_worker._persist_clip_embeddings(tmp_path, [adjusted])

    cached = ImageCaption(**json.loads((caption_dir / "hill.json").read_text(encoding="utf-8")))
    assert cached.clip_embedding == [0.25, 0.5]
    assert cached.has_person is True and cached.fit_mode == "full" and len(cached.face_regions) == 1
