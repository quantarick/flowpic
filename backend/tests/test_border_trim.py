"""Finding and cutting off a frame that is printed on a photo."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.models import FaceRegion, ImageCaption
from app.services.border_trim import Box, find_frame, reframe_caption, remove_frame


def scene(width: int, height: int, seed: int = 0) -> np.ndarray:
    """A busy picture with no flat areas: smooth colour ramps plus texture."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    img = np.stack([
        60 + 100 * x / width,
        90 + 80 * y / height,
        140 - 70 * x / width,
    ], axis=2)
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 20, 215).astype(np.uint8)


def framed(
    picture: np.ndarray, left: int, top: int, right: int, bottom: int,
    colour: int = 255, line: int = 0, gap: int = 0, text: bool = False,
) -> np.ndarray:
    """Put `picture` on a flat mat. `line` and `gap` add a thin outline with a gap inside it."""
    h, w = picture.shape[:2]
    out = np.full((h + top + bottom, w + left + right, 3), colour, np.uint8)
    out[top:top + h, left:left + w] = picture
    if line:
        d = line + gap
        cv2.rectangle(out, (left - d, top - d), (left + w + d - 1, top + h + d - 1), (25, 25, 25), line)
        # cv2 draws the outline centred on the path; redraw the gap so it is clean
        out[top - gap:top, left - gap:left + w + gap] = colour
        out[top + h:top + h + gap, left - gap:left + w + gap] = colour
        out[top - gap:top + h + gap, left - gap:left] = colour
        out[top - gap:top + h + gap, left + w:left + w + gap] = colour
    if text:
        cv2.putText(out, "CAMERA MODEL NAME", (out.shape[1] // 2 - 90, max(14, top // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(out, "2020-01-01", (left, out.shape[0] - max(6, bottom // 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    return out


def assert_box(box: Box | None, left: int, top: int, right: int, bottom: int, slack: int = 4):
    """The box is inside the picture and within `slack` pixels of each of its edges."""
    assert box is not None
    assert left <= box.left <= left + slack
    assert top <= box.top <= top + slack
    assert right - slack <= box.right <= right
    assert bottom - slack <= box.bottom <= bottom


# --- frames that should be found ---

def test_plain_white_mat_is_found_and_none_of_it_is_left():
    img = framed(scene(600, 1000), left=24, top=45, right=24, bottom=50)

    box = find_frame(img)

    assert_box(box, 24, 45, 624, 1045)
    cut = img[box.top:box.bottom, box.left:box.right]
    assert cut.min(axis=2).max() < 235          # no white pixel survives


def test_thin_outline_with_a_gap_inside_it_goes_too():
    # The frame on the photos this was written for: white mat, a dark hairline,
    # a few pixels of white, then the picture, with small text on the mat.
    img = framed(scene(900, 1560), left=32, top=74, right=32, bottom=80, line=3, gap=4, text=True)

    box = find_frame(img)

    assert_box(box, 32, 74, 932, 1634)
    cut = img[box.top:box.bottom, box.left:box.right]
    assert cut.min(axis=2).max() < 235          # the gap is gone
    assert (cut.max(axis=2) < 40).mean() < 0.001  # and so is the hairline


def test_large_photo_is_measured_at_full_size():
    img = framed(scene(2700, 4660, seed=3), left=90, top=224, right=92, bottom=240, line=4, gap=4, text=True)

    assert_box(find_frame(img), 90, 224, 2790, 4884)


def test_black_frame():
    img = framed(scene(800, 600, seed=1) + 30, left=40, top=30, right=40, bottom=30, colour=0)

    assert_box(find_frame(img), 40, 30, 840, 630)


def test_picture_that_is_white_along_most_of_one_edge():
    # A blown-out sky: 70% of the top edge is the same white as the mat. The
    # edge is still found from the part that is not.
    picture = scene(800, 1200, seed=2)
    picture[:260, :560] = 255
    img = framed(picture, left=30, top=50, right=30, bottom=50)

    assert_box(find_frame(img), 30, 50, 830, 1250)


# --- things that are not frames ---

def test_ordinary_photo_has_no_frame():
    assert find_frame(scene(900, 600)) is None


def test_bars_on_two_sides_only_are_left_alone():
    img = framed(scene(900, 500), left=0, top=60, right=0, bottom=60, colour=0)
    assert find_frame(img) is None


def test_flat_picture_with_no_edges_has_no_frame():
    assert find_frame(np.full((400, 600, 3), 255, np.uint8)) is None


def test_object_on_a_white_background_is_not_cut_down_to_the_object():
    img = np.full((800, 800, 3), 255, np.uint8)
    cv2.circle(img, (400, 400), 300, (40, 90, 160), -1)
    assert find_frame(img) is None


def test_mat_wider_than_a_fifth_of_the_side_is_not_treated_as_a_frame():
    img = framed(scene(300, 300), left=200, top=200, right=200, bottom=200)
    assert find_frame(img) is None


@pytest.mark.parametrize("shape", [(40, 40, 3), (300, 300)])
def test_tiny_or_grey_images_are_skipped(shape):
    assert find_frame(np.full(shape, 255, np.uint8)) is None


# --- cutting, and moving the caption with the picture ---

def test_remove_frame_returns_the_same_objects_when_there_is_no_frame():
    img = scene(600, 400)
    caption = ImageCaption(filename="a.jpg", caption="x")

    out_img, out_caption = remove_frame(img, caption)

    assert out_img is img and out_caption is caption


def test_remove_frame_cuts_the_image_and_moves_the_caption():
    img = framed(scene(800, 1200), left=100, top=200, right=100, bottom=200)   # 1000 x 1600
    caption = ImageCaption(
        filename="a.jpg", caption="x",
        img_width=1000, img_height=1600,
        focus_x=0.5, focus_y=0.5,
        subject_x1=0.1, subject_y1=0.125, subject_x2=0.9, subject_y2=0.875,   # exactly the picture
        horizon_y=0.5,
        people_centers=[[0.3, 0.5]],
        face_regions=[FaceRegion(x=400, y=600, w=100, h=120), FaceRegion(x=0, y=0, w=50, h=50)],
    )

    cut, moved = remove_frame(img, caption)

    assert abs(cut.shape[1] - 800) <= 8 and abs(cut.shape[0] - 1200) <= 8
    assert moved is not caption and caption.img_width == 1000      # the original is untouched
    assert (moved.img_width, moved.img_height) == (cut.shape[1], cut.shape[0])
    assert moved.focus_x == pytest.approx(0.5, abs=0.01)
    assert moved.focus_y == pytest.approx(0.5, abs=0.01)
    assert (moved.subject_x1, moved.subject_y1) == pytest.approx((0.0, 0.0), abs=0.01)
    assert (moved.subject_x2, moved.subject_y2) == pytest.approx((1.0, 1.0), abs=0.01)
    assert moved.horizon_y == pytest.approx(0.5, abs=0.01)
    assert moved.people_centers[0] == pytest.approx([0.25, 0.5], abs=0.01)
    # The face inside the picture moves with it; the one that sat on the mat is dropped.
    assert len(moved.face_regions) == 1
    face = moved.face_regions[0]
    assert (face.x, face.y) == pytest.approx((300, 400), abs=5)
    assert (face.w, face.h) == (100, 120)


def test_reframe_caption_uses_the_captions_own_pixel_size():
    # A caption measured on a half-size copy of the photo.
    caption = ImageCaption(
        filename="a.jpg", caption="x", img_width=500, img_height=800,
        face_regions=[FaceRegion(x=200, y=300, w=50, h=60)],
    )

    moved = reframe_caption(caption, Box(left=100, top=200, right=900, bottom=1400), 1000, 1600)

    assert (moved.img_width, moved.img_height) == (400, 600)
    assert moved.face_regions == [FaceRegion(x=150, y=200, w=50, h=60)]


def test_caption_without_optional_fields_stays_without_them():
    caption = ImageCaption(filename="a.jpg", caption="x")

    moved = reframe_caption(caption, Box(10, 10, 90, 90), 100, 100)

    assert moved.subject_x1 is None and moved.horizon_y is None and moved.people_centers is None
    assert moved.img_width is None and moved.img_height is None
