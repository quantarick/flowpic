"""Find and remove a plain frame that is part of a photo.

Some photos arrive with a border already printed on them: a white or black
mat, often carrying small text such as a title or a camera watermark. In a
video that border shows up as bands that slide around with the picture.
`find_frame` locates the picture inside such a border and `remove_frame`
cuts the image down to it, moving a caption's coordinates to match.

Only a frame on all four sides in one flat colour is removed. Bars on two
sides, or a picture whose own edges are simply dark or pale, are left alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.models import FaceRegion, ImageCaption

# Detection runs on a copy this many pixels on its longer side.
_WORK_SIDE = 1200
# How far a pixel may be from the frame colour, per channel, and still count as frame.
_TOLERANCE = 10
# Share of the outermost ring that must be frame-coloured before we look further.
_RING_UNIFORM = 0.97
# A border is searched for within this share of each side...
_MAX_BORDER = 0.20
# ...and has to be at least this share of the side to count.
_MIN_BORDER = 0.003
# Picture content is a run of non-frame pixels at least this share of the side long.
# Text printed on the frame is much shorter, so it is skipped over.
_CONTENT_RUN = 0.04
# Share of lines that must agree on where the picture starts.
_MIN_SUPPORT = 0.12
# A full-size row with fewer non-frame pixels than this share is pure frame.
_PURE_FRAME_ROW = 0.02
# How far past the coarse edge to look for a thin line and gap, as a share of the side.
_SETTLE_REACH = 0.015
# Full-size pixels cut off inside the settled edge, where frame and picture blend.
_INSET = 2


@dataclass(frozen=True)
class Box:
    """The picture inside a frame, in pixels. `right` and `bottom` are exclusive."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def _first_content(not_frame: np.ndarray, run: int, limit: int) -> np.ndarray:
    """For each column, the first row that starts a long run of non-frame pixels.

    `not_frame` is rows x columns, scanned from row 0. Returns -1 for a column
    with no such run starting inside the first `limit` rows.
    """
    rows = not_frame.shape[0]
    run = max(2, min(run, rows))
    csum = np.cumsum(not_frame, axis=0, dtype=np.int32)
    csum = np.vstack([np.zeros((1, not_frame.shape[1]), np.int32), csum])
    last = rows - run
    if last < 0:
        return np.full(not_frame.shape[1], -1, np.int32)
    window = csum[run:run + last + 1] - csum[:last + 1]          # non-frame pixels in [y, y + run)
    starts = not_frame[:last + 1] & (window >= int(np.ceil(run * 0.97)))
    starts = starts[:limit + 1]
    depth = np.argmax(starts, axis=0).astype(np.int32)
    depth[~starts.any(axis=0)] = -1
    return depth


def _edge(not_frame: np.ndarray, run: int, limit: int) -> Optional[int]:
    """Where the picture starts when scanning `not_frame` from row 0, or None.

    Lines that reach picture content vote for the row where it starts. The
    picture edge is the outermost row with enough votes: lines where the
    picture is frame-coloured at its edge (a blown-out sky) vote for rows
    further in and are outvoted.
    """
    depth = _first_content(not_frame, run, limit)
    valid = depth[depth >= 0]
    if valid.size < max(8, int(0.2 * depth.size)):
        return None
    votes = np.bincount(valid, minlength=limit + 3).astype(np.float64)
    near = votes.copy()
    near[1:] += votes[:-1]
    near[:-1] += votes[1:]
    agreed = np.nonzero(near >= _MIN_SUPPORT * valid.size)[0]
    if agreed.size == 0:
        return None
    # The first agreed row can be the faint outer fringe of the edge; take the
    # strongest row among it and its two inner neighbours.
    first = int(agreed[0])
    return first + int(np.argmax(votes[first:first + 3]))


def _settle(off_frame: np.ndarray, coarse: int, back: int, reach: int) -> int:
    """Move a coarse edge, found on the small copy, to the first row of picture.

    `off_frame` is the share of non-frame pixels in each full-size row, counted
    inward from the image edge. A frame may carry a thin line with a gap inside
    it, too fine to see on the small copy. The picture starts after the last
    row that is pure frame near the coarse edge.
    """
    lo = max(0, coarse - back)
    hi = min(off_frame.size, coarse + reach)
    pure = np.nonzero(off_frame[lo:hi] < _PURE_FRAME_ROW)[0]
    if pure.size == 0:
        return coarse
    return lo + int(pure[-1]) + 1


def find_frame(image: np.ndarray) -> Optional[Box]:
    """Locate the picture inside a flat frame. Returns None when there is no frame.

    `image` is a BGR or RGB array as loaded. The box is in its pixels.
    """
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return None
    full_h, full_w = image.shape[:2]
    if min(full_h, full_w) < 64:
        return None

    scale = max(full_h, full_w) / _WORK_SIDE
    if scale > 1.0:
        work = cv2.resize(
            image, (max(1, round(full_w / scale)), max(1, round(full_h / scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
        work = image
    work = work[:, :, :3].astype(np.int16)
    h, w = work.shape[:2]

    # The frame colour is whatever the outermost ring is, if the ring is one colour.
    ring_px = max(2, round(0.006 * min(h, w)))
    ring = np.concatenate([
        work[:ring_px].reshape(-1, 3), work[-ring_px:].reshape(-1, 3),
        work[:, :ring_px].reshape(-1, 3), work[:, -ring_px:].reshape(-1, 3),
    ])
    colour = np.median(ring, axis=0)
    if (np.abs(ring - colour).max(axis=1) <= _TOLERANCE).mean() < _RING_UNIFORM:
        return None

    not_frame = np.abs(work - colour).max(axis=2) > _TOLERANCE

    top = _edge(not_frame, round(_CONTENT_RUN * h), int(_MAX_BORDER * h))
    bottom = _edge(not_frame[::-1], round(_CONTENT_RUN * h), int(_MAX_BORDER * h))
    left = _edge(not_frame.T, round(_CONTENT_RUN * w), int(_MAX_BORDER * w))
    right = _edge(not_frame.T[::-1], round(_CONTENT_RUN * w), int(_MAX_BORDER * w))
    if None in (top, bottom, left, right):
        return None
    if min(top, bottom) < max(2, _MIN_BORDER * h) or min(left, right) < max(2, _MIN_BORDER * w):
        return None

    # What lies outside the picture must really be frame: flat, apart from a
    # thin outline and a little text. Picture content would be mostly not flat.
    outside = np.ones((h, w), bool)
    outside[top:h - bottom, left:w - right] = False
    if (~not_frame[outside]).mean() < 0.75:
        return None

    # A framed picture is a rectangle, so it reaches into the corners of its
    # box. An object on a plain backdrop (a bottle, a ball) does not, and its
    # backdrop is part of the picture. One corner may be empty: a pale sky.
    inner = not_frame[top:h - bottom, left:w - right]
    ph, pw = max(4, round(0.04 * inner.shape[0])), max(4, round(0.04 * inner.shape[1]))
    corners = [inner[2:2 + ph, 2:2 + pw], inner[2:2 + ph, -2 - pw:-2],
               inner[-2 - ph:-2, 2:2 + pw], inner[-2 - ph:-2, -2 - pw:-2]]
    if sum(c.mean() >= 0.5 for c in corners) < 3:
        return None

    # Back to full size. Each edge is settled on the full-size pixels, looking
    # only at the middle of that side so the corners and the other borders
    # stay out of it.
    def off_frame(band: np.ndarray, axis: int) -> np.ndarray:
        return (np.abs(band[:, :, :3].astype(np.int16) - colour).max(axis=2) > _TOLERANCE).mean(axis=axis)

    back = int(np.ceil(2 * scale))
    reach_y = max(8, round(_SETTLE_REACH * full_h))
    reach_x = max(8, round(_SETTLE_REACH * full_w))
    t0, b0, l0, r0 = (int(v * scale) for v in (top, bottom, left, right))
    pad_x = int(0.1 * (full_w - l0 - r0))
    pad_y = int(0.1 * (full_h - t0 - b0))
    xs = slice(l0 + pad_x, full_w - r0 - pad_x)
    ys = slice(t0 + pad_y, full_h - b0 - pad_y)

    top_px = _settle(off_frame(image[:t0 + reach_y, xs], 1), t0, back, reach_y)
    bottom_px = _settle(off_frame(image[full_h - (b0 + reach_y):, xs], 1)[::-1], b0, back, reach_y)
    left_px = _settle(off_frame(image[ys, :l0 + reach_x], 0), l0, back, reach_x)
    right_px = _settle(off_frame(image[ys, full_w - (r0 + reach_x):], 0)[::-1], r0, back, reach_x)

    box = Box(
        left=min(full_w, left_px + _INSET),
        top=min(full_h, top_px + _INSET),
        right=max(0, full_w - right_px - _INSET),
        bottom=max(0, full_h - bottom_px - _INSET),
    )
    if box.width < 0.5 * full_w or box.height < 0.5 * full_h:
        return None
    return box


def _shift(value: Optional[float], start: int, size: int, full: int) -> Optional[float]:
    """Move a 0-1 position measured on the full image onto the cut-down one."""
    if value is None:
        return None
    return float(min(1.0, max(0.0, (value * full - start) / size)))


def reframe_caption(caption: ImageCaption, box: Box, full_w: int, full_h: int) -> ImageCaption:
    """A copy of `caption` whose coordinates refer to the picture inside `box`."""
    # Face boxes are in the caption's own pixels, which may not be the loaded size.
    cap_w = caption.img_width or full_w
    cap_h = caption.img_height or full_h
    sx, sy = cap_w / full_w, cap_h / full_h
    new_w, new_h = max(1, round(box.width * sx)), max(1, round(box.height * sy))

    faces: list[FaceRegion] = []
    for f in caption.face_regions:
        x0 = max(0, round(f.x - box.left * sx))
        y0 = max(0, round(f.y - box.top * sy))
        x1 = min(new_w, round(f.x + f.w - box.left * sx))
        y1 = min(new_h, round(f.y + f.h - box.top * sy))
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            faces.append(FaceRegion(x=x0, y=y0, w=x1 - x0, h=y1 - y0))

    def x(value: Optional[float]) -> Optional[float]:
        return _shift(value, box.left, box.width, full_w)

    def y(value: Optional[float]) -> Optional[float]:
        return _shift(value, box.top, box.height, full_h)

    people = None
    if caption.people_centers:
        people = [[x(p[0]), y(p[1])] for p in caption.people_centers if len(p) >= 2]

    return caption.model_copy(update={
        "face_regions": faces,
        "focus_x": x(caption.focus_x),
        "focus_y": y(caption.focus_y),
        "subject_x1": x(caption.subject_x1),
        "subject_y1": y(caption.subject_y1),
        "subject_x2": x(caption.subject_x2),
        "subject_y2": y(caption.subject_y2),
        "horizon_y": y(caption.horizon_y),
        "people_centers": people,
        "img_width": new_w if caption.img_width else caption.img_width,
        "img_height": new_h if caption.img_height else caption.img_height,
    })


def remove_frame(
    image: np.ndarray, caption: Optional[ImageCaption] = None,
) -> tuple[np.ndarray, Optional[ImageCaption]]:
    """Cut a printed frame off `image`. Returns the image and caption unchanged if it has none."""
    box = find_frame(image)
    if box is None:
        return image, caption
    full_h, full_w = image.shape[:2]
    cut = np.ascontiguousarray(image[box.top:box.bottom, box.left:box.right])
    if caption is not None:
        caption = reframe_caption(caption, box, full_w, full_h)
    return cut, caption
