"""Smart image fitting for full-screen video frames.

Two modes decided by the LLM:
- "crop": scale to cover canvas, crop centered on interest point (faces/saliency)
- "full": blurred background fill — full image centered over a blurred zoom of itself,
  used for portraits/people to avoid cutting off subjects.
No black bars in either mode.
"""

import cv2
import numpy as np
from dataclasses import dataclass

from app.models import AspectRatio, FaceRegion, Quality

RESOLUTIONS: dict[tuple[str, str], tuple[int, int]] = {
    ("16:9", "720p"): (1280, 720),
    ("16:9", "1080p"): (1920, 1080),
    ("16:9", "2k"): (2560, 1440),
    ("16:9", "4k"): (3840, 2160),
    ("21:9", "720p"): (1680, 720),
    ("21:9", "1080p"): (2520, 1080),
    ("21:9", "2k"): (3440, 1440),
    ("21:9", "4k"): (5040, 2160),
    ("9:16", "720p"): (720, 1280),
    ("9:16", "1080p"): (1080, 1920),
    ("9:16", "2k"): (1440, 2560),
    ("9:16", "4k"): (2160, 3840),
    ("1:1", "720p"): (720, 720),
    ("1:1", "1080p"): (1080, 1080),
    ("1:1", "2k"): (1440, 1440),
    ("1:1", "4k"): (2160, 2160),
    ("4:3", "720p"): (960, 720),
    ("4:3", "1080p"): (1440, 1080),
    ("4:3", "2k"): (1920, 1440),
    ("4:3", "4k"): (2880, 2160),
}


@dataclass
class FitResult:
    canvas: np.ndarray          # The output image (canvas with fitted image)
    x_offset: int               # Where the image content starts (x)
    y_offset: int               # Where the image content starts (y)
    fit_w: int                  # Width of the fitted image on canvas
    fit_h: int                  # Height of the fitted image on canvas
    scale: float                # Scale factor applied to original image
    content_center_x: float     # Center of content area in canvas coords (normalized 0..1)
    content_center_y: float     # Center of content area in canvas coords (normalized 0..1)


def get_output_resolution(
    aspect_ratio: AspectRatio, quality: Quality
) -> tuple[int, int]:
    return RESOLUTIONS[(aspect_ratio.value, quality.value)]


def smart_fit(
    image: np.ndarray,
    target_w: int,
    target_h: int,
    face_regions: list[FaceRegion] | None = None,
    focus_x: float = 0.5,
    focus_y: float = 0.5,
    scale_factor: float = 1.1,
    fit_mode: str = "crop",
    subject_box: tuple[float, float, float, float] | None = None,
    horizon_y: float | None = None,
    people_centers: list[list[float]] | None = None,
) -> FitResult:
    """Fit image to canvas using LLM-decided mode.

    fit_mode="crop": scale to cover, crop centered on interest point.
    fit_mode="full": blurred background fill to preserve full subject.
    subject_box: (x1, y1, x2, y2) normalized 0-1 bounding box of main subject.
    horizon_y: LLM-provided horizon position (0-1), used instead of CV detection.
    people_centers: [[x,y], ...] LLM-provided person center positions (0-1).
    """
    h, w = image.shape[:2]
    out_w = int(target_w * scale_factor)
    out_h = int(target_h * scale_factor)

    if fit_mode == "full" and not face_regions:
        # LLM said person but no CV face detection to confirm.
        # Use people_centers if available (more precise than subject_box).
        if people_centers and len(people_centers) > 0:
            avg_y = sum(p[1] for p in people_centers) / len(people_centers)
            avg_x = sum(p[0] for p in people_centers) / len(people_centers)
            return _crop_fill(image, out_w, out_h, None, avg_x, avg_y, None,
                              horizon_y=horizon_y)
        # Use subject_box if valid, otherwise fall back to saliency.
        if subject_box is not None:
            sx1, sy1, sx2, sy2 = subject_box
            box_area = (sx2 - sx1) * (sy2 - sy1)
            box_height = sy2 - sy1
            if box_area >= 0.05 and box_height >= 0.15:
                # Partial-body capture: box starts far down (>60%) AND is
                # small relative to its position (ratio > 3×) — LLM only
                # identified legs/lower body, head is much higher.
                if sy1 > 0.6 and sy1 / max(box_height, 0.01) > 3.0:
                    head_fy = sy1 * 0.5
                    head_fy = max(0.20, min(0.45, head_fy))
                    return _crop_fill(image, out_w, out_h, None,
                                      (sx1 + sx2) / 2, head_fy, None,
                                      horizon_y=horizon_y)
                # Valid box covering person — use upper-fifth as focus
                # (biased toward head area for person subjects).
                box_focus_y = sy1 + box_height * 0.20
                return _crop_fill(image, out_w, out_h, None,
                                  (sx1 + sx2) / 2, box_focus_y, None,
                                  horizon_y=horizon_y)
        return _crop_fill(image, out_w, out_h, None, 0.5, 0.5, None,
                          horizon_y=horizon_y)

    if fit_mode == "full":
        return _blur_fill(image, out_w, out_h, face_regions, focus_x, focus_y, subject_box,
                          people_centers=people_centers)

    return _crop_fill(image, out_w, out_h, face_regions, focus_x, focus_y, subject_box,
                      horizon_y=horizon_y)


def _crop_fill(
    image: np.ndarray,
    out_w: int,
    out_h: int,
    face_regions: list[FaceRegion] | None,
    focus_x: float,
    focus_y: float,
    subject_box: tuple[float, float, float, float] | None = None,
    horizon_y: float | None = None,
) -> FitResult:
    """Crop-to-fill: scale to cover, crop centered on interest point."""
    h, w = image.shape[:2]

    # Priority: subject_box > face_regions > focus point > saliency
    # subject_box knows the full subject extent (head to toe), so it's most reliable.
    # Reject tiny boxes (area < 5% or height < 15%) — likely hallucinated by LLM.
    if subject_box is not None:
        sx1, sy1, sx2, sy2 = subject_box
        box_area = (sx2 - sx1) * (sy2 - sy1)
        box_height = sy2 - sy1
        if box_area >= 0.05 and box_height >= 0.15:
            return _subject_box_crop(image, out_w, out_h, subject_box)
        # Rejected subject_box → focus from same LLM call is equally unreliable.
        # Reset to default so saliency centering takes over.
        focus_x = 0.5
        focus_y = 0.5

    img_aspect = w / h
    target_aspect = out_w / out_h

    if img_aspect > target_aspect:
        scale = out_h / h
    else:
        scale = out_w / w

    scaled_w = int(w * scale)
    scaled_h = int(h * scale)
    scaled = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

    # Determine crop center: faces > LLM focus > LLM horizon > saliency
    if face_regions:
        face_cx = sum(f.x + f.w / 2 for f in face_regions) / len(face_regions)
        face_cy = sum(f.y + f.h / 2 for f in face_regions) / len(face_regions)
        crop_cx = face_cx * scale
        crop_cy = face_cy * scale
    elif focus_x != 0.5 or focus_y != 0.5:
        crop_cx = focus_x * scaled_w
        crop_cy = focus_y * scaled_h
    elif horizon_y is not None and horizon_y <= 0.65:
        # LLM-provided horizon: position depends on where it falls.
        crop_cx = scaled_w / 2
        if horizon_y < 0.45:
            # Upper horizon: shift below to show boundary near crop top
            horizon_px = horizon_y * scaled_h
            crop_cy = horizon_px + (scaled_h - horizon_px) * 0.30
        else:
            # Mid horizon (45-65%): center on it to show both above and below.
            # Without knowing what's scenic (trees vs sand), centering is safest.
            crop_cy = horizon_y * scaled_h
    else:
        sal_cx, sal_cy = _saliency_center(scaled)
        crop_cx = sal_cx
        crop_cy = sal_cy

    x_off = int(crop_cx - out_w / 2)
    y_off = int(crop_cy - out_h / 2)
    x_off = max(0, min(x_off, scaled_w - out_w))
    y_off = max(0, min(y_off, scaled_h - out_h))

    # Ensure faces (with head padding) are fully within the crop window
    if face_regions:
        x_off, y_off = _adjust_crop_for_faces_limited(
            face_regions, scale, x_off, y_off, out_w, out_h, scaled_w, scaled_h,
            img_w=w, img_h=h,
        )

    canvas = scaled[y_off:y_off + out_h, x_off:x_off + out_w]

    cx = max(0.0, min(1.0, (crop_cx - x_off) / out_w))
    cy = max(0.0, min(1.0, (crop_cy - y_off) / out_h))

    return FitResult(
        canvas=canvas,
        x_offset=0, y_offset=0,
        fit_w=out_w, fit_h=out_h,
        scale=scale,
        content_center_x=cx, content_center_y=cy,
    )


def _subject_box_crop(
    image: np.ndarray,
    out_w: int,
    out_h: int,
    subject_box: tuple[float, float, float, float],
) -> FitResult:
    """Bounding-box-aware crop: zoom to fit the subject with padding."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = subject_box

    # Subject dimensions in pixels
    subj_w = (x2 - x1) * w
    subj_h = (y2 - y1) * h

    # Base scale: minimum to cover the canvas
    base_scale = max(out_w / w, out_h / h)

    # Scale that fits the subject into 80% of the canvas (10% padding each side)
    usable_w = out_w * 0.8
    usable_h = out_h * 0.8
    subject_fit_scale = min(usable_w / max(subj_w, 1), usable_h / max(subj_h, 1))

    # Clamp between base_scale and 3x base_scale
    scale = max(base_scale, min(subject_fit_scale, base_scale * 3.0))

    scaled_w = int(w * scale)
    scaled_h = int(h * scale)
    scaled = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

    # Center crop on the subject center
    subj_cx = ((x1 + x2) / 2) * scaled_w
    subj_cy = ((y1 + y2) / 2) * scaled_h

    x_off = int(subj_cx - out_w / 2)
    y_off = int(subj_cy - out_h / 2)
    x_off = max(0, min(x_off, scaled_w - out_w))
    y_off = max(0, min(y_off, scaled_h - out_h))

    canvas = scaled[y_off:y_off + out_h, x_off:x_off + out_w]

    cx = max(0.0, min(1.0, (subj_cx - x_off) / out_w))
    cy = max(0.0, min(1.0, (subj_cy - y_off) / out_h))

    return FitResult(
        canvas=canvas,
        x_offset=0, y_offset=0,
        fit_w=out_w, fit_h=out_h,
        scale=scale,
        content_center_x=cx, content_center_y=cy,
    )


def _blur_fill(
    image: np.ndarray,
    out_w: int,
    out_h: int,
    face_regions: list[FaceRegion] | None = None,
    focus_x: float = 0.5,
    focus_y: float = 0.5,
    subject_box: tuple[float, float, float, float] | None = None,
    people_centers: list[list[float]] | None = None,
) -> FitResult:
    """Zoom-to-cover and crop centered on the subject.

    Like crop_fill but uses the LLM-provided focus point (informed by
    HOG + face detection) to keep the person in frame.
    When subject_box is provided, uses bounding-box-aware zoom.
    """
    h, w = image.shape[:2]

    # Only use subject_box_crop when no faces — face-aware body estimation
    # is more reliable than LLM bounding boxes for person images.
    # Reject tiny boxes (area < 5% or height < 15%) — likely hallucinated by LLM.
    if subject_box is not None and not face_regions:
        sx1, sy1, sx2, sy2 = subject_box
        box_area = (sx2 - sx1) * (sy2 - sy1)
        box_height = sy2 - sy1
        if box_area >= 0.05 and box_height >= 0.15:
            return _subject_box_crop(image, out_w, out_h, subject_box)

    # Scale to COVER the canvas (same as crop mode)
    img_aspect = w / h
    target_aspect = out_w / out_h
    if img_aspect > target_aspect:
        scale = out_h / h
    else:
        scale = out_w / w

    scaled_w = int(w * scale)
    scaled_h = int(h * scale)
    scaled = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

    # Crop centered on the subject's focus point.
    # The proximity-based face filter in _adjust_crop_for_faces handles
    # showing heads correctly (pads 0.8× above face for forehead/hair).
    crop_cx = int(focus_x * scaled_w)
    crop_cy = int(focus_y * scaled_h)

    # When focus is default (0.5) but faces exist, estimate a better body
    # center from the face position. A person's body center is typically
    # ~1.0 face height below their face center — shift down to show more body
    # while preserving scenic background above.
    # Only use faces in the upper 45% of the image (where real heads are).
    # Skip when there's a single small face in the upper portion — likely a
    # false positive from clouds/rock texture, not a real person.
    skip_face_logic = False
    used_faces = False
    if face_regions and focus_x == 0.5 and focus_y == 0.5:
        filtered = _filter_faces(face_regions, w, h)
        # Single small face in upper 40% of image → likely false positive
        if (len(filtered) == 1
                and filtered[0].h < h * 0.10
                and (filtered[0].y + filtered[0].h / 2) < h * 0.40):
            skip_face_logic = True
        else:
            # Only use faces in upper portion (real heads) and near image center
            upper_faces = [f for f in filtered
                           if (f.y + f.h / 2) < h * 0.45
                           and abs((f.y + f.h / 2) - 0.5 * h) <= h * 0.30]
            if upper_faces:
                face_cy = sum(f.y + f.h / 2 for f in upper_faces) / len(upper_faces)
                avg_face_h = sum(f.h for f in upper_faces) / len(upper_faces)
                body_center_y = face_cy + avg_face_h * 1.0
                crop_cy = int(body_center_y * scale)
                used_faces = True

    # Fallback: use LLM people_centers when face-based positioning wasn't used
    # (e.g., all faces filtered by size cap, or no usable upper faces).
    if not used_faces and people_centers and len(people_centers) > 0:
        avg_y = sum(p[1] for p in people_centers) / len(people_centers)
        avg_x = sum(p[0] for p in people_centers) / len(people_centers)
        crop_cx = int(avg_x * scaled_w)
        crop_cy = int(avg_y * scaled_h)

    x_off = max(0, min(crop_cx - out_w // 2, scaled_w - out_w))
    y_off = max(0, min(crop_cy - out_h // 2, scaled_h - out_h))

    # Ensure faces (with head padding) are fully within the crop window
    if face_regions and not skip_face_logic:
        x_off, y_off = _adjust_crop_for_faces_limited(
            face_regions, scale, x_off, y_off, out_w, out_h, scaled_w, scaled_h,
            focus_y=focus_y, img_h=h, img_w=w,
        )

    canvas = scaled[y_off:y_off + out_h, x_off:x_off + out_w]

    cx = max(0.0, min(1.0, (crop_cx - x_off) / out_w))
    cy = max(0.0, min(1.0, (crop_cy - y_off) / out_h))

    return FitResult(
        canvas=canvas,
        x_offset=0, y_offset=0,
        fit_w=out_w, fit_h=out_h,
        scale=scale,
        content_center_x=cx, content_center_y=cy,
    )


def _filter_faces(
    face_regions: list[FaceRegion],
    img_w: int = 0,
    img_h: int = 0,
) -> list[FaceRegion]:
    """Filter out likely false-positive face detections.

    Keeps only faces whose area is at least 10% of the largest face.
    Additionally enforces a minimum absolute size of 3.5% and a maximum
    of 20% of the image's smaller dimension to reject both tiny texture
    false positives and huge cloud/canopy regions detected as faces.
    """
    if not face_regions:
        return face_regions

    faces = face_regions

    if img_w > 0 and img_h > 0:
        min_dim = min(img_w, img_h)
        # Minimum size: reject tiny texture detections (rocks, tree bark)
        min_face_px = int(min_dim * 0.035)
        # Maximum size: reject huge regions (cloud formations, canopy)
        max_face_px = int(min_dim * 0.20)
        faces = [f for f in faces
                 if min_face_px <= f.w <= max_face_px]
        if not faces:
            return faces

    # Relative filter: 10% of largest face area
    max_area = max(f.w * f.h for f in faces)
    threshold = max_area * 0.10
    return [f for f in faces if f.w * f.h >= threshold]


def _adjust_crop_for_faces(
    face_regions: list[FaceRegion],
    scale: float,
    x_off: int,
    y_off: int,
    out_w: int,
    out_h: int,
    scaled_w: int,
    scaled_h: int,
    focus_y: float | None = None,
    img_h: int | None = None,
    img_w: int | None = None,
) -> tuple[int, int]:
    """Adjust crop offset so faces (with head padding) stay in the crop window.

    Haar cascade face boxes often miss the top of the head (forehead, hair).
    We add padding above to prevent cutting off heads.

    Uses proximity-based filtering: when faces are spread far apart (>35% of
    image height), keeps only the face(s) closest to the body center (focus_y)
    to reject false positives from textures/trees.  The 35% threshold avoids
    filtering real group photos where people stand at different heights.
    """
    faces = _filter_faces(face_regions, img_w or 0, img_h or 0)
    if not faces:
        return x_off, y_off

    # Single-face distance check: if the only face is far from focus_y,
    # it's likely a false positive (tree texture, rock pattern, etc.)
    # Only apply when focus is at default (0.5) — when the LLM provides a
    # specific focus, the face may be a real person at a different location.
    if (focus_y is not None and img_h is not None and len(faces) == 1
            and abs(focus_y - 0.5) < 0.05):
        face_cy = faces[0].y + faces[0].h / 2
        dist = abs(face_cy - focus_y * img_h)
        if dist > img_h * 0.30:
            return x_off, y_off

    # Proximity-based face filter using face centroid (not LLM focus)
    if img_h is not None and len(faces) > 1:
        centroid_y = sum(f.y + f.h / 2 for f in faces) / len(faces)
        face_ys = [(f, abs((f.y + f.h / 2) - centroid_y)) for f in faces]
        y_min = min(f.y + f.h / 2 for f in faces)
        y_max = max(f.y + f.h / 2 for f in faces)
        span = y_max - y_min

        # Always keep the largest face — it's the strongest signal of a real
        # detection (false positives from textures tend to be smaller).
        largest_face = max(faces, key=lambda f: f.w * f.h)

        if span > img_h * 0.35:
            # Wide spread — likely includes false positives
            # Keep only the closest face(s) to centroid (± 15% tolerance)
            closest_dist = min(d for _, d in face_ys)
            tolerance = img_h * 0.15
            filtered = [f for f, d in face_ys if d <= closest_dist + tolerance]
            if largest_face not in filtered:
                filtered.append(largest_face)
            faces = filtered
        else:
            # Tight cluster — keep faces within 30% of image height from centroid
            threshold = img_h * 0.30
            nearby = [f for f, d in face_ys if d <= threshold]
            if nearby:
                faces = nearby

    if not faces:
        return x_off, y_off

    avg_face_h = sum(f.h for f in faces) / len(faces) * scale

    # Padded face bounds in scaled image coordinates
    # 1.0x face height above for forehead/hair, 0.3x below for chin margin
    pad_top = min(f.y for f in faces) * scale - avg_face_h * 1.0
    pad_bottom = max((f.y + f.h) for f in faces) * scale + avg_face_h * 0.3
    pad_left = min(f.x for f in faces) * scale - avg_face_h * 0.3
    pad_right = max((f.x + f.w) for f in faces) * scale + avg_face_h * 0.3

    # Vertical: prioritize TOP (heads) over bottom (feet)
    if pad_bottom - pad_top > out_h:
        # Faces span more than the crop height — just show heads
        y_off = max(0, int(pad_top))
    else:
        # All faces fit — try to include both top and bottom
        if pad_top < y_off:
            y_off = max(0, int(pad_top))
        if pad_bottom > y_off + out_h:
            y_off = min(scaled_h - out_h, int(pad_bottom - out_h))
            # Re-check top isn't lost after downward shift
            if pad_top < y_off:
                y_off = max(0, int(pad_top))

    # Horizontal: same priority (left over right)
    if pad_right - pad_left > out_w:
        center_x = (pad_left + pad_right) / 2
        x_off = max(0, min(int(center_x - out_w / 2), scaled_w - out_w))
    else:
        if pad_left < x_off:
            x_off = max(0, int(pad_left))
        if pad_right > x_off + out_w:
            x_off = min(scaled_w - out_w, int(pad_right - out_w))

    # Final clamp
    x_off = max(0, min(x_off, scaled_w - out_w))
    y_off = max(0, min(y_off, scaled_h - out_h))

    return x_off, y_off


def _adjust_crop_for_faces_limited(
    face_regions: list[FaceRegion],
    scale: float,
    x_off: int,
    y_off: int,
    out_w: int,
    out_h: int,
    scaled_w: int,
    scaled_h: int,
    focus_y: float | None = None,
    img_h: int | None = None,
    img_w: int | None = None,
) -> tuple[int, int]:
    """Like _adjust_crop_for_faces but limits vertical shift to 50% of crop height.

    Prevents false positive faces (clouds, textures) from pulling the crop
    into sky-only territory. The initial y_off from body estimation or focus
    is usually reasonable; face adjustment should refine it, not override it.
    """
    orig_y_off = y_off
    new_x_off, new_y_off = _adjust_crop_for_faces(
        face_regions, scale, x_off, y_off, out_w, out_h, scaled_w, scaled_h,
        focus_y=focus_y, img_h=img_h, img_w=img_w,
    )

    # Limit vertical shift to 50% of crop height
    max_shift = int(out_h * 0.50)
    if abs(new_y_off - orig_y_off) > max_shift:
        if new_y_off < orig_y_off:
            new_y_off = orig_y_off - max_shift
        else:
            new_y_off = orig_y_off + max_shift
        new_y_off = max(0, min(new_y_off, scaled_h - out_h))

    return new_x_off, new_y_off


def check_face_fits(
    img_w: int,
    img_h: int,
    face_regions: list[FaceRegion],
    out_w: int,
    out_h: int,
    scale_factor: float = 1.1,
) -> bool:
    """Check whether the largest face can be properly framed in the output crop.

    Returns False for close-up portraits where the face (with natural padding
    for forehead/hair above and chin margin below) exceeds 85% of the crop
    height at cover-scale. Such images can't be cropped without cutting off
    the face and should be excluded from candidates.
    """
    if not face_regions or img_w <= 0 or img_h <= 0:
        return True

    target_w = int(out_w * scale_factor)
    target_h = int(out_h * scale_factor)

    # Cover scale: minimum scale to fill the canvas
    scale = max(target_w / img_w, target_h / img_h)

    largest_face_h = max(f.h for f in face_regions)
    # Padded height: 0.8× above (forehead/hair) + 1.0× face + 0.3× below (chin)
    padded_h = largest_face_h * 2.1 * scale

    return padded_h <= target_h * 0.85


def _saliency_center(image: np.ndarray) -> tuple[float, float]:
    """Find the visual interest center using gradient magnitude + color contrast
    + horizontal transition detection.

    For landscape images, natural boundaries (horizon, treeline, water-land)
    are strong horizontal edges. We detect these and weight them heavily so
    crops follow natural transitions rather than centering on texture-rich
    areas like tree canopies.

    Returns (cx, cy) in pixel coords.
    """
    h, w = image.shape[:2]

    # Downscale for speed if large
    max_dim = 256
    if max(h, w) > max_dim:
        ds = max_dim / max(h, w)
        small = cv2.resize(image, (int(w * ds), int(h * ds)))
    else:
        small = image
        ds = 1.0

    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Edge density via Sobel gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_mag = grad_mag / (grad_mag.max() + 1e-6)

    # Color saturation from HSV
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float64) / 255.0

    # Horizon / natural boundary detection via color-based boundary analysis.
    # For each row, compare the average color above vs below to find where
    # the image content changes most (sky↔ground, water↔land, etc.).
    # Uses full color (not just brightness) to detect boundaries even when
    # brightness is similar (e.g., blue sky vs golden grass).
    row_color_means = small.astype(np.float64).mean(axis=1)  # (sh, 3)
    block = max(sh // 10, 5)
    boundary_raw = np.zeros(sh)
    for y in range(block, sh - block):
        upper = row_color_means[y - block:y].mean(axis=0)
        lower = row_color_means[y:y + block].mean(axis=0)
        boundary_raw[y] = np.linalg.norm(upper - lower)

    # Check if there's a clear horizon (peak >> mean)
    b_peak = boundary_raw.max()
    b_mean = boundary_raw[block:sh - block].mean() if sh > 2 * block else 0
    has_horizon = b_peak > 2 * b_mean and b_peak > 1.0
    horizon_row = int(np.argmax(boundary_raw)) if has_horizon else -1

    # Standard interest map: texture + saturation
    interest = 0.6 * grad_mag + 0.4 * saturation

    # Apply gentle center bias so equally interesting regions prefer center
    yy, xx = np.mgrid[0:sh, 0:sw]
    center_bias = 1.0 - 0.3 * np.sqrt(
        ((xx - sw / 2) / (sw / 2)) ** 2 + ((yy - sh / 2) / (sh / 2)) ** 2
    )
    center_bias = np.clip(center_bias, 0.5, 1.0)
    interest = interest * center_bias

    # Blur to get a smooth region rather than a single pixel
    kernel = max(sh, sw) // 4 | 1  # ensure odd
    interest = cv2.GaussianBlur(interest, (kernel, kernel), 0)

    # Find peak of 2D interest map for x (and fallback y)
    _, _, _, max_loc = cv2.minMaxLoc(interest)
    cx = max_loc[0] / ds

    if has_horizon:
        horizon_y = horizon_row / ds
        horizon_frac = horizon_y / h

        if horizon_frac > 0.65:
            # "Horizon" very low (>65%) — likely false positive from ground
            # texture changes (pine needles, shadows). Ignore it.
            has_horizon = False
        elif horizon_frac < 0.45:
            # Horizon in upper portion — shift below so the crop shows the
            # boundary near the top with mostly ground below.
            cy = horizon_y + (h - horizon_y) * 0.30
        else:
            # Horizon in mid area (45-65%) — center on it. Without knowing
            # what's above vs below (sky vs ground, trees vs beach), centering
            # is the safest choice.
            cy = horizon_y

    if not has_horizon:
        # No clear horizon — use vertical center as safe default.
        cy = h / 2

    # Sanity clamp: prevent extreme positions that produce bad crops.
    cx = max(w * 0.25, min(cx, w * 0.75))
    cy = max(h * 0.25, min(cy, h * 0.75))

    return cx, cy


def remap_face_regions(
    face_regions: list[FaceRegion],
    fit: FitResult,
) -> list[FaceRegion]:
    """Remap face regions from original image coords to canvas coords."""
    if not face_regions:
        return []
    return [
        FaceRegion(
            x=int(f.x * fit.scale) - fit.x_offset,
            y=int(f.y * fit.scale) - fit.y_offset,
            w=int(f.w * fit.scale),
            h=int(f.h * fit.scale),
        )
        for f in face_regions
    ]


# Keep old name for any remaining references
smart_crop = smart_fit
