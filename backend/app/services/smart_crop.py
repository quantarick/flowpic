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
) -> FitResult:
    """Fit image to canvas using LLM-decided mode.

    fit_mode="crop": scale to cover, crop centered on interest point.
    fit_mode="full": blurred background fill to preserve full subject.
    """
    h, w = image.shape[:2]
    out_w = int(target_w * scale_factor)
    out_h = int(target_h * scale_factor)

    if fit_mode == "full":
        return _blur_fill(image, out_w, out_h, focus_x, focus_y)

    return _crop_fill(image, out_w, out_h, face_regions, focus_x, focus_y)


def _crop_fill(
    image: np.ndarray,
    out_w: int,
    out_h: int,
    face_regions: list[FaceRegion] | None,
    focus_x: float,
    focus_y: float,
) -> FitResult:
    """Crop-to-fill: scale to cover, crop centered on interest point."""
    h, w = image.shape[:2]
    img_aspect = w / h
    target_aspect = out_w / out_h

    if img_aspect > target_aspect:
        scale = out_h / h
    else:
        scale = out_w / w

    scaled_w = int(w * scale)
    scaled_h = int(h * scale)
    scaled = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

    # Determine crop center: faces > LLM focus > saliency
    if face_regions:
        face_cx = sum(f.x + f.w / 2 for f in face_regions) / len(face_regions)
        face_cy = sum(f.y + f.h / 2 for f in face_regions) / len(face_regions)
        crop_cx = face_cx * scale
        crop_cy = face_cy * scale
    elif focus_x != 0.5 or focus_y != 0.5:
        crop_cx = focus_x * scaled_w
        crop_cy = focus_y * scaled_h
    else:
        sal_cx, sal_cy = _saliency_center(scaled)
        crop_cx = sal_cx
        crop_cy = sal_cy

    x_off = int(crop_cx - out_w / 2)
    y_off = int(crop_cy - out_h / 2)
    x_off = max(0, min(x_off, scaled_w - out_w))
    y_off = max(0, min(y_off, scaled_h - out_h))

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


def _blur_fill(
    image: np.ndarray,
    out_w: int,
    out_h: int,
    focus_x: float = 0.5,
    focus_y: float = 0.5,
) -> FitResult:
    """Zoom-to-cover and crop centered on the subject.

    Like crop_fill but uses the LLM-provided focus point (informed by
    HOG + face detection) to keep the person in frame. No padding.
    """
    h, w = image.shape[:2]

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

    # Crop centered on the subject's focus point
    crop_cx = int(focus_x * scaled_w)
    crop_cy = int(focus_y * scaled_h)

    x_off = max(0, min(crop_cx - out_w // 2, scaled_w - out_w))
    y_off = max(0, min(crop_cy - out_h // 2, scaled_h - out_h))

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


def _saliency_center(image: np.ndarray) -> tuple[float, float]:
    """Find the visual interest center using gradient magnitude + color contrast.

    Combines edge density (detail-rich areas) and color saturation
    into a weighted interest map. Returns (cx, cy) in pixel coords.
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

    # Combined interest map
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

    # Find peak
    _, _, _, max_loc = cv2.minMaxLoc(interest)
    cx = max_loc[0] / ds
    cy = max_loc[1] / ds

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
