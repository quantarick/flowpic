"""Image preparation for hosted providers."""

from __future__ import annotations

import logging

from app.llm.base import ImageInput

logger = logging.getLogger(__name__)


def normalize_for_upload(image: ImageInput, max_side: int, quality: int = 85) -> ImageInput:
    """Re-encode an image as a JPEG whose longer side is at most `max_side`.

    Hosted APIs cap request size, and FlowPic accepts uploads up to 50 MB, so
    photos are shrunk before they leave the machine. Decoding applies the EXIF
    rotation, which keeps the pixels in the same frame the OpenCV detectors
    and the crop code use. Re-encoding drops all metadata, GPS included.

    Returns the input unchanged if it cannot be decoded.
    """
    import cv2
    import numpy as np

    arr = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        logger.warning("Could not decode image for upload; sending original bytes")
        return image

    h, w = arr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        arr = cv2.resize(
            arr,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        logger.warning("Could not re-encode image for upload; sending original bytes")
        return image
    return ImageInput(data=buf.tobytes(), mime_type="image/jpeg")
