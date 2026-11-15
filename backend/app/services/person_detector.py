"""Tell whether a photo has people in it.

Used by the "exclude photos with people" option. The face and body detectors
in the captioner are not good enough for this: on landscape photos they fire
on leaves and clouds and miss a person standing in the distance.

This uses a COCO-trained Faster R-CNN from torchvision. It looks over the
whole photo and then over a grid of overlapping tiles, so that people who are
small in the frame are still found.

What it can and cannot do, measured on real landscape photos:

- A person who is clearly a person scores 0.9 or more, down to about 20
  pixels tall as the model sees them (roughly 1% of the picture height).
- Below that size a swan, a rock or a chimney scores just as high as a real
  figure, and neither a closer second look nor a CLIP check tells them apart.
  So detections that small are ignored: a sailor in a far-off dinghy stays in.
- Between 0.6 and 0.9 are branches, ducks and buoys as often as people, which
  is why the default threshold is 0.9.

The model is loaded on first use and should be unloaded afterwards: it shares
the GPU with the other pipeline models.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

MODEL_NAME = "fasterrcnn_resnet50_fpn_v2"
# Bump when the method changes, so cached answers from the old method are not reused.
CACHE_VERSION = 2

_PERSON = 1                # COCO label
_MIN_SEEN_HEIGHT = 16      # pixels, at the size the model works at; smaller is a guess
_TILE_OVERLAP = 0.2
_TILE_MIN_SIDE = 1000      # photos smaller than this are not tiled
# The model resizes what it is given so the short side is 800, the long side at most 1333.
_MODEL_SHORT, _MODEL_LONG = 800, 1333


@dataclass(frozen=True)
class Person:
    """A person found in a photo. The box is in the photo's pixels."""

    score: float
    x0: float
    y0: float
    x1: float
    y1: float


def tile_grid(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles covering a photo, as (x0, y0, x1, y1). Empty for a small photo."""
    if max(width, height) < _TILE_MIN_SIDE:
        return []
    cols, rows = (2, 3) if height >= width else (3, 2)
    tile_w = int(width / (cols - (cols - 1) * _TILE_OVERLAP))
    tile_h = int(height / (rows - (rows - 1) * _TILE_OVERLAP))
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = round(c * (width - tile_w) / (cols - 1))
            y0 = round(r * (height - tile_h) / (rows - 1))
            tiles.append((x0, y0, x0 + tile_w, y0 + tile_h))
    return tiles


def seen_scale(width: int, height: int) -> float:
    """How much the model shrinks or enlarges a region of this size before looking at it."""
    return min(_MODEL_SHORT / min(width, height), _MODEL_LONG / max(width, height))


class PersonDetector:
    """Finds people in photos. Not thread-safe: use one photo at a time."""

    def __init__(self, score: Optional[float] = None, device: Optional[str] = None):
        self.score = settings.person_score if score is None else score
        self._device_name = device
        self._model = None
        self._device = None

    # --- model lifecycle ---

    def _load(self):
        if self._model is not None:
            return
        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        self._device = torch.device(
            self._device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Loading person detector ({MODEL_NAME}) on {self._device}")
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_score_thresh=0.3,
        )
        self._model = model.eval().to(self._device)

    def unload(self):
        """Free the model and its GPU memory."""
        if self._model is None:
            return
        import torch

        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Person detector unloaded")

    # --- detection ---

    def _run(self, rgb: np.ndarray) -> list[tuple[int, float, tuple[float, float, float, float]]]:
        """Every object the model sees in `rgb`, as (label, score, box)."""
        import torch

        self._load()
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(self._device)
        tensor = tensor.permute(2, 0, 1).float().div_(255)
        with torch.inference_mode():
            out = self._model([tensor])[0]
        return [
            (int(label), float(score), tuple(box))
            for label, score, box in zip(out["labels"].tolist(), out["scores"].tolist(), out["boxes"].tolist())
        ]

    def _people_in(self, rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> list[Person]:
        """People in one region of the photo that are sure enough and large enough to trust."""
        scale = seen_scale(x1 - x0, y1 - y0)
        found = []
        for label, score, box in self._run(rgb[y0:y1, x0:x1]):
            if label != _PERSON or score < self.score:
                continue
            if (box[3] - box[1]) * scale < _MIN_SEEN_HEIGHT:
                continue
            found.append(Person(score, box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0))
        return found

    def find_people(self, image: np.ndarray, stop_at_first: bool = True) -> list[Person]:
        """People in a BGR image, as loaded by OpenCV.

        With `stop_at_first` the search ends at the first region that has a
        person, which is all that "does this photo have people" needs. The
        whole photo is looked at first, so a photo of someone ends there.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        people: list[Person] = []
        for region in [(0, 0, w, h), *tile_grid(w, h)]:
            people += self._people_in(rgb, *region)
            if people and stop_at_first:
                break
        return people

    def has_people(self, image_path: Path) -> bool:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path.name}")
        return bool(self.find_people(image))


# --- filtering a project's photos, with a cache ---

_cache_lock = threading.Lock()


def _cache_file(image_path: Path) -> Path:
    return image_path.parent.parent / "people" / f"{image_path.stem}.json"


def _read_cache(image_path: Path, score: float) -> Optional[bool]:
    path = _cache_file(image_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("version") != CACHE_VERSION or data.get("model") != MODEL_NAME:
        return None
    if data.get("score") != score:
        return None
    return bool(data.get("has_people"))


def _write_cache(image_path: Path, score: float, has_people: bool) -> None:
    path = _cache_file(image_path)
    with _cache_lock:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({
            "filename": image_path.name,
            "has_people": has_people,
            "model": MODEL_NAME,
            "score": score,
            "version": CACHE_VERSION,
        }), encoding="utf-8")


def split_by_people(
    image_paths: list[Path],
    detector: Optional[PersonDetector] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[list[Path], list[Path]]:
    """Split photos into (without people, with people).

    Answers are cached next to the project's photos, in `people/`, so a second
    run does not load the model at all. A photo that cannot be read counts as
    having no people; later stages deal with unreadable files.
    """
    detector = detector or PersonDetector()
    without: list[Path] = []
    with_people: list[Path] = []
    total = len(image_paths)
    try:
        for done, path in enumerate(image_paths, start=1):
            answer = _read_cache(path, detector.score)
            if answer is None:
                try:
                    answer = detector.has_people(path)
                except ValueError as e:
                    logger.warning(f"People check skipped: {e}")
                    answer = False
                else:
                    _write_cache(path, detector.score, answer)
            (with_people if answer else without).append(path)
            if progress_callback:
                progress_callback(done, total)
    finally:
        detector.unload()
    logger.info(
        f"People check: {len(with_people)} of {total} photos have people"
        + (f": {[p.name for p in with_people]}" if with_people else "")
    )
    return without, with_people
