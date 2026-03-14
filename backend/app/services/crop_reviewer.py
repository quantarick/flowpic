"""Post-crop AI review for person images via Ollama vision model.

After smart_crop produces a cropped canvas, this service sends it to the
vision model to check whether the person is fully visible. If not, it
adjusts the ImageCaption fields (switch to blur-fill, expand box, shift
focus) and re-crops once. Debug images are saved to the output folder.
"""

import base64
import logging
from pathlib import Path
from typing import Callable

import cv2
import httpx
import numpy as np

from app.config import settings
from app.models import ImageCaption, MatchResult
from app.services.smart_crop import get_output_resolution, smart_fit, AspectRatio, Quality

logger = logging.getLogger(__name__)

REVIEW_PROMPT = (
    "Look at this cropped image. Is the main person fully visible — head to toe, "
    "not cut off at any edge?\n\n"
    "Respond in exactly this format:\n"
    "COMPLETE: YES or NO\n"
    "CUT_OFF: none, top, bottom, left, right (which edge cuts the person, if any)"
)

# Models that only support /api/generate (not /api/chat)
_generate_only_models = {"moondream"}


def _crop_image(
    img: np.ndarray,
    caption: ImageCaption,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Produce the cropped canvas for a given caption's parameters."""
    subject_box = None
    if caption.subject_x1 is not None:
        subject_box = (
            caption.subject_x1, caption.subject_y1,
            caption.subject_x2, caption.subject_y2,
        )
    fit = smart_fit(
        img, out_w, out_h,
        face_regions=caption.face_regions or None,
        focus_x=caption.focus_x,
        focus_y=caption.focus_y,
        scale_factor=1.0,
        fit_mode=caption.fit_mode,
        subject_box=subject_box,
    )
    return fit.canvas


class CropReviewer:
    def __init__(self, model: str | None = None):
        self._model = model or settings.ollama_model

    def review_crops(
        self,
        matches: list[MatchResult],
        image_captions: list[ImageCaption],
        images_dir: Path,
        aspect_ratio: AspectRatio,
        quality: Quality,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[ImageCaption]:
        """Review cropped person images and adjust captions if person is cut off.

        Saves debug images (before/after) to {project}/output/crop_debug/.
        """
        out_w, out_h = get_output_resolution(aspect_ratio, quality)

        # Debug output directory
        debug_dir = images_dir.parent / "output" / "crop_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Build lookup: filename -> caption index
        caption_idx = {ic.filename: i for i, ic in enumerate(image_captions)}

        # Collect person image filenames that appear in matches
        person_filenames = set()
        for m in matches:
            idx = caption_idx.get(m.image_filename)
            if idx is not None and image_captions[idx].has_person:
                person_filenames.add(m.image_filename)

        person_list = sorted(person_filenames)
        total = len(person_list)
        if total == 0:
            logger.info("Crop review: no person images to review")
            return image_captions

        logger.info(f"Crop review: reviewing {total} person images → debug at {debug_dir}")

        for done, filename in enumerate(person_list):
            idx = caption_idx[filename]
            caption = image_captions[idx]
            stem = Path(filename).stem

            img_path = images_dir / filename
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Crop review: could not read {filename}")
                if progress_callback:
                    progress_callback(done + 1, total)
                continue

            # Save original for comparison
            cv2.imwrite(str(debug_dir / f"{stem}_original.jpg"), img)

            # --- Initial crop ---
            canvas_before = _crop_image(img, caption, out_w, out_h)
            cv2.imwrite(str(debug_dir / f"{stem}_before.jpg"), canvas_before)

            # Ask vision model
            canvas_b64 = self._encode_canvas(canvas_before)
            feedback = self._ask_vision(canvas_b64)
            logger.info(
                f"Crop review [{filename}]: complete={feedback['complete']}, "
                f"cut_off={feedback['cut_off']}, fit_mode={caption.fit_mode}"
            )

            if not feedback["complete"] and feedback["cut_off"] != "none":
                # Adjust and re-crop
                caption = self._adjust_caption(caption, feedback["cut_off"])
                image_captions[idx] = caption

                canvas_after = _crop_image(img, caption, out_w, out_h)
                cv2.imwrite(str(debug_dir / f"{stem}_after.jpg"), canvas_after)
                logger.info(
                    f"Crop review [{filename}]: adjusted for {feedback['cut_off']} "
                    f"cut-off → fit_mode={caption.fit_mode}"
                )
            else:
                logger.info(f"Crop review [{filename}]: OK, no adjustment needed")

            if progress_callback:
                progress_callback(done + 1, total)

        return image_captions

    @staticmethod
    def _encode_canvas(canvas: np.ndarray) -> str:
        """Encode numpy canvas to base64 JPEG string."""
        _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _ask_vision(self, image_b64: str) -> dict:
        """Call Ollama vision model with review prompt, return parsed feedback."""
        try:
            with httpx.Client(timeout=settings.ollama_timeout) as client:
                model_base = self._model.split(":")[0].lower()
                if model_base in _generate_only_models:
                    raw = self._ollama_generate(client, image_b64)
                else:
                    raw = self._ollama_chat(client, image_b64)
                return self._parse_review(raw)
        except Exception as e:
            logger.warning(f"Crop review vision call failed: {e}")
            return {"complete": True, "cut_off": "none"}

    def _ollama_generate(self, client: httpx.Client, image_b64: str) -> str:
        payload = {
            "model": self._model,
            "prompt": REVIEW_PROMPT,
            "images": [image_b64],
            "stream": False,
        }
        resp = client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _ollama_chat(self, client: httpx.Client, image_b64: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": REVIEW_PROMPT,
                    "images": [image_b64],
                }
            ],
            "stream": False,
        }
        resp = client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    @staticmethod
    def _parse_review(raw: str) -> dict:
        """Parse vision model response into structured feedback."""
        result = {"complete": True, "cut_off": "none"}

        for line in raw.splitlines():
            upper = line.strip().upper()
            if upper.startswith("COMPLETE:"):
                result["complete"] = "YES" in upper
            elif upper.startswith("CUT_OFF:"):
                text = line.strip()[len("CUT_OFF:"):].strip().lower()
                for edge in ("top", "bottom", "left", "right"):
                    if edge in text:
                        result["cut_off"] = edge
                        break

        return result

    @staticmethod
    def _adjust_caption(caption: ImageCaption, cut_off: str) -> ImageCaption:
        """Adjust caption fields based on which edge cuts off the person.

        Expands subject_box by 15% toward the cut edge so _subject_box_crop
        zooms out enough to include the missing part.  Also shifts the focus
        point by 0.12 as a fallback for images without a subject box.
        """
        data = caption.model_dump()
        has_box = data["subject_x1"] is not None

        if has_box:
            if cut_off == "top":
                data["subject_y1"] = max(0.0, data["subject_y1"] - 0.15)
            elif cut_off == "bottom":
                data["subject_y2"] = min(1.0, data["subject_y2"] + 0.15)
            elif cut_off == "left":
                data["subject_x1"] = max(0.0, data["subject_x1"] - 0.15)
            elif cut_off == "right":
                data["subject_x2"] = min(1.0, data["subject_x2"] + 0.15)

        # Shift focus point away from the cut edge
        if cut_off == "top":
            data["focus_y"] = min(1.0, data["focus_y"] + 0.12)
        elif cut_off == "bottom":
            data["focus_y"] = max(0.0, data["focus_y"] - 0.12)
        elif cut_off == "left":
            data["focus_x"] = min(1.0, data["focus_x"] + 0.12)
        elif cut_off == "right":
            data["focus_x"] = max(0.0, data["focus_x"] - 0.12)

        return ImageCaption(**data)
