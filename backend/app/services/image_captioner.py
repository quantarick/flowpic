"""Image captioning via Ollama moondream + person detection via OpenCV."""

import base64
import json
import logging
import re
import threading

logger = logging.getLogger(__name__)
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import httpx
import numpy as np

from app.config import settings
from app.models import FaceRegion, ImageCaption

BASE_PROMPT = (
    "Analyze this image and respond in exactly this format:\n"
    "CAPTION: (2-3 sentences describing scene, mood, visual qualities)\n"
    "ELEMENT: (one word: person, animal, landscape, architecture, object, or other)\n"
    "SUBJECT: x%, y% (center position of the main subject as percentages from top-left)\n"
    "BOUNDS: x1%, y1%, x2%, y2% (bounding box of the main subject — "
    "top-left and bottom-right corners as percentages)\n"
    "PERSON: YES or NO (is there any human figure — front, back, side, silhouette, any pose)\n"
    "PEOPLE: count, then center of each person as x%,y% (or 0 if no people)\n"
    "HORIZON: y% (vertical position of the horizon/skyline/treeline, or NONE if indoors/no visible horizon)\n"
    "Example response:\n"
    "CAPTION: A woman walks along the beach at sunset. The mood is serene and warm.\n"
    "ELEMENT: person\n"
    "SUBJECT: 40%, 50%\n"
    "BOUNDS: 25%, 20%, 55%, 80%\n"
    "PERSON: YES\n"
    "PEOPLE: 1, 40%,50%\n"
    "HORIZON: 65%"
)


class ImageCaptioner:
    def __init__(self, model: str | None = None):
        self._face_cascade = None
        self._hog_detector = None
        self._model = model or settings.ollama_model

    def _get_face_cascade(self) -> cv2.CascadeClassifier:
        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        return self._face_cascade

    def _get_hog_detector(self) -> cv2.HOGDescriptor:
        if self._hog_detector is None:
            self._hog_detector = cv2.HOGDescriptor()
            self._hog_detector.setSVMDetector(
                cv2.HOGDescriptor_getDefaultPeopleDetector()
            )
        return self._hog_detector

    def caption_image(self, image_path: Path) -> ImageCaption:
        """Generate caption for a single image using Ollama + detect faces/bodies."""
        # Check cache
        cache_dir = image_path.parent.parent / "captions"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{image_path.stem}.json"

        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return ImageCaption(**data)

        # Step 1: Run CV detectors FIRST
        logger.info(f"[{image_path.name}] Step 1: detecting faces...")
        face_regions = self._detect_faces(image_path)
        logger.info(f"[{image_path.name}] Step 1: detecting bodies...")
        body_rects = self._detect_bodies(image_path)
        logger.info(f"[{image_path.name}] Step 1 done: {len(face_regions)} faces, {len(body_rects)} bodies")

        # Read image dimensions and filter tiny false-positive faces
        img_for_dims = cv2.imread(str(image_path))
        img_height, img_width = (img_for_dims.shape[:2] if img_for_dims is not None
                                 else (0, 0))
        if face_regions and img_width > 0:
            min_dim = min(img_width, img_height)
            min_face_px = int(min_dim * 0.035)
            before = len(face_regions)
            face_regions = [f for f in face_regions if f.w >= min_face_px]
            if len(face_regions) < before:
                logger.info(f"[{image_path.name}] Filtered {before - len(face_regions)} "
                            f"tiny faces (< {min_face_px}px in {min_dim}px image)")

        # Step 2: Build prompt with CV hints for the LLM
        prompt = self._build_prompt(face_regions, body_rects, image_path)

        # Step 3: Call LLM with enriched prompt
        logger.info(f"[{image_path.name}] Step 3: calling Ollama...")
        info = self._call_ollama(image_path, prompt)
        logger.info(f"[{image_path.name}] Step 3 done: caption received")

        # Step 4: Combine signals — require CV confirmation for person
        # LLM hallucination rate is high; only trust person detection when
        # at least one CV detector (face or body) confirms.
        has_person = bool(face_regions) or bool(body_rects)

        # If no CV detection but caption mentions humans, the LLM likely
        # hallucinated (e.g., parroting the example response).  Replace
        # caption and clear all LLM-derived positioning to prevent bad crops.
        if not face_regions and not body_rects:
            _human_words = {
                "woman", "man", "person", "people", "child", "boy", "girl",
                "he ", "she ", "his ", "her ", "they ", "walks", "walking",
                "standing", "sitting", "holding",
            }
            caption_lower = info["caption"].lower()
            if any(w in caption_lower for w in _human_words):
                logger.info(f"[{image_path.name}] Caption mentions humans but no CV detection — replacing")
                info["caption"] = "A photograph with various visual elements."
                info["focus_x"] = 0.5
                info["focus_y"] = 0.5
                info["subject_x1"] = None
                info["subject_y1"] = None
                info["subject_x2"] = None
                info["subject_y2"] = None
                # People centers are from the same unreliable LLM call
                info["people_centers"] = None
                if info.get("element_type") == "person":
                    info["element_type"] = None

        # If CV found bodies but LLM focus is default, use body center as focus
        focus_x, focus_y = info["focus_x"], info["focus_y"]
        if body_rects and focus_x == 0.5 and focus_y == 0.5 and img_width > 0:
            cx = sum(r[0] + r[2] / 2 for r in body_rects) / len(body_rects)
            cy = sum(r[1] + r[3] / 2 for r in body_rects) / len(body_rects)
            focus_x = cx / img_width
            focus_y = cy / img_height

        # Step 5: Extract GPS and reverse geocode
        from app.services.gps_extractor import extract_gps, reverse_geocode
        gps = extract_gps(image_path)
        latitude = None
        longitude = None
        place_name = None
        if gps:
            latitude, longitude = gps
            place_name = reverse_geocode(latitude, longitude)
            logger.info(f"GPS for {image_path.name}: ({latitude:.4f}, {longitude:.4f}) → '{place_name}'")
        else:
            logger.info(f"No GPS data for {image_path.name}")

        result = ImageCaption(
            filename=image_path.name,
            caption=info["caption"],
            face_regions=face_regions,
            has_person=has_person,
            focus_x=focus_x,
            focus_y=focus_y,
            subject_x1=info.get("subject_x1"),
            subject_y1=info.get("subject_y1"),
            subject_x2=info.get("subject_x2"),
            subject_y2=info.get("subject_y2"),
            fit_mode="full" if has_person else "crop",
            element_type=info.get("element_type"),
            horizon_y=info.get("horizon_y"),
            people_centers=info.get("people_centers"),
            img_width=img_width if img_width > 0 else None,
            img_height=img_height if img_height > 0 else None,
            latitude=latitude,
            longitude=longitude,
            place_name=place_name,
        )

        # Cache result
        cache_file.write_text(result.model_dump_json(), encoding="utf-8")
        return result

    def caption_images(
        self,
        image_paths: list[Path],
        progress_callback=None,
    ) -> list[ImageCaption]:
        """Caption multiple images in parallel to maximize GPU utilization."""
        total = len(image_paths)
        results: dict[int, ImageCaption] = {}
        lock = threading.Lock()
        done_count = 0

        def _process(idx: int, path: Path) -> None:
            nonlocal done_count
            caption = self.caption_image(path)
            with lock:
                results[idx] = caption
                done_count += 1
                if progress_callback:
                    progress_callback(done_count, total)

        with ThreadPoolExecutor(max_workers=settings.caption_parallel) as pool:
            futures = [
                pool.submit(_process, i, p) for i, p in enumerate(image_paths)
            ]
            for f in as_completed(futures):
                f.result()  # raise any exceptions

        return [results[i] for i in range(total)]

    # --- Prompt building ---

    @staticmethod
    def _build_prompt(
        face_regions: list[FaceRegion],
        body_rects: list[tuple[int, int, int, int]],
        image_path: Path,
    ) -> str:
        """Build LLM prompt with CV detection coordinates as reference."""
        # Get image dimensions to convert pixel coords to percentages
        img = cv2.imread(str(image_path))
        if img is None:
            return BASE_PROMPT
        ih, iw = img.shape[:2]

        hints = []
        for i, f in enumerate(face_regions):
            cx = int((f.x + f.w / 2) / iw * 100)
            cy = int((f.y + f.h / 2) / ih * 100)
            w_pct = int(f.w / iw * 100)
            h_pct = int(f.h / ih * 100)
            hints.append(
                f"- Face #{i+1} detected at center ({cx}%, {cy}%), "
                f"size {w_pct}%x{h_pct}% of image"
            )

        for i, (x, y, bw, bh) in enumerate(body_rects):
            cx = int((x + bw / 2) / iw * 100)
            cy = int((y + bh / 2) / ih * 100)
            w_pct = int(bw / iw * 100)
            h_pct = int(bh / ih * 100)
            hints.append(
                f"- Human body #{i+1} detected at center ({cx}%, {cy}%), "
                f"size {w_pct}%x{h_pct}% of image (may be facing any direction)"
            )

        if hints:
            hint_block = (
                "\nCV detection results (use as reference for SUBJECT position):\n"
                + "\n".join(hints)
                + "\nUse these coordinates to pinpoint the main subject precisely. "
                "PERSON must be YES if any detections are listed above."
            )
            return BASE_PROMPT + hint_block
        return BASE_PROMPT

    # --- LLM ---

    # Models that only support /api/generate (not /api/chat)
    _generate_only_models = {"moondream"}

    _default_info = {
        "caption": "A photograph with various visual elements.",
        "focus_x": 0.5,
        "focus_y": 0.5,
        "has_person": False,
    }

    def _call_ollama(self, image_path: Path, prompt: str) -> dict:
        """Call Ollama API to get caption + subject position + person detection.

        Uses /api/chat for most models (qwen2.5vl, gemma3, llava, etc.)
        and /api/generate for moondream which only supports that endpoint.
        """
        image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        model_base = self._model.split(":")[0].lower()
        use_generate = model_base in self._generate_only_models

        try:
            with httpx.Client(timeout=settings.ollama_timeout) as client:
                if use_generate:
                    raw = self._ollama_generate(client, image_b64, prompt)
                else:
                    raw = self._ollama_chat(client, image_b64, prompt)
                return self._parse_response(raw)
        except httpx.TimeoutException:
            return dict(self._default_info)
        except Exception as e:
            return {**self._default_info, "caption": f"An image. (Caption unavailable: {e})"}

    def _ollama_generate(self, client: httpx.Client, image_b64: str, prompt: str) -> str:
        """Call /api/generate (moondream and similar models)."""
        payload = {
            "model": self._model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        resp = client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _ollama_chat(self, client: httpx.Client, image_b64: str, prompt: str) -> str:
        """Call /api/chat (qwen2.5vl, gemma3, llava, etc.)."""
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
        }
        resp = client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Parse structured LLM response into caption + subject info."""
        result = {
            "caption": raw,
            "focus_x": 0.5,
            "focus_y": 0.5,
            "has_person": False,
            "subject_x1": None,
            "subject_y1": None,
            "subject_x2": None,
            "subject_y2": None,
            "element_type": None,
            "horizon_y": None,
            "people_centers": None,
        }

        for line in raw.splitlines():
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("CAPTION:"):
                result["caption"] = stripped[len("CAPTION:"):].strip()

            elif upper.startswith("ELEMENT:"):
                val = stripped[len("ELEMENT:"):].strip().lower()
                if val in ("person", "animal", "landscape", "architecture", "object", "other"):
                    result["element_type"] = val

            elif upper.startswith("SUBJECT:"):
                nums = re.findall(r"(\d+(?:\.\d+)?)", stripped)
                if len(nums) >= 2:
                    result["focus_x"] = max(0.0, min(1.0, float(nums[0]) / 100))
                    result["focus_y"] = max(0.0, min(1.0, float(nums[1]) / 100))

            elif upper.startswith("BOUNDS:"):
                nums = re.findall(r"(\d+(?:\.\d+)?)", stripped)
                if len(nums) >= 4:
                    x1 = max(0.0, min(1.0, float(nums[0]) / 100))
                    y1 = max(0.0, min(1.0, float(nums[1]) / 100))
                    x2 = max(0.0, min(1.0, float(nums[2]) / 100))
                    y2 = max(0.0, min(1.0, float(nums[3]) / 100))
                    # Validate: x2>x1, y2>y1, min size >2%
                    if x2 > x1 and y2 > y1 and (x2 - x1) > 0.02 and (y2 - y1) > 0.02:
                        result["subject_x1"] = x1
                        result["subject_y1"] = y1
                        result["subject_x2"] = x2
                        result["subject_y2"] = y2
                        # Override focus to box center for consistency
                        result["focus_x"] = (x1 + x2) / 2
                        result["focus_y"] = (y1 + y2) / 2

            elif upper.startswith("PERSON:"):
                result["has_person"] = "YES" in upper

            elif upper.startswith("PEOPLE:"):
                nums = re.findall(r"(\d+(?:\.\d+)?)", stripped)
                if nums:
                    count = int(nums[0])
                    if count > 0 and len(nums) >= 1 + count * 2:
                        centers = []
                        for i in range(count):
                            px = max(0.0, min(1.0, float(nums[1 + i * 2]) / 100))
                            py = max(0.0, min(1.0, float(nums[2 + i * 2]) / 100))
                            centers.append([px, py])
                        result["people_centers"] = centers

            elif upper.startswith("HORIZON:"):
                if "NONE" not in upper:
                    nums = re.findall(r"(\d+(?:\.\d+)?)", stripped)
                    if nums:
                        hy = max(0.0, min(1.0, float(nums[0]) / 100))
                        result["horizon_y"] = hy

        return result

    # --- CV detectors ---

    def _detect_faces(self, image_path: Path) -> list[FaceRegion]:
        """Detect faces using OpenCV Haar cascade."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return []

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cascade = self._get_face_cascade()
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if isinstance(faces, np.ndarray) and len(faces) > 0:
                return [
                    FaceRegion(x=int(x), y=int(y), w=int(w), h=int(h))
                    for (x, y, w, h) in faces
                ]
        except Exception:
            pass
        return []

    def _detect_bodies(self, image_path: Path) -> list[tuple[int, int, int, int]]:
        """Detect human bodies from any angle using HOG pedestrian detector.

        Returns list of (x, y, w, h) in original image coordinates.
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return []

            h, w = img.shape[:2]
            max_dim = 512
            scale = 1.0
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            hog = self._get_hog_detector()
            rects, weights = hog.detectMultiScale(
                img, winStride=(8, 8), padding=(4, 4), scale=1.05
            )

            if len(rects) == 0:
                return []

            # Map back to original image coordinates
            return [
                (int(x / scale), int(y / scale), int(bw / scale), int(bh / scale))
                for (x, y, bw, bh) in rects
            ]
        except Exception:
            return []
