"""CLIP encoder for multimodal image-text matching."""

import gc
import logging
from pathlib import Path

import numpy as np
import torch

from app.config import settings

logger = logging.getLogger(__name__)


class CLIPEncoder:
    def __init__(self, model_name: str | None = None):
        self._model = None
        self._preprocess = None
        self._tokenize = None
        self._device = None
        self._model_name = model_name or settings.clip_model

    def _load_model(self):
        if self._model is not None:
            return
        import clip

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._preprocess = clip.load(
            self._model_name, device=self._device
        )
        self._tokenize = clip.tokenize
        self._model.eval()
        logger.info(f"CLIP {self._model_name} loaded on {self._device}")

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        """Encode images to CLIP embeddings. Returns (N, 512) normalized array."""
        self._load_model()
        from PIL import Image

        images = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(self._preprocess(img))
            except Exception as e:
                logger.warning(f"Failed to load image {p}: {e}")
                # Use a blank image as fallback
                img = Image.new("RGB", (224, 224))
                images.append(self._preprocess(img))

        batch = torch.stack(images).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def encode_images_batch(
        self, paths: list[Path], batch_size: int = 16
    ) -> np.ndarray:
        """Encode images in batches to limit VRAM usage. Returns (N, 512)."""
        all_features = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            features = self.encode_images(batch_paths)
            all_features.append(features)
        return np.concatenate(all_features, axis=0)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts to CLIP embeddings. Returns (N, 512) normalized array.

        CLIP tokenizer has a 77-token limit; long texts are truncated.
        """
        self._load_model()
        tokens = self._tokenize(texts, truncate=True).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def unload(self):
        """Free VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
            self._preprocess = None
            self._tokenize = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("CLIP model unloaded")
