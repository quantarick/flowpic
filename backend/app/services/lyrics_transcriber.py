"""Lyrics transcription from vocals using OpenAI Whisper."""

import gc
import logging
from pathlib import Path

import torch

from app.config import settings
from app.models import LyricsResult, TranscribedWord

logger = logging.getLogger(__name__)


class LyricsTranscriber:
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        import whisper

        self._model = whisper.load_model(
            settings.whisper_model_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        logger.info(f"Whisper '{settings.whisper_model_size}' loaded")

    def transcribe(self, vocals_path: Path) -> LyricsResult:
        """Transcribe vocals to text with word-level timestamps."""
        self._load_model()

        result = self._model.transcribe(
            str(vocals_path),
            word_timestamps=True,
            task="transcribe",
        )

        words: list[TranscribedWord] = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                words.append(TranscribedWord(
                    word=w["word"].strip(),
                    start=w["start"],
                    end=w["end"],
                    probability=w.get("probability", 0.0),
                ))

        full_text = result.get("text", "").strip()
        language = result.get("language", "unknown")

        # Quality gate: avg word probability > 0.3 and word count > 3
        if not self._passes_quality_gate(words):
            logger.info(
                f"Transcription failed quality gate "
                f"(words={len(words)}, "
                f"avg_prob={self._avg_probability(words):.2f})"
            )
            return LyricsResult(
                text="", words=[], language=language, has_vocals=False,
            )

        logger.info(
            f"Transcribed {len(words)} words, language={language}, "
            f"avg_prob={self._avg_probability(words):.2f}"
        )
        return LyricsResult(
            text=full_text, words=words, language=language, has_vocals=True,
        )

    def _passes_quality_gate(self, words: list[TranscribedWord]) -> bool:
        if len(words) <= 3:
            return False
        return self._avg_probability(words) > 0.3

    @staticmethod
    def _avg_probability(words: list[TranscribedWord]) -> float:
        if not words:
            return 0.0
        return sum(w.probability for w in words) / len(words)

    def unload(self):
        """Free GPU memory before loading next model."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Whisper model unloaded")
