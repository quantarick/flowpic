"""Per-segment lyric emotion analysis via Ollama text model (Qwen2.5)."""

import json
import logging
from pathlib import Path

import httpx

from app.config import settings
from app.models import (
    AudioFeatures,
    LyricEmotion,
    LyricsResult,
    SegmentLyrics,
    TranscribedWord,
)

logger = logging.getLogger(__name__)

LYRIC_EMOTION_PROMPT = """\
Analyze the emotional content of these song lyrics.

Lyrics:
{lyrics}

Respond in JSON format only:
{{
  "theme": "a short phrase describing the main theme (e.g., longing for lost love, celebration of freedom)",
  "mood_keywords": ["keyword1", "keyword2", "keyword3"],
  "mood_description": "One sentence describing the emotional tone conveyed by these lyrics"
}}"""


class LyricEmotionAnalyzer:
    def analyze(
        self,
        lyrics_result: LyricsResult,
        audio_features: AudioFeatures,
    ) -> list[LyricEmotion]:
        """Map transcribed words to audio segments and analyze emotion per segment."""
        segment_lyrics = self._map_words_to_segments(
            lyrics_result.words, audio_features,
        )

        results: list[LyricEmotion] = []
        for seg_lyric in segment_lyrics:
            if seg_lyric.word_count < 2:
                continue
            emotion = self._analyze_segment(seg_lyric)
            if emotion:
                results.append(emotion)

        logger.info(f"Analyzed lyrics for {len(results)} segments")
        return results

    def _map_words_to_segments(
        self,
        words: list[TranscribedWord],
        audio_features: AudioFeatures,
    ) -> list[SegmentLyrics]:
        """Map timestamped words to audio segments by time overlap."""
        segment_lyrics: list[SegmentLyrics] = []

        for i, seg in enumerate(audio_features.segments):
            seg_words = [
                w for w in words
                if w.end > seg.start and w.start < seg.end
            ]
            text = " ".join(w.word for w in seg_words)
            segment_lyrics.append(SegmentLyrics(
                segment_index=i,
                start=seg.start,
                end=seg.end,
                text=text,
                word_count=len(seg_words),
            ))

        return segment_lyrics

    def _analyze_segment(self, seg_lyric: SegmentLyrics) -> LyricEmotion | None:
        """Call Ollama text model for a single segment's lyrics."""
        prompt = LYRIC_EMOTION_PROMPT.format(lyrics=seg_lyric.text)

        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            with httpx.Client(timeout=settings.ollama_timeout) as client:
                resp = client.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")
        except Exception as e:
            logger.warning(f"Ollama lyric analysis failed for segment {seg_lyric.segment_index}: {e}")
            return None

        return self._parse_response(raw, seg_lyric.segment_index)

    def _parse_response(self, raw: str, segment_index: int) -> LyricEmotion | None:
        """Parse JSON response from Ollama."""
        # Try to extract JSON from the response
        try:
            # Handle potential markdown code blocks
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)
            return LyricEmotion(
                segment_index=segment_index,
                theme=data.get("theme", ""),
                mood_keywords=data.get("mood_keywords", []),
                mood_description=data.get("mood_description", ""),
            )
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to parse lyric emotion JSON for segment {segment_index}: {e}")
            return None
