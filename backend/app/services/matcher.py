"""Semantic matching: music mood ↔ image captions via sentence-transformer embeddings."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.models import AudioFeatures, ImageCaption, MatchResult, SegmentEmotion


class SemanticMatcher:
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def match(
        self,
        segment_emotions: list[SegmentEmotion],
        image_captions: list[ImageCaption],
        audio_features: AudioFeatures,
    ) -> list[MatchResult]:
        """Match music segments to images using semantic similarity."""
        self._load_model()

        n_segments = len(segment_emotions)
        n_images = len(image_captions)

        if n_images == 0 or n_segments == 0:
            return []

        # Handle segment/image count mismatch
        if n_segments > n_images:
            segment_emotions = self._merge_segments(
                segment_emotions, audio_features, target_count=n_images
            )
            n_segments = len(segment_emotions)

        # Encode mood descriptions and captions
        mood_texts = [se.mood_description for se in segment_emotions]
        caption_texts = [ic.caption for ic in image_captions]

        mood_embeddings = self._model.encode(mood_texts, normalize_embeddings=True)
        caption_embeddings = self._model.encode(caption_texts, normalize_embeddings=True)

        # Cosine similarity matrix (embeddings already normalized)
        similarity = mood_embeddings @ caption_embeddings.T  # (n_segments, n_images)

        # Select best subset if more images than segments
        if n_images > n_segments:
            # Use Hungarian on full matrix, then take assigned columns
            cost = -similarity
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            # Square matrix: one-to-one
            cost = -similarity
            row_ind, col_ind = linear_sum_assignment(cost)

        results: list[MatchResult] = []
        for r, c in zip(row_ind, col_ind):
            results.append(
                MatchResult(
                    segment_index=segment_emotions[r].segment_index,
                    image_filename=image_captions[c].filename,
                    similarity_score=float(similarity[r, c]),
                )
            )

        # Sort by segment order
        results.sort(key=lambda m: m.segment_index)
        return results

    def _merge_segments(
        self,
        segments: list[SegmentEmotion],
        audio_features: AudioFeatures,
        target_count: int,
    ) -> list[SegmentEmotion]:
        """Merge adjacent low-energy segments until count matches target."""
        if len(segments) <= target_count:
            return segments

        # Build energy map from audio features
        energy_map = {
            i: audio_features.segments[i].rms_energy
            for i in range(len(audio_features.segments))
            if i < len(segments)
        }

        merged = list(segments)
        while len(merged) > target_count:
            # Find pair with lowest combined energy to merge
            min_energy = float("inf")
            merge_idx = 0
            for i in range(len(merged) - 1):
                e1 = energy_map.get(merged[i].segment_index, 0)
                e2 = energy_map.get(merged[i + 1].segment_index, 0)
                combined = e1 + e2
                if combined < min_energy:
                    min_energy = combined
                    merge_idx = i

            # Merge: keep first segment's index, extend end time, average emotions
            a = merged[merge_idx]
            b = merged[merge_idx + 1]
            merged_seg = SegmentEmotion(
                segment_index=a.segment_index,
                start=a.start,
                end=b.end,
                valence=round((a.valence + b.valence) / 2, 2),
                arousal=round((a.arousal + b.arousal) / 2, 2),
                mood_description=a.mood_description,  # Keep dominant mood
            )
            merged[merge_idx] = merged_seg
            del merged[merge_idx + 1]

        return merged
