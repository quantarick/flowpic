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
    ) -> tuple[list[MatchResult], list[SegmentEmotion]]:
        """Match music segments to images using semantic similarity.

        Returns (matches, merged_segment_emotions) — the caller must use
        the returned emotions for video timing, as segments may have been
        merged to fit the available image count.
        """
        self._load_model()

        n_segments = len(segment_emotions)
        n_images = len(image_captions)

        if n_images == 0 or n_segments == 0:
            return [], segment_emotions

        # Handle segment/image count mismatch
        if n_segments > n_images:
            segment_emotions = self._merge_segments(
                segment_emotions, audio_features, target_count=n_images
            )
        elif n_images > n_segments:
            # More images than segments: split longest segments to create
            # more slots so we can use more images.
            segment_emotions = self._split_segments(
                segment_emotions, target_count=n_images
            )
            n_segments = len(segment_emotions)
        else:
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
        return results, segment_emotions

    def _merge_segments(
        self,
        segments: list[SegmentEmotion],
        audio_features: AudioFeatures,
        target_count: int,
    ) -> list[SegmentEmotion]:
        """Merge adjacent low-energy segments until count matches target."""
        if len(segments) <= target_count:
            return segments

        merged = self._do_merge(segments, audio_features, target_count)

        # Second pass: merge any remaining short segments (< MIN_CLIP_DURATION)
        merged = self._enforce_min_duration(merged, audio_features)

        return merged

    @staticmethod
    def _do_merge(
        segments: list[SegmentEmotion],
        audio_features: AudioFeatures,
        target_count: int,
    ) -> list[SegmentEmotion]:
        """Core merge loop: merge adjacent low-energy pairs until target count."""
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
                mood_description=a.mood_description,
            )
            merged[merge_idx] = merged_seg
            del merged[merge_idx + 1]

        return merged

    @staticmethod
    def _split_segments(
        segments: list[SegmentEmotion],
        target_count: int,
        min_clip: float = 1.5,
    ) -> list[SegmentEmotion]:
        """Split longest segments to create more slots for images.

        Repeatedly splits the longest segment in half until we reach
        target_count or no segment can be split without going below min_clip.
        """
        result = list(segments)
        while len(result) < target_count:
            # Find the longest segment that can be split
            best_idx = -1
            best_dur = 0.0
            for i, seg in enumerate(result):
                dur = seg.end - seg.start
                if dur > best_dur and dur / 2 >= min_clip:
                    best_dur = dur
                    best_idx = i
            if best_idx < 0:
                break  # No segment can be split further
            seg = result[best_idx]
            mid = (seg.start + seg.end) / 2
            first_half = SegmentEmotion(
                segment_index=seg.segment_index,
                start=seg.start,
                end=round(mid, 3),
                valence=seg.valence,
                arousal=seg.arousal,
                mood_description=seg.mood_description,
            )
            second_half = SegmentEmotion(
                segment_index=seg.segment_index + 1000,  # synthetic index
                start=round(mid, 3),
                end=seg.end,
                valence=seg.valence,
                arousal=seg.arousal,
                mood_description=seg.mood_description,
            )
            result[best_idx] = first_half
            result.insert(best_idx + 1, second_half)
        # Re-index so segment_index is sequential
        for i, seg in enumerate(result):
            result[i] = SegmentEmotion(
                segment_index=i,
                start=seg.start,
                end=seg.end,
                valence=seg.valence,
                arousal=seg.arousal,
                mood_description=seg.mood_description,
            )
        return result

    @staticmethod
    def _enforce_min_duration(
        segments: list[SegmentEmotion],
        audio_features: AudioFeatures,
        min_duration: float = 1.5,
    ) -> list[SegmentEmotion]:
        """Merge segments shorter than min_duration into their shortest neighbor."""
        merged = list(segments)
        changed = True
        while changed:
            changed = False
            for i in range(len(merged)):
                dur = merged[i].end - merged[i].start
                if dur >= min_duration or len(merged) <= 1:
                    continue
                # Pick neighbor with shortest duration to merge into
                if i == 0:
                    merge_with = 1
                elif i == len(merged) - 1:
                    merge_with = i - 1
                else:
                    dur_prev = merged[i - 1].end - merged[i - 1].start
                    dur_next = merged[i + 1].end - merged[i + 1].start
                    merge_with = i - 1 if dur_prev <= dur_next else i + 1
                # Merge: earlier segment absorbs later
                lo, hi = min(i, merge_with), max(i, merge_with)
                a, b = merged[lo], merged[hi]
                merged[lo] = SegmentEmotion(
                    segment_index=a.segment_index,
                    start=a.start,
                    end=b.end,
                    valence=round((a.valence + b.valence) / 2, 2),
                    arousal=round((a.arousal + b.arousal) / 2, 2),
                    mood_description=a.mood_description,
                )
                del merged[hi]
                changed = True
                break
        return merged
