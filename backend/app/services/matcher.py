"""Semantic matching: music mood ↔ images via CLIP multimodal embeddings."""

import logging
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.config import settings
from app.models import AudioFeatures, ImageCaption, MatchResult, SegmentEmotion

logger = logging.getLogger(__name__)


class SemanticMatcher:
    def __init__(self):
        self._clip = None

    def _load_clip(self):
        if self._clip is not None:
            return
        from app.services.clip_encoder import CLIPEncoder
        self._clip = CLIPEncoder()

    def match(
        self,
        segment_emotions: list[SegmentEmotion],
        image_captions: list[ImageCaption],
        audio_features: AudioFeatures,
        images_dir: Path | None = None,
    ) -> tuple[list[MatchResult], list[SegmentEmotion]]:
        """Match music segments to images using CLIP multimodal similarity.

        Uses a hybrid score:
        - clip_image_weight: mood text ↔ image pixels (CLIP cross-modal)
        - clip_text_weight: mood text ↔ caption text (CLIP text-text)

        Returns (matches, merged_segment_emotions) — the caller must use
        the returned emotions for video timing, as segments may have been
        merged to fit the available image count.
        """
        self._load_clip()

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
            segment_emotions = self._split_segments(
                segment_emotions, target_count=n_images
            )
            n_segments = len(segment_emotions)
        else:
            n_segments = len(segment_emotions)

        # Prefer visual_mood_description for CLIP (falls back to mood_description)
        mood_texts = [
            se.visual_mood_description or se.mood_description
            for se in segment_emotions
        ]

        # Encode mood texts with CLIP
        mood_embeddings = self._clip.encode_texts(mood_texts)

        # Encode images with CLIP (use cached embeddings if available)
        w_img = settings.clip_image_weight
        w_txt = settings.clip_text_weight

        image_embeddings = self._encode_or_load_images(
            image_captions, images_dir
        )

        if image_embeddings is not None and w_img > 0:
            # Cross-modal: mood text ↔ image pixels
            sim_image = mood_embeddings @ image_embeddings.T
        else:
            sim_image = None
            w_img = 0.0
            w_txt = 1.0

        # Text-text: mood text ↔ caption text
        caption_texts = [ic.caption for ic in image_captions]
        caption_embeddings = self._clip.encode_texts(caption_texts)
        sim_text = mood_embeddings @ caption_embeddings.T

        # Hybrid similarity
        if sim_image is not None:
            similarity = w_img * sim_image + w_txt * sim_text
        else:
            similarity = sim_text

        logger.info(
            f"CLIP similarity stats — "
            f"image signal: mean={sim_image.mean():.3f} std={sim_image.std():.3f}, "
            f"text signal: mean={sim_text.mean():.3f} std={sim_text.std():.3f}, "
            f"hybrid: mean={similarity.mean():.3f} std={similarity.std():.3f}"
            if sim_image is not None else
            f"CLIP similarity stats (text-only) — "
            f"mean={similarity.mean():.3f} std={similarity.std():.3f}"
        )

        # Hungarian assignment
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

        # Unload CLIP to free VRAM before rendering
        self._clip.unload()
        self._clip = None

        return results, segment_emotions

    def _encode_or_load_images(
        self,
        image_captions: list[ImageCaption],
        images_dir: Path | None,
    ) -> np.ndarray | None:
        """Encode images via CLIP, using cached embeddings where available."""
        if images_dir is None:
            logger.warning("No images_dir provided, skipping CLIP image encoding")
            return None

        # Check which images have cached embeddings
        cached = []
        uncached_indices = []
        for i, ic in enumerate(image_captions):
            if ic.clip_embedding is not None:
                cached.append((i, np.array(ic.clip_embedding, dtype=np.float32)))
            else:
                uncached_indices.append(i)

        if cached:
            logger.info(
                f"CLIP image embeddings: {len(cached)} from cache, "
                f"{len(uncached_indices)} to encode"
            )

        # Encode uncached images
        if uncached_indices:
            paths = [images_dir / image_captions[i].filename for i in uncached_indices]
            new_embeddings = self._clip.encode_images_batch(paths)

            # Store back into ImageCaption objects for cache persistence
            for j, idx in enumerate(uncached_indices):
                image_captions[idx].clip_embedding = new_embeddings[j].tolist()

        # Assemble full embedding matrix in order
        n = len(image_captions)
        dim = 512
        result = np.zeros((n, dim), dtype=np.float32)
        for i, emb in cached:
            result[i] = emb
        if uncached_indices:
            for j, idx in enumerate(uncached_indices):
                result[idx] = new_embeddings[j]

        return result

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
                visual_mood_description=a.visual_mood_description,
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
                visual_mood_description=seg.visual_mood_description,
            )
            second_half = SegmentEmotion(
                segment_index=seg.segment_index + 1000,  # synthetic index
                start=round(mid, 3),
                end=seg.end,
                valence=seg.valence,
                arousal=seg.arousal,
                mood_description=seg.mood_description,
                visual_mood_description=seg.visual_mood_description,
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
                visual_mood_description=seg.visual_mood_description,
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
                    visual_mood_description=a.visual_mood_description,
                )
                del merged[hi]
                changed = True
                break
        return merged
