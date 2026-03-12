"""Audio analysis using librosa: beats, onsets, segments, spectral features."""

from pathlib import Path

import librosa
import numpy as np

from app.models import AudioFeatures, AudioSegment


class AudioAnalyzer:
    def __init__(self, sr: int = 22050):
        self.sr = sr

    def analyze(self, audio_path: Path) -> AudioFeatures:
        y, sr = librosa.load(str(audio_path), sr=self.sr)
        duration = len(y) / sr

        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        tempo_val = float(np.atleast_1d(tempo)[0])

        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

        # Build segments at 4-beat boundaries, min 2s / max 6s
        segments = self._build_segments(y, sr, beat_times, duration)

        return AudioFeatures(
            beat_times=beat_times,
            onset_times=onset_times,
            tempo=tempo_val,
            duration=duration,
            segments=segments,
        )

    def _build_segments(
        self,
        y: np.ndarray,
        sr: int,
        beat_times: list[float],
        duration: float,
    ) -> list[AudioSegment]:
        """Create segments at 4-beat boundaries with min 2s / max 6s constraint."""
        if len(beat_times) < 2:
            # Fallback: single segment
            return [self._compute_segment_features(y, sr, 0.0, duration, len(beat_times))]

        # Build 4-beat boundary times
        boundaries = [0.0]
        for i in range(0, len(beat_times), 4):
            t = beat_times[i]
            if t > boundaries[-1]:
                boundaries.append(t)
        if boundaries[-1] < duration:
            boundaries.append(duration)

        # Merge short segments, split long ones
        segments: list[AudioSegment] = []
        i = 0
        while i < len(boundaries) - 1:
            start = boundaries[i]
            end = boundaries[i + 1]
            seg_dur = end - start

            # Merge with next if too short
            if seg_dur < 2.0 and i + 2 < len(boundaries):
                end = boundaries[i + 2]
                i += 2
            elif seg_dur > 6.0:
                # Split in half
                mid = start + seg_dur / 2
                beat_count_1 = sum(1 for b in beat_times if start <= b < mid)
                beat_count_2 = sum(1 for b in beat_times if mid <= b < end)
                segments.append(self._compute_segment_features(y, sr, start, mid, beat_count_1))
                segments.append(self._compute_segment_features(y, sr, mid, end, beat_count_2))
                i += 1
                continue
            else:
                i += 1

            beat_count = sum(1 for b in beat_times if start <= b < end)
            segments.append(self._compute_segment_features(y, sr, start, end, beat_count))

        return segments

    def _compute_segment_features(
        self,
        y: np.ndarray,
        sr: int,
        start: float,
        end: float,
        beat_count: int,
    ) -> AudioSegment:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_y = y[start_sample:end_sample]

        if len(segment_y) == 0:
            return AudioSegment(
                start=start, end=end, beat_count=beat_count,
                rms_energy=0.0, spectral_centroid=0.0, chroma=[0.0] * 12,
            )

        rms = float(np.sqrt(np.mean(segment_y ** 2)))

        spec_cent = librosa.feature.spectral_centroid(y=segment_y, sr=sr)
        centroid = float(np.mean(spec_cent))

        chroma = librosa.feature.chroma_stft(y=segment_y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1).tolist()

        return AudioSegment(
            start=start,
            end=end,
            beat_count=beat_count,
            rms_energy=rms,
            spectral_centroid=centroid,
            chroma=chroma_mean,
        )
