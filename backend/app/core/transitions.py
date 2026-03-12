"""Soft transitions: GPU-accelerated blur-dissolve at switch points only.

Guidelines:
- Pictures shown crisp and unmodified during their segment
- Subtle blur + soft fade only at transition points
- Smooth and natural, no abrupt scene changes
- GPU-accelerated blending and blur
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from moviepy import VideoClip

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_use_gpu = torch.cuda.is_available()


def compute_crossfade_duration(beat_times: list[float]) -> float:
    """Crossfade = 50% of median beat interval, clamped [0.3s, 1.2s]."""
    if len(beat_times) < 2:
        return 0.6
    intervals = [beat_times[i + 1] - beat_times[i] for i in range(len(beat_times) - 1)]
    return max(0.3, min(1.2, float(np.median(intervals)) * 0.5))


def snap_to_beat(time: float, beat_times: list[float]) -> float:
    if not beat_times:
        return time
    idx = int(np.argmin([abs(b - time) for b in beat_times]))
    return beat_times[idx]


def _smooth_alpha(t: float) -> float:
    """Smoothstep ease-in-out."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _gpu_blur_blend(
    frame1: np.ndarray, frame2: np.ndarray, alpha: float, blur_k: int
) -> np.ndarray:
    """GPU-accelerated blur + blend of two frames."""
    t1 = torch.from_numpy(frame1).float().permute(2, 0, 1).unsqueeze(0).to(_device)
    t2 = torch.from_numpy(frame2).float().permute(2, 0, 1).unsqueeze(0).to(_device)

    if blur_k >= 3:
        pad = blur_k // 2
        t1 = F.avg_pool2d(F.pad(t1, [pad] * 4, mode="reflect"), blur_k, stride=1)
        t2 = F.avg_pool2d(F.pad(t2, [pad] * 4, mode="reflect"), blur_k, stride=1)

    blended = ((1.0 - alpha) * t1 + alpha * t2).clamp(0, 255)
    return blended[0].permute(1, 2, 0).byte().cpu().numpy()


def _cpu_blur_blend(
    frame1: np.ndarray, frame2: np.ndarray, alpha: float, blur_k: int
) -> np.ndarray:
    """CPU fallback blur + blend."""
    if blur_k >= 3:
        frame1 = cv2.GaussianBlur(frame1, (blur_k, blur_k), 0)
        frame2 = cv2.GaussianBlur(frame2, (blur_k, blur_k), 0)
    return ((1.0 - alpha) * frame1 + alpha * frame2).astype(np.uint8)


def crossfade_clips(clips: list[VideoClip], xfade_duration: float) -> VideoClip:
    """Soft transition between clips.

    - Single clip zone: original frame, zero processing
    - Crossfade zone: smoothstep alpha + light blur at midpoint only
    """
    if len(clips) <= 1:
        return clips[0] if clips else VideoClip(
            lambda t: np.zeros((1, 1, 3), dtype=np.uint8), duration=0
        )

    xfade = min(xfade_duration, min(c.duration for c in clips) * 0.4)

    starts = [0.0]
    for i in range(len(clips) - 1):
        starts.append(starts[-1] + clips[i].duration - xfade)
    total_duration = starts[-1] + clips[-1].duration

    def make_frame(t):
        active = []
        for i, clip in enumerate(clips):
            s = starts[i]
            if s <= t < s + clip.duration:
                active.append((i, t - s))

        if not active:
            return clips[-1].get_frame(clips[-1].duration - 0.001)

        if len(active) == 1:
            idx, lt = active[0]
            return clips[idx].get_frame(min(lt, clips[idx].duration - 0.001))

        # Two clips overlapping — crossfade
        idx1, t1 = active[0]
        idx2, t2 = active[1]

        raw_alpha = (t - starts[idx2]) / xfade if xfade > 0 else 1.0
        alpha = _smooth_alpha(raw_alpha)

        f1 = clips[idx1].get_frame(min(t1, clips[idx1].duration - 0.001))
        f2 = clips[idx2].get_frame(min(t2, clips[idx2].duration - 0.001))

        if f1.shape != f2.shape:
            return f2

        # Blur only at midpoint of transition (center 40%)
        center_dist = abs(raw_alpha - 0.5) * 2.0
        blur_intensity = max(0.0, 1.0 - center_dist * 2.5)

        blur_k = 0
        if blur_intensity > 0.01:
            blur_k = int(blur_intensity * 15)
            blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1
            if blur_k < 3:
                blur_k = 0

        if _use_gpu:
            return _gpu_blur_blend(f1, f2, alpha, blur_k)
        return _cpu_blur_blend(f1, f2, alpha, blur_k)

    return VideoClip(make_frame, duration=total_duration).with_fps(clips[0].fps)
