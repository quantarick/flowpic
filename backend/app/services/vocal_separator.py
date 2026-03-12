"""Vocal separation from music using Demucs (htdemucs model)."""

import gc
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VocalSeparator:
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from demucs.pretrained import get_model

        self._model = get_model("htdemucs")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._model.eval()
        logger.info(f"Demucs loaded on {device}")

    def separate(self, audio_path: Path, output_dir: Path) -> Path | None:
        """Separate vocals from audio. Returns vocals path or None if instrumental."""
        self._load_model()

        import torchaudio
        from demucs.apply import apply_model

        wav, sr = torchaudio.load(str(audio_path))
        # Demucs expects a specific sample rate
        if sr != self._model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, self._model.samplerate)
            sr = self._model.samplerate

        # Ensure stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        device = next(self._model.parameters()).device
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        # apply_model expects (batch, channels, samples)
        sources = apply_model(self._model, wav[None].to(device), progress=False)[0]
        sources = sources * ref.std() + ref.mean()

        # htdemucs sources order: drums, bass, other, vocals
        vocals = sources[-1].cpu()

        if not self._has_meaningful_vocals(vocals, sr):
            logger.info("No meaningful vocals detected — instrumental track")
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        vocals_path = output_dir / "vocals.wav"
        torchaudio.save(str(vocals_path), vocals, sr)
        logger.info(f"Vocals saved to {vocals_path}")
        return vocals_path

    def _has_meaningful_vocals(self, vocals: torch.Tensor, sr: int) -> bool:
        """Check if separated vocals have meaningful content via RMS + active frame ratio."""
        audio = vocals.mean(0).numpy()  # mono
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.01:
            return False

        # Check active frame ratio (frames above noise floor)
        frame_length = int(0.025 * sr)  # 25ms frames
        hop = frame_length // 2
        n_frames = max(1, (len(audio) - frame_length) // hop)
        active = 0
        for i in range(n_frames):
            start = i * hop
            frame = audio[start:start + frame_length]
            frame_rms = np.sqrt(np.mean(frame ** 2))
            if frame_rms > 0.02:
                active += 1

        active_ratio = active / n_frames
        logger.info(f"Vocal check: RMS={rms:.4f}, active_ratio={active_ratio:.2f}")
        return active_ratio > 0.1

    def unload(self):
        """Free GPU memory before loading next model."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Demucs model unloaded")
