"""Orchestration pipeline: coordinates the full video generation process."""

import logging
import shutil
from pathlib import Path
from typing import Callable

from app.config import settings
from app.models import (
    AudioFeatures,
    ProjectConfig,
    ProgressMessage,
    TaskStatus,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    project_id: str,
    task_id_arg: str,
    config: ProjectConfig,
    progress_callback: Callable[[ProgressMessage], None],
    cancel_check: Callable[[], bool],
):
    """Run the full video generation pipeline."""
    proj_dir = settings.data_dir / project_id
    images_dir = proj_dir / "images"
    image_paths = sorted(images_dir.glob("*"))
    music_files = list(proj_dir.glob("music.*"))
    audio_path = music_files[0]
    output_dir = proj_dir / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{task_id_arg}.mp4"

    def _progress(status: TaskStatus, progress: float, step: str, detail: str = ""):
        progress_callback(ProgressMessage(
            status=status, progress=progress,
            current_step=step, detail=detail,
        ))

    def _check_cancel():
        if cancel_check():
            _progress(TaskStatus.CANCELLED, 0, "Cancelled")
            raise RuntimeError("Task cancelled")

    # === Step 1: Analyze Audio ===
    _progress(TaskStatus.ANALYZING_AUDIO, 1, "Analyzing audio", "Extracting beats and features...")
    _check_cancel()

    from app.services.audio_analyzer import AudioAnalyzer
    analyzer = AudioAnalyzer()
    audio_features: AudioFeatures = analyzer.analyze(audio_path)

    _progress(TaskStatus.ANALYZING_AUDIO, 5, "Audio analyzed",
              f"Tempo: {audio_features.tempo:.0f} BPM, {len(audio_features.segments)} segments")

    # === Step 2: Classify Emotion ===
    _progress(TaskStatus.CLASSIFYING_EMOTION, 6, "Classifying music emotion", "Running Music2Emo model...")
    _check_cancel()

    from app.services.emotion_classifier import EmotionClassifier
    classifier = EmotionClassifier()
    try:
        segment_emotions = classifier.classify(audio_path, audio_features)
    except Exception as e:
        logger.warning(f"Music2Emo failed, using fallback: {e}")
        # Fallback: generate synthetic emotions from audio features
        segment_emotions = _fallback_emotions(audio_features)

    _progress(TaskStatus.CLASSIFYING_EMOTION, 10, "Emotion classified",
              f"Global mood: {segment_emotions[0].mood_description if segment_emotions else 'unknown'}")

    # === Step 2.5: Lyrics Analysis Pipeline ===
    if settings.lyrics_enabled:
        _check_cancel()
        lyric_emotions = _run_lyrics_pipeline(
            audio_path, audio_features, proj_dir, _progress, _check_cancel,
        )
        if lyric_emotions:
            from app.services.emotion_classifier import EmotionClassifier as EC
            segment_emotions = EC.enrich_with_lyrics(segment_emotions, lyric_emotions)
            _progress(TaskStatus.ANALYZING_LYRICS, 15, "Lyrics enrichment complete",
                      f"Enriched {len(lyric_emotions)} segments with lyric themes")

    # === Step 3: Caption Images ===
    _progress(TaskStatus.CAPTIONING_IMAGES, 16, "Captioning images", f"0/{len(image_paths)} images...")
    _check_cancel()

    from app.services.image_captioner import ImageCaptioner
    captioner = ImageCaptioner()

    def caption_progress(done: int, total: int):
        pct = 16 + (done / total) * 14  # 16% to 30%
        _progress(TaskStatus.CAPTIONING_IMAGES, pct, "Captioning images",
                  f"{done}/{total} images captioned")
        _check_cancel()

    image_captions = captioner.caption_images(image_paths, progress_callback=caption_progress)

    _progress(TaskStatus.CAPTIONING_IMAGES, 30, "Images captioned",
              f"{len(image_captions)} captions generated")

    # === Step 4: Semantic Matching ===
    _progress(TaskStatus.MATCHING, 31, "Matching images to music", "Computing semantic similarity...")
    _check_cancel()

    from app.services.matcher import SemanticMatcher
    matcher = SemanticMatcher()
    matches = matcher.match(segment_emotions, image_captions, audio_features)

    _progress(TaskStatus.MATCHING, 35, "Matching complete",
              f"{len(matches)} segment-image pairs")

    # === Step 5: Render Video ===
    _progress(TaskStatus.RENDERING, 36, "Rendering video", "Generating Ken Burns frames...")
    _check_cancel()

    from app.services.video_generator import VideoGenerator
    generator = VideoGenerator(
        aspect_ratio=config.aspect_ratio,
        quality=config.quality,
        fps=config.fps,
    )

    def render_progress(done: int, total: int):
        pct = 36 + (done / total) * 44  # 36% to 80%
        _progress(TaskStatus.RENDERING, pct, "Rendering video",
                  f"{done}/{total} segments rendered")
        _check_cancel()

    # === Step 5+6: Generate + Encode ===
    _progress(TaskStatus.ENCODING, 80, "Encoding video", "Assembling final MP4...")

    generator.generate(
        matches=matches,
        segment_emotions=segment_emotions,
        image_captions=image_captions,
        audio_features=audio_features,
        images_dir=images_dir,
        audio_path=audio_path,
        output_path=output_path,
        progress_callback=render_progress,
    )

    # === Cleanup: remove uploaded images, music, and captions ===
    _progress(TaskStatus.ENCODING, 98, "Cleaning up", "Removing temporary files...")
    _cleanup_project(proj_dir)

    _progress(TaskStatus.DONE, 100, "Done", str(output_path))
    return str(output_path)


def _run_lyrics_pipeline(
    audio_path: Path,
    audio_features,
    proj_dir: Path,
    _progress,
    _check_cancel,
) -> list:
    """Run vocal separation → transcription → lyric emotion analysis.

    Models load/unload sequentially to keep peak VRAM ≤ 2.5GB.
    Returns list of LyricEmotion or empty list if instrumental / failure.
    """
    from app.models import LyricEmotion

    _progress(TaskStatus.ANALYZING_LYRICS, 11, "Separating vocals", "Running Demucs...")

    # Step 1: Vocal separation
    from app.services.vocal_separator import VocalSeparator
    separator = VocalSeparator()
    try:
        vocals_dir = proj_dir / "vocals"
        vocals_path = separator.separate(audio_path, vocals_dir)
    except Exception as e:
        logger.warning(f"Vocal separation failed: {e}")
        return []
    finally:
        separator.unload()

    if vocals_path is None:
        _progress(TaskStatus.ANALYZING_LYRICS, 15, "Instrumental track",
                  "No vocals detected, skipping lyrics analysis")
        return []

    _check_cancel()
    _progress(TaskStatus.ANALYZING_LYRICS, 12, "Transcribing lyrics", "Running Whisper...")

    # Step 2: Transcription
    from app.services.lyrics_transcriber import LyricsTranscriber
    transcriber = LyricsTranscriber()
    try:
        lyrics_result = transcriber.transcribe(vocals_path)
    except Exception as e:
        logger.warning(f"Lyrics transcription failed: {e}")
        return []
    finally:
        transcriber.unload()

    if not lyrics_result.has_vocals:
        _progress(TaskStatus.ANALYZING_LYRICS, 15, "Low-quality transcription",
                  "Lyrics too noisy, skipping analysis")
        return []

    _check_cancel()
    _progress(TaskStatus.ANALYZING_LYRICS, 13, "Analyzing lyric emotions",
              f"Processing {len(lyrics_result.words)} words...")

    # Step 3: Per-segment emotion analysis via Ollama
    from app.services.lyric_emotion_analyzer import LyricEmotionAnalyzer
    analyzer = LyricEmotionAnalyzer()
    try:
        lyric_emotions = analyzer.analyze(lyrics_result, audio_features)
    except Exception as e:
        logger.warning(f"Lyric emotion analysis failed: {e}")
        return []

    return lyric_emotions


def _cleanup_project(proj_dir: Path):
    """Remove uploaded images, music, cached captions, and vocals after video is generated."""
    try:
        # Remove images directory
        images_dir = proj_dir / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)

        # Remove music files
        for f in proj_dir.glob("music.*"):
            f.unlink()

        # Remove caption cache
        captions_dir = proj_dir / "captions"
        if captions_dir.exists():
            shutil.rmtree(captions_dir)

        # Remove separated vocals
        vocals_dir = proj_dir / "vocals"
        if vocals_dir.exists():
            shutil.rmtree(vocals_dir)

        logger.info(f"Cleaned up project files in {proj_dir}")
    except Exception as e:
        logger.warning(f"Cleanup failed for {proj_dir}: {e}")


def _fallback_emotions(audio_features: AudioFeatures):
    """Generate synthetic emotions when Music2Emo is unavailable."""
    from app.models import SegmentEmotion

    results = []
    max_energy = max((s.rms_energy for s in audio_features.segments), default=1e-6)
    max_centroid = max((s.spectral_centroid for s in audio_features.segments), default=1e-6)

    for i, seg in enumerate(audio_features.segments):
        energy_ratio = seg.rms_energy / max(max_energy, 1e-6)
        brightness = seg.spectral_centroid / max(max_centroid, 1e-6)

        arousal = 3.0 + energy_ratio * 4.0  # 3..7
        valence = 3.0 + brightness * 4.0    # 3..7

        # Determine mood
        if valence >= 5.0 and arousal >= 5.0:
            mood = "Energetic, joyful, and uplifting"
        elif valence >= 5.0:
            mood = "Peaceful, warm, and serene"
        elif arousal >= 5.0:
            mood = "Intense, dramatic, and dark"
        else:
            mood = "Melancholic, introspective, and somber"

        tempo = audio_features.tempo
        if tempo > 140:
            mood += ". Fast-paced with steady energy."
        elif tempo > 100:
            mood += ". Moderate tempo with calm flow."
        else:
            mood += ". Gentle with quiet stillness."

        results.append(SegmentEmotion(
            segment_index=i,
            start=seg.start,
            end=seg.end,
            valence=round(valence, 2),
            arousal=round(arousal, 2),
            mood_description=mood,
        ))

    return results
