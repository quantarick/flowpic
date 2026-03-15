"""Orchestration pipeline: coordinates the full video generation process."""

import json
import logging
import shutil
from pathlib import Path
from typing import Callable

from app.config import settings
from app.models import (
    AudioFeatures,
    ImageCaption,
    LocationGroup,
    MatchResult,
    ProjectConfig,
    ProgressMessage,
    SegmentEmotion,
    TaskStatus,
)

logger = logging.getLogger(__name__)


def _cache_path(proj_dir: Path, key: str) -> Path:
    """Return path to a pipeline stage cache file."""
    cache_dir = proj_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{key}.json"


def _save_cache(proj_dir: Path, key: str, data) -> None:
    """Save a list of Pydantic models to cache."""
    path = _cache_path(proj_dir, key)
    path.write_text(
        json.dumps([m.model_dump() for m in data], ensure_ascii=False),
        encoding="utf-8",
    )


def _load_cache(proj_dir: Path, key: str, model_cls):
    """Load cached list of Pydantic models. Returns None if no cache."""
    path = _cache_path(proj_dir, key)
    if not path.exists():
        return None
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
        return [model_cls(**item) for item in items]
    except Exception as e:
        logger.warning(f"Cache load failed for {key}: {e}")
        return None


def _load_single_cache(proj_dir: Path, key: str, model_cls):
    """Load a single cached Pydantic model. Returns None if no cache."""
    path = _cache_path(proj_dir, key)
    if not path.exists():
        return None
    try:
        return model_cls(**json.loads(path.read_text(encoding="utf-8")))
    except Exception as e:
        logger.warning(f"Cache load failed for {key}: {e}")
        return None


def _save_single_cache(proj_dir: Path, key: str, data) -> None:
    """Save a single Pydantic model to cache."""
    path = _cache_path(proj_dir, key)
    path.write_text(data.model_dump_json(), encoding="utf-8")


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

    _last_db_status = [None]  # mutable container for closure

    def _progress(status: TaskStatus, progress: float, step: str, detail: str = ""):
        if status != _last_db_status[0]:
            from app.services.task_db import update_task_status
            update_task_status(task_id_arg, status)
            _last_db_status[0] = status
        progress_callback(ProgressMessage(
            status=status, progress=progress,
            current_step=step, detail=detail,
        ))

    def _check_cancel():
        if cancel_check():
            _progress(TaskStatus.CANCELLED, 0, "Cancelled")
            raise RuntimeError("Task cancelled")

    # === Step 1: Analyze Audio ===
    audio_features = _load_single_cache(proj_dir, "audio_features", AudioFeatures)
    if audio_features:
        logger.info("Loaded audio features from cache")
        _progress(TaskStatus.ANALYZING_AUDIO, 5, "Audio analyzed (cached)",
                  f"Tempo: {audio_features.tempo:.0f} BPM, {len(audio_features.segments)} segments")
    else:
        _progress(TaskStatus.ANALYZING_AUDIO, 1, "Analyzing audio", "Extracting beats and features...")
        _check_cancel()

        from app.services.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        audio_features = analyzer.analyze(audio_path)
        _save_single_cache(proj_dir, "audio_features", audio_features)

        _progress(TaskStatus.ANALYZING_AUDIO, 5, "Audio analyzed",
                  f"Tempo: {audio_features.tempo:.0f} BPM, {len(audio_features.segments)} segments")

    # === Step 2: Classify Emotion ===
    segment_emotions = _load_cache(proj_dir, "segment_emotions", SegmentEmotion)
    if segment_emotions:
        logger.info("Loaded segment emotions from cache")
        _progress(TaskStatus.CLASSIFYING_EMOTION, 15, "Emotion classified (cached)",
                  f"Global mood: {segment_emotions[0].mood_description if segment_emotions else 'unknown'}")
    else:
        _progress(TaskStatus.CLASSIFYING_EMOTION, 6, "Classifying music emotion", "Running Music2Emo model...")
        _check_cancel()

        from app.services.emotion_classifier import EmotionClassifier
        classifier = EmotionClassifier()
        try:
            segment_emotions = classifier.classify(audio_path, audio_features)
        except Exception as e:
            logger.warning(f"Music2Emo failed, using fallback: {e}")
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

        _save_cache(proj_dir, "segment_emotions", segment_emotions)

    # === Step 3: Caption Images ===
    _progress(TaskStatus.CAPTIONING_IMAGES, 16, "Captioning images", f"0/{len(image_paths)} images...")
    _check_cancel()

    from app.services.image_captioner import ImageCaptioner
    captioner = ImageCaptioner(model=config.vision_model)

    def caption_progress(done: int, total: int):
        pct = 16 + (done / total) * 14  # 16% to 30%
        _progress(TaskStatus.CAPTIONING_IMAGES, pct, "Captioning images",
                  f"{done}/{total} images captioned")
        _check_cancel()

    image_captions = captioner.caption_images(image_paths, progress_callback=caption_progress)

    _progress(TaskStatus.CAPTIONING_IMAGES, 30, "Images captioned",
              f"{len(image_captions)} captions generated")

    # === Step 3.5: Deduplicate similar images ===
    image_captions = _deduplicate_images(image_captions, images_dir)
    _progress(TaskStatus.CAPTIONING_IMAGES, 31, "Deduplicated",
              f"{len(image_captions)} unique images after dedup")

    # === Step 3.6: Drop close-up portraits that can't be properly cropped ===
    from app.services.smart_crop import check_face_fits, get_output_resolution
    out_w, out_h = get_output_resolution(config.aspect_ratio, config.quality)
    before_closeup = len(image_captions)
    image_captions = [
        ic for ic in image_captions
        if not ic.face_regions or not ic.img_width
        or check_face_fits(ic.img_width, ic.img_height, ic.face_regions, out_w, out_h)
    ]
    dropped = before_closeup - len(image_captions)
    if dropped:
        logger.info(f"Dropped {dropped} close-up images that can't be properly cropped")
        _progress(TaskStatus.CAPTIONING_IMAGES, 31, "Close-up filter",
                  f"Dropped {dropped} uncropable close-ups, {len(image_captions)} remaining")

    # === Step 4: Semantic Matching ===
    matches = _load_cache(proj_dir, "matches", MatchResult)
    location_groups = _load_cache(proj_dir, "location_groups", LocationGroup)
    merged_emotions = _load_cache(proj_dir, "merged_emotions", SegmentEmotion)
    if matches and location_groups is not None and merged_emotions:
        segment_emotions = merged_emotions
        logger.info("Loaded matches and location groups from cache")
        _progress(TaskStatus.MATCHING, 35, "Matching complete (cached)",
                  f"{len(matches)} segment-image pairs")
    else:
        _progress(TaskStatus.MATCHING, 31, "Matching images to music", "Computing semantic similarity...")
        _check_cancel()

        from app.services.matcher import SemanticMatcher
        matcher = SemanticMatcher()
        matches, segment_emotions = matcher.match(segment_emotions, image_captions, audio_features)

        _progress(TaskStatus.MATCHING, 35, "Matching complete",
                  f"{len(matches)} segment-image pairs")

        # === Step 4.5: Cluster by location + Compute Location Groups ===
        caption_map_debug = {ic.filename: ic for ic in image_captions}
        for m in matches:
            cap = caption_map_debug.get(m.image_filename)
            pn = cap.place_name if cap else None
            logger.info(f"  Pre-cluster match seg={m.segment_index} img={m.image_filename} place={pn}")
        matches = _cluster_matches_by_location(matches, image_captions)
        for m in matches:
            cap = caption_map_debug.get(m.image_filename)
            pn = cap.place_name if cap else None
            logger.info(f"  Post-cluster match seg={m.segment_index} img={m.image_filename} place={pn}")
        location_groups = _compute_location_groups(matches, image_captions)
        if location_groups:
            logger.info(f"Location groups: {len(location_groups)} distinct locations")

        _save_cache(proj_dir, "matches", matches)
        _save_cache(proj_dir, "location_groups", location_groups)
        _save_cache(proj_dir, "merged_emotions", segment_emotions)

    # === Step 4.75: Review person crops ===
    _progress(TaskStatus.REVIEWING_CROPS, 36, "Reviewing crops", "Checking person visibility...")
    _check_cancel()

    from app.services.crop_reviewer import CropReviewer
    reviewer = CropReviewer(model=config.vision_model)
    person_count = sum(1 for ic in image_captions if ic.has_person)
    if person_count > 0:
        def review_progress(done, total):
            pct = 36 + (done / total) * 4  # 36% to 40%
            _progress(TaskStatus.REVIEWING_CROPS, pct, "Reviewing crops",
                      f"{done}/{total} person images reviewed")
            _check_cancel()

        image_captions = reviewer.review_crops(
            matches, image_captions, images_dir,
            config.aspect_ratio, config.quality,
            progress_callback=review_progress,
        )

    # === Step 5: Render Video ===
    _progress(TaskStatus.RENDERING, 40, "Rendering video", "Generating Ken Burns frames...")
    _check_cancel()

    from app.services.video_generator import VideoGenerator
    generator = VideoGenerator(
        aspect_ratio=config.aspect_ratio,
        quality=config.quality,
        fps=config.fps,
    )

    def render_progress(done: int, total: int):
        pct = 40 + (done / total) * 40  # 40% to 80%
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
        location_groups=location_groups,
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
    """No-op: keep all intermediate data for troubleshooting."""
    logger.info(f"Keeping all intermediate data in {proj_dir}")


def _deduplicate_images(
    image_captions: list[ImageCaption],
    images_dir: Path,
    hash_size: int = 16,
    threshold: int = 6,
) -> list[ImageCaption]:
    """Remove near-duplicate images, keeping the sharpest from each group.

    Uses difference hash (dHash) for fast perceptual similarity detection.
    Images with hamming distance <= threshold are considered duplicates.
    """
    import cv2

    def _dhash(img_gray, size: int = 16) -> int:
        """Compute difference hash — a 256-bit perceptual fingerprint."""
        resized = cv2.resize(img_gray, (size + 1, size))
        diff = resized[:, 1:] > resized[:, :-1]
        h = 0
        for bit in diff.flatten():
            h = (h << 1) | int(bit)
        return h

    def _sharpness(img_gray) -> float:
        """Laplacian variance — higher = sharper."""
        return cv2.Laplacian(img_gray, cv2.CV_64F).var()

    def _hamming(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    # Compute hash + sharpness for each image
    hashes: list[tuple[int, int, float]] = []  # (index, hash, sharpness)
    for i, cap in enumerate(image_captions):
        img_path = images_dir / cap.filename
        img = cv2.imread(str(img_path))
        if img is None:
            hashes.append((i, 0, 0.0))
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = _dhash(gray, hash_size)
        s = _sharpness(gray)
        hashes.append((i, h, s))

    # Group duplicates using union-find
    n = len(hashes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if _hamming(hashes[i][1], hashes[j][1]) <= threshold:
                union(i, j)

    # Pick the sharpest image from each group
    groups: dict[int, list[tuple[int, float]]] = {}
    for i, (idx, h, s) in enumerate(hashes):
        root = find(i)
        groups.setdefault(root, []).append((idx, s))

    keep_indices: set[int] = set()
    for root, members in groups.items():
        best_idx = max(members, key=lambda x: x[1])[0]
        keep_indices.add(best_idx)
        if len(members) > 1:
            dropped = [image_captions[idx].filename for idx, _ in members if idx != best_idx]
            logger.info(
                f"Dedup: keeping '{image_captions[best_idx].filename}' "
                f"(sharpness={max(members, key=lambda x: x[1])[1]:.0f}), "
                f"dropped {len(dropped)} duplicates: {dropped}"
            )

    result = [cap for i, cap in enumerate(image_captions) if i in keep_indices]
    if len(result) < len(image_captions):
        logger.info(f"Deduplication: {len(image_captions)} → {len(result)} images")
    else:
        logger.info("Deduplication: no duplicates found")
    return result


def _cluster_matches_by_location(
    matches: list[MatchResult],
    image_captions: list[ImageCaption],
) -> list[MatchResult]:
    """Reorder matches so images from the same GPS location are consecutive.

    Keeps the segment timeline (segment_index order) intact but reassigns
    which image plays at each position so same-location images are grouped.
    """
    caption_map = {ic.filename: ic for ic in image_captions}

    # Build place_name for each match
    def get_place(m: MatchResult) -> str | None:
        cap = caption_map.get(m.image_filename)
        if cap and cap.place_name and cap.latitude is not None:
            return cap.place_name
        return None

    # Collect image filenames sorted by place_name (stable sort keeps
    # original relative order within same place)
    indexed = [(i, m, get_place(m)) for i, m in enumerate(matches)]
    gps = [(i, m, p) for i, m, p in indexed if p is not None]
    no_gps = [(i, m) for i, m, p in indexed if p is None]

    if not gps:
        return matches

    # Sort GPS images by place_name
    gps.sort(key=lambda x: x[2])
    clustered_images = [m.image_filename for _, m, _ in gps]
    clustered_scores = [m.similarity_score for _, m, _ in gps]

    # Original segment indices in timeline order (sorted by position)
    gps_positions = sorted([i for i, _, _ in gps])

    # Original segment_index values at those positions
    segment_indices = [matches[pos].segment_index for pos in gps_positions]

    # Rebuild matches: assign clustered images to timeline positions
    result = list(matches)
    for slot, pos in enumerate(gps_positions):
        result[pos] = MatchResult(
            segment_index=segment_indices[slot],
            image_filename=clustered_images[slot],
            similarity_score=clustered_scores[slot],
        )

    places = [get_place(result[pos]) for pos in gps_positions]
    logger.info(f"Clustered {len(gps)} GPS matches by location: {places}")
    return result


def _compute_location_groups(
    matches: list[MatchResult],
    image_captions: list[ImageCaption],
    threshold_meters: float = 500.0,
) -> list[LocationGroup]:
    """Group consecutive clips by GPS location.

    Returns a list of LocationGroup, one for each distinct location change.
    Only images with GPS data and a resolved place_name are considered.
    """
    from app.services.gps_extractor import haversine_distance

    caption_map = {ic.filename: ic for ic in image_captions}
    groups: list[LocationGroup] = []
    current_place: str | None = None
    current_lat: float | None = None
    current_lon: float | None = None
    group_start: int = 0

    for clip_idx, match in enumerate(matches):
        cap = caption_map.get(match.image_filename)
        if not cap or not cap.place_name or cap.latitude is None or cap.longitude is None:
            continue

        if current_place is None:
            # First GPS-tagged clip
            current_place = cap.place_name
            current_lat = cap.latitude
            current_lon = cap.longitude
            group_start = clip_idx
        else:
            dist = haversine_distance(current_lat, current_lon, cap.latitude, cap.longitude)
            if dist > threshold_meters or cap.place_name != current_place:
                # Location changed — finalize previous group
                groups.append(LocationGroup(
                    place_name=current_place,
                    start_clip_index=group_start,
                    end_clip_index=clip_idx - 1,
                ))
                current_place = cap.place_name
                current_lat = cap.latitude
                current_lon = cap.longitude
                group_start = clip_idx

    # Finalize last group
    if current_place is not None:
        groups.append(LocationGroup(
            place_name=current_place,
            start_clip_index=group_start,
            end_clip_index=len(matches) - 1,
        ))

    return groups


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
