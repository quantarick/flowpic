"""Video generation: GPU-accelerated Ken Burns + crossfade + moviepy assembly."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from moviepy import AudioFileClip, VideoClip, concatenate_videoclips

from app.core.ken_burns import KenBurnsEngine
from app.core.transitions import (
    clip_durations,
    compute_crossfade_duration,
    crossfade_clips,
    fit_crossfade,
)
from app.models import (
    AspectRatio,
    AudioFeatures,
    FitStrategy,
    FrameStyle,
    ImageCaption,
    LocationGroup,
    MatchResult,
    Quality,
    SegmentEmotion,
)
from app.services.border_trim import remove_frame
from app.services.smart_crop import get_output_resolution, smart_fit, pan_fit, frame_fit, remap_face_regions
from app.services.subtitle_renderer import (
    compute_title_font_size,
    generate_title_card,
)
from app.styles import DEFAULT_STYLE, get_style

logger = logging.getLogger(__name__)
_use_gpu = torch.cuda.is_available()

# Put the MP4 index at the front of the file, so a phone or browser can start
# playing while the rest is still downloading.
_FASTSTART = ["-movflags", "+faststart"]


def _pick_encoder(width: int = 0, height: int = 0) -> tuple[str, list[str]]:
    """Select the best available H.264 encoder: NVENC GPU → CPU fallback.

    NVENC on consumer GPUs has a max encode width of 4096px.
    If the output exceeds this, fall back to libx264.
    """
    import subprocess
    from moviepy.config import FFMPEG_BINARY

    # NVENC max dimension on most consumer NVIDIA GPUs
    NVENC_MAX_DIM = 4096

    try:
        result = subprocess.run(
            [FFMPEG_BINARY, "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        encoders = result.stdout
    except Exception:
        encoders = ""

    if "h264_nvenc" in encoders and width <= NVENC_MAX_DIM and height <= NVENC_MAX_DIM:
        logger.info("Using NVENC GPU encoder (h264_nvenc) for %dx%d", width, height)
        return "h264_nvenc", [
            "-pix_fmt", "yuv420p",
            "-preset", "p4", "-rc", "vbr", "-cq", "23",
            *_FASTSTART,
        ]
    else:
        if width > NVENC_MAX_DIM or height > NVENC_MAX_DIM:
            logger.info("Resolution %dx%d exceeds NVENC limit (%d), using CPU encoder",
                        width, height, NVENC_MAX_DIM)
        else:
            logger.info("Using CPU encoder (libx264)")
        return "libx264", ["-pix_fmt", "yuv420p", *_FASTSTART]


class VideoGenerator:
    def __init__(
        self,
        aspect_ratio: AspectRatio,
        quality: Quality,
        fps: int = 30,
        fit_strategy: FitStrategy = FitStrategy.CROP,
        frame_style: FrameStyle = FrameStyle.FILM,
        video_style: str = DEFAULT_STYLE,
        remove_borders: bool = False,
    ):
        self.aspect_ratio = aspect_ratio
        self.quality = quality
        self.fps = fps
        self.fit_strategy = fit_strategy
        self.frame_style = frame_style
        # Cut off a frame that is printed on a photo before fitting it
        self.remove_borders = remove_borders
        self.style = get_style(video_style)
        self.out_w, self.out_h = get_output_resolution(aspect_ratio, quality)
        # Pictures are fitted and animated inside the picture area. It equals
        # the output frame unless the style adds letterbox bars.
        self.pic_w, self.pic_h = self.style.picture_size(self.out_w, self.out_h)
        self.ken_burns = KenBurnsEngine(self.pic_w, self.pic_h)

    def _grade_bgr(self, img: np.ndarray) -> np.ndarray:
        """Apply the style's grade to a BGR image."""
        graded = self.style.grade(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return cv2.cvtColor(graded, cv2.COLOR_RGB2BGR)

    def _title_background(self, canvas: np.ndarray) -> np.ndarray:
        """Centre-crop a fit canvas to the picture area, for use behind a title card.

        Fit canvases are larger than the picture area (headroom for Ken Burns
        motion, or a tall strip for panning). A title card is shown as a frame
        and blended with frames, so it has to be exactly the picture size.
        """
        h, w = canvas.shape[:2]
        target = self.pic_w / self.pic_h
        if w / h > target:
            crop_w = max(1, int(round(h * target)))
            x0 = (w - crop_w) // 2
            canvas = canvas[:, x0:x0 + crop_w]
        else:
            crop_h = max(1, int(round(w / target)))
            y0 = (h - crop_h) // 2
            canvas = canvas[y0:y0 + crop_h]
        return cv2.resize(
            np.ascontiguousarray(canvas), (self.pic_w, self.pic_h), interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _plan_timing(
        segments: list[SegmentEmotion], xfade_wanted: float,
    ) -> tuple[float, list[float]]:
        """The crossfade length and each clip's length for the segments being shown.

        A clip is on screen from the start of its own segment to the start of
        the next one shown, so a segment with no clip does not shift the rest:
        the clip before it stays up longer. The first clip starts with the music.
        """
        if not segments:
            return 0.0, []
        starts = [s.start for s in segments]
        in_order = all(b > a for a, b in zip(starts, starts[1:]))
        if in_order and segments[-1].end > starts[-1]:
            edges = [0.0, *starts[1:], segments[-1].end]
            spans = [b - a for a, b in zip(edges, edges[1:])]
        else:
            # Segments out of time order: fall back to each one's own length
            spans = [s.end - s.start for s in segments]
        xfade = fit_crossfade(spans, xfade_wanted)
        return xfade, clip_durations(spans, xfade)

    def generate(
        self,
        matches: list[MatchResult],
        segment_emotions: list[SegmentEmotion],
        image_captions: list[ImageCaption],
        audio_features: AudioFeatures,
        images_dir: Path,
        audio_path: Path,
        output_path: Path,
        progress_callback=None,
        location_groups: list[LocationGroup] | None = None,
    ):
        """Generate the final video."""
        caption_map = {ic.filename: ic for ic in image_captions}
        emotion_map = {se.segment_index: se for se in segment_emotions}

        # Build title card map: first clip of each location group (including first frame)
        # Only show when location changes (deduplicate consecutive same-name groups)
        title_card_map: dict[int, str] = {}  # clip_index → place_name
        prev_place: str | None = None
        if location_groups:
            logger.info(f"Location groups received: {len(location_groups)}")
            for lg in location_groups:
                logger.info(f"  Group '{lg.place_name}': clips {lg.start_clip_index}-{lg.end_clip_index}")
                if lg.place_name != prev_place:
                    title_card_map[lg.start_clip_index] = lg.place_name
                    prev_place = lg.place_name
        else:
            logger.info("No location groups provided")
        logger.info(f"Title card map: {title_card_map}")
        title_font_size = compute_title_font_size(self.out_h)

        # Compute global mood for font selection
        avg_valence = sum(se.valence for se in segment_emotions) / max(len(segment_emotions), 1)
        avg_arousal = sum(se.arousal for se in segment_emotions) / max(len(segment_emotions), 1)
        logger.info(f"Global mood: valence={avg_valence:.1f}, arousal={avg_arousal:.1f}")

        # Work out every clip's length before building any, so that each
        # picture change is centred on the beat where its segment starts.
        planned = [
            (i, emotion_map[m.segment_index])
            for i, m in enumerate(matches)
            if m.segment_index in emotion_map
            and cv2.haveImageReader(str(images_dir / m.image_filename))
        ]
        xfade, clip_length = self._plan_timing(
            [seg for _, seg in planned],
            compute_crossfade_duration(audio_features.beat_times),
        )
        clip_duration = {i: d for (i, _), d in zip(planned, clip_length)}
        logger.info(f"Timing: {len(planned)} clips, crossfade {xfade:.3f}s centred on segment boundaries")

        clips: list[VideoClip] = []
        total = len(matches)

        # Save all cropped images for troubleshooting
        crops_dir = images_dir.parent / "output" / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Rendering with GPU: {_use_gpu}")

        # Screen-fixed finishing (bars, vignette, grain) for every output frame.
        # None when the style has none, which costs nothing per frame.
        finish_frame = self.style.frame_finisher(self.out_w, self.out_h, self.fps)
        logger.info(
            f"Video style: {self.style.name} "
            f"(picture area {self.pic_w}x{self.pic_h} in {self.out_w}x{self.out_h})"
        )

        for i, match in enumerate(matches):
            if i not in clip_duration:
                continue
            seg_emo = emotion_map[match.segment_index]
            seg_duration = clip_duration[i]

            # Load image (keep BGR for smart_fit saliency calculations)
            img_path = images_dir / match.image_filename
            img = cv2.imread(str(img_path))
            if img is None:
                # The file has an image header but will not decode. Leaving the
                # clip out would move every later picture change off its beat.
                raise RuntimeError(f"Could not read image: {match.image_filename}")

            caption_info = caption_map.get(match.image_filename)
            if self.remove_borders:
                img, caption_info = remove_frame(img, caption_info)
            face_regions = caption_info.face_regions if caption_info else []
            focus_x = caption_info.focus_x if caption_info else 0.5
            focus_y = caption_info.focus_y if caption_info else 0.5
            fit_mode = caption_info.fit_mode if caption_info else "crop"

            subject_box = None
            if caption_info and caption_info.subject_x1 is not None:
                subject_box = (
                    caption_info.subject_x1, caption_info.subject_y1,
                    caption_info.subject_x2, caption_info.subject_y2,
                )

            stem = Path(match.image_filename).stem

            # Detect orientation mismatch: portrait image in landscape output
            img_h, img_w = img.shape[:2]
            img_aspect = img_w / img_h
            out_aspect = self.pic_w / self.pic_h
            use_pan = img_aspect / out_aspect < 0.65

            if self.fit_strategy == FitStrategy.FRAME:
                # Frame mode: all images get decorative border, no cropping.
                # The picture is graded before framing so the border keeps its own colour.
                fit_result = frame_fit(
                    self._grade_bgr(img), self.pic_w, self.pic_h,
                    frame_style=self.frame_style,
                    scale_factor=1.05,
                    metadata={"frame_number": i + 1},
                )

                cv2.imwrite(
                    str(crops_dir / f"{i:03d}_{stem}_frame.jpg"),
                    fit_result.canvas,
                )

                canvas = cv2.cvtColor(fit_result.canvas, cv2.COLOR_BGR2RGB)

                kb_params = self.ken_burns.generate_params(
                    segment_index=i,
                    arousal=seg_emo.arousal if seg_emo else 5.0,
                    face_regions=[],
                    source_w=canvas.shape[1],
                    source_h=canvas.shape[0],
                    content_center=(fit_result.content_center_x, fit_result.content_center_y),
                    max_zoom_pct=0.05,
                )

                logger.info(
                    f"Clip {i}: FRAME mode ({self.frame_style.value}) "
                    f"canvas={canvas.shape[1]}x{canvas.shape[0]} "
                    f"zoom={kb_params.zoom_start:.3f}→{kb_params.zoom_end:.3f}"
                )

            elif use_pan:
                # Portrait image → pan mode: scale to fill width, pan vertically
                pan_result = pan_fit(
                    img, self.pic_w, self.pic_h,
                    face_regions=face_regions,
                )

                # Save debug crop: center frame of the oversized canvas
                center_y = max(0, (pan_result.canvas_h - self.pic_h) // 2)
                debug_crop = pan_result.canvas[center_y:center_y + self.pic_h, 0:self.pic_w]
                cv2.imwrite(
                    str(crops_dir / f"{i:03d}_{stem}_pan.jpg"),
                    debug_crop,
                )

                # Convert to RGB for moviepy/Ken Burns, then apply the style's look
                canvas = self.style.grade(cv2.cvtColor(pan_result.canvas, cv2.COLOR_BGR2RGB))

                kb_params = self.ken_burns.generate_pan_params(
                    segment_index=i,
                    clip_duration=seg_duration,
                    canvas_h=pan_result.canvas_h,
                    canvas_w=pan_result.canvas_w,
                    face_regions=pan_result.face_regions_mapped,
                    pan_axis=pan_result.pan_axis,
                )

                logger.info(
                    f"Clip {i}: PAN mode ({pan_result.pan_axis}) "
                    f"canvas={pan_result.canvas_w}x{pan_result.canvas_h} "
                    f"pan_y={kb_params.pan_y_start:.3f}→{kb_params.pan_y_end:.3f}"
                )
            else:
                # Landscape/square image → crop mode with headroom for zoom
                fit_result = smart_fit(
                    img, self.pic_w, self.pic_h,
                    face_regions=face_regions,
                    focus_x=focus_x,
                    focus_y=focus_y,
                    scale_factor=1.08,
                    fit_mode=fit_mode,
                    subject_box=subject_box,
                    horizon_y=caption_info.horizon_y if caption_info else None,
                    horizon_valid=caption_info.horizon_valid if caption_info else False,
                    people_centers=caption_info.people_centers if caption_info else None,
                )

                # Save cropped image for debugging (already BGR)
                cv2.imwrite(
                    str(crops_dir / f"{i:03d}_{stem}_{fit_mode}.jpg"),
                    fit_result.canvas,
                )

                # Convert to RGB for moviepy/Ken Burns rendering, then apply the style's look
                canvas = self.style.grade(cv2.cvtColor(fit_result.canvas, cv2.COLOR_BGR2RGB))

                # Remap face regions to canvas coordinates
                canvas_faces = remap_face_regions(face_regions, fit_result)

                kb_params = self.ken_burns.generate_params(
                    segment_index=i,
                    arousal=seg_emo.arousal if seg_emo else 5.0,
                    face_regions=canvas_faces,
                    source_w=canvas.shape[1],
                    source_h=canvas.shape[0],
                    content_center=(fit_result.content_center_x, fit_result.content_center_y),
                )

                logger.info(
                    f"Clip {i}: ZOOM mode "
                    f"canvas={canvas.shape[1]}x{canvas.shape[0]} "
                    f"zoom={kb_params.zoom_start:.3f}→{kb_params.zoom_end:.3f}"
                )

            if _use_gpu:
                # Upload image to GPU once, render all frames via grid_sample
                source_gpu = self.ken_burns.upload_source(canvas)
                kb = self.ken_burns

                def make_frame(t, _gpu=source_gpu, _p=kb_params, _d=seg_duration, _kb=kb):
                    progress = max(0.0, min(1.0, t / max(_d, 0.001)))
                    return _kb.render_frame_gpu(_gpu, _p, progress)

            else:
                source_img = canvas.copy()
                kb = self.ken_burns

                def make_frame(t, _img=source_img, _p=kb_params, _d=seg_duration, _kb=kb):
                    progress = max(0.0, min(1.0, t / max(_d, 0.001)))
                    return _kb.render_frame(_img, _p, progress)

            # Wrap make_frame with title card + crossfade if this clip starts a large location group
            if i in title_card_map:
                place_name = title_card_map[i]
                logger.info(f"Generating title card for clip {i}: '{place_name}' (seg_duration={seg_duration:.2f}s)")
                title_img = generate_title_card(
                    self._title_background(canvas), place_name, title_font_size, avg_valence, avg_arousal,
                )
                title_duration = min(2.0, seg_duration * 0.4)
                fade_duration = 0.5
                _base_make_frame = make_frame

                def make_frame(
                    t,
                    _base=_base_make_frame,
                    _title=title_img,
                    _td=title_duration,
                    _fd=fade_duration,
                ):
                    if t < _td - _fd:
                        # Static title card
                        return _title
                    elif t < _td:
                        # Crossfade from title to normal frame
                        alpha = (t - (_td - _fd)) / _fd
                        normal = _base(t)
                        blended = (
                            _title.astype(np.float32) * (1.0 - alpha)
                            + normal.astype(np.float32) * alpha
                        )
                        return blended.astype(np.uint8)
                    else:
                        # Normal Ken Burns frame
                        return _base(t)

            # Finish every frame, title card included, with the style's bars, vignette and grain
            if finish_frame is not None:
                _unfinished = make_frame

                def make_frame(t, _base=_unfinished, _finish=finish_frame):
                    return _finish(_base(t), t)

            clip = VideoClip(make_frame, duration=seg_duration).with_fps(self.fps)
            clips.append(clip)

            if progress_callback:
                progress_callback(i + 1, total)

        if not clips:
            raise RuntimeError("No clips generated")

        # Apply crossfades between clips
        if len(clips) > 1 and xfade > 0:
            final_clip = crossfade_clips(clips, xfade)
        else:
            final_clip = concatenate_videoclips(clips)

        # Add audio — match durations
        audio_clip = AudioFileClip(str(audio_path))
        if abs(audio_clip.duration - final_clip.duration) > 0.5:
            target_dur = min(audio_clip.duration, final_clip.duration)
            audio_clip = audio_clip.subclipped(0, target_dur)
            final_clip = final_clip.subclipped(0, target_dur)
        else:
            audio_clip = audio_clip.subclipped(0, final_clip.duration)
        final_clip = final_clip.with_audio(audio_clip)

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        codec, ffmpeg_params = _pick_encoder(self.out_w, self.out_h)
        final_clip.write_videofile(
            str(output_path),
            fps=self.fps,
            codec=codec,
            audio_codec="aac",
            ffmpeg_params=ffmpeg_params,
            logger=None,
        )

        # Clean up
        final_clip.close()
        audio_clip.close()
        for c in clips:
            c.close()

        # Free GPU memory
        if _use_gpu:
            torch.cuda.empty_cache()
