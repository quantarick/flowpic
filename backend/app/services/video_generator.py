"""Video generation: GPU-accelerated Ken Burns + crossfade + moviepy assembly."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from moviepy import AudioFileClip, VideoClip, concatenate_videoclips

from app.core.ken_burns import KenBurnsEngine
from app.core.transitions import compute_crossfade_duration, crossfade_clips
from app.models import (
    AspectRatio,
    AudioFeatures,
    ImageCaption,
    MatchResult,
    Quality,
    SegmentEmotion,
)
from app.services.smart_crop import get_output_resolution, smart_fit, remap_face_regions

logger = logging.getLogger(__name__)
_use_gpu = torch.cuda.is_available()


class VideoGenerator:
    def __init__(
        self,
        aspect_ratio: AspectRatio,
        quality: Quality,
        fps: int = 30,
    ):
        self.aspect_ratio = aspect_ratio
        self.quality = quality
        self.fps = fps
        self.out_w, self.out_h = get_output_resolution(aspect_ratio, quality)
        self.ken_burns = KenBurnsEngine(self.out_w, self.out_h)

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
    ):
        """Generate the final video."""
        caption_map = {ic.filename: ic for ic in image_captions}
        emotion_map = {se.segment_index: se for se in segment_emotions}

        xfade_dur = compute_crossfade_duration(audio_features.beat_times)

        n_clips = sum(
            1 for m in matches if emotion_map.get(m.segment_index) is not None
        )
        total_xfade_loss = max(0, (n_clips - 1)) * xfade_dur
        per_clip_extra = total_xfade_loss / max(n_clips, 1)

        clips: list[VideoClip] = []
        total = len(matches)

        logger.info(f"Rendering with GPU: {_use_gpu}")

        for i, match in enumerate(matches):
            seg_emo = emotion_map.get(match.segment_index)
            if seg_emo is None:
                continue

            seg_duration = (seg_emo.end - seg_emo.start) + per_clip_extra

            # Load and crop image
            img_path = images_dir / match.image_filename
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            caption_info = caption_map.get(match.image_filename)
            face_regions = caption_info.face_regions if caption_info else []
            focus_x = caption_info.focus_x if caption_info else 0.5
            focus_y = caption_info.focus_y if caption_info else 0.5
            fit_mode = caption_info.fit_mode if caption_info else "crop"

            fit_result = smart_fit(
                img, self.out_w, self.out_h,
                face_regions=face_regions,
                focus_x=focus_x,
                focus_y=focus_y,
                scale_factor=1.0,
                fit_mode=fit_mode,
            )
            canvas = fit_result.canvas

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

            clip = VideoClip(make_frame, duration=seg_duration).with_fps(self.fps)
            clips.append(clip)

            if progress_callback:
                progress_callback(i + 1, total)

        if not clips:
            raise RuntimeError("No clips generated")

        # Apply crossfades between clips
        if len(clips) > 1 and xfade_dur > 0:
            final_clip = crossfade_clips(clips, xfade_dur)
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
        final_clip.write_videofile(
            str(output_path),
            fps=self.fps,
            codec="libx264",
            audio_codec="aac",
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
