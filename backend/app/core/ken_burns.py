"""Ken Burns engine: zoom for landscape crops, vertical pan for portrait images.

Landscape crops get a subtle alternating zoom (3-7% based on arousal).
Portrait images (via pan_fit oversized canvas) get a smooth vertical pan
that reveals the full image, with face-aware direction and speed limiting.
"""

import numpy as np
import torch
import torch.nn.functional as F

import cv2

from app.models import FaceRegion, KenBurnsParams

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_use_gpu = torch.cuda.is_available()


def _cubic_ease_in_out(t: float) -> float:
    """Smooth cubic ease-in-out for natural motion."""
    if t < 0.5:
        return 4.0 * t * t * t
    else:
        return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


class KenBurnsEngine:
    def __init__(self, output_w: int, output_h: int):
        self.output_w = output_w
        self.output_h = output_h

    def generate_params(
        self,
        segment_index: int,
        arousal: float,
        face_regions: list[FaceRegion],
        source_w: int,
        source_h: int,
        content_center: tuple[float, float] = (0.5, 0.5),
        max_zoom_pct: float = 0.07,
    ) -> KenBurnsParams:
        """Generate subtle zoom params for landscape crops.

        Source is already cropped to ~1.08x output size by smart_fit.
        We alternate zoom-in / zoom-out per segment.
        Zoom range: 3-max_zoom_pct based on arousal (0-10 scale).
        """
        # Zoom magnitude: 3% at arousal=0, max_zoom_pct at arousal=10
        zoom_pct = 0.03 + (arousal / 10.0) * (max_zoom_pct - 0.03)
        zoom_pct = max(0.03, min(max_zoom_pct, zoom_pct))

        # Alternate direction per segment
        if segment_index % 2 == 0:
            # Zoom in: start at 1.0, end at 1.0+zoom_pct
            zoom_start = 1.0
            zoom_end = 1.0 + zoom_pct
        else:
            # Zoom out: start zoomed, end at 1.0
            zoom_start = 1.0 + zoom_pct
            zoom_end = 1.0

        return KenBurnsParams(
            zoom_start=zoom_start,
            zoom_end=zoom_end,
            pan_x_start=0.0,
            pan_x_end=0.0,
            pan_y_start=0.0,
            pan_y_end=0.0,
            face_center=content_center,
        )

    def generate_pan_params(
        self,
        segment_index: int,
        clip_duration: float,
        canvas_h: int,
        canvas_w: int,
        face_regions: list[FaceRegion] | None = None,
        pan_axis: str = "vertical",
    ) -> KenBurnsParams:
        """Generate pan params for portrait images in oversized canvas.

        pan_y values are viewport top-left position as fraction of canvas height.
        pan_y=0.0: viewport at top, pan_y=max_pan: viewport at bottom.
        Speed limit: ~400px/sec at 1080p, scaled proportionally.

        For very short clips (<3s) with very tall portraits (>2.5x output),
        fall back to center with subtle zoom instead of pan.
        """
        out_h = self.output_h
        out_w = self.output_w

        if pan_axis == "vertical":
            max_pan = max(0.0, (canvas_h - out_h) / canvas_h)

            # Very short clip + very tall image → center zoom fallback
            aspect_ratio = canvas_h / max(out_h, 1)
            if clip_duration < 3.0 and aspect_ratio > 2.5:
                center_y = max_pan / 2.0
                return KenBurnsParams(
                    zoom_start=1.0,
                    zoom_end=1.03,
                    pan_x_start=0.0,
                    pan_x_end=0.0,
                    pan_y_start=center_y,
                    pan_y_end=center_y,
                    face_center=(0.5, 0.5),
                )

            # Speed limit: 400px/sec at 1080p, scale proportionally
            max_pan_pixels = 400.0 * clip_duration * (out_h / 1080.0)
            actual_pan_pixels = max_pan * canvas_h

            if actual_pan_pixels > max_pan_pixels and actual_pan_pixels > 0:
                # Scale down pan range
                limited_pan = max_pan * (max_pan_pixels / actual_pan_pixels)
            else:
                limited_pan = max_pan

            # Determine direction based on face positions
            go_top_to_bottom = (segment_index % 2 == 0)  # default alternating

            if face_regions:
                face_centers_y = [(f.y + f.h / 2) / canvas_h for f in face_regions]
                avg_face_y = sum(face_centers_y) / len(face_centers_y)

                if avg_face_y < 0.4:
                    # Faces in top 40% → start from top (show face first)
                    go_top_to_bottom = True
                elif avg_face_y > 0.6:
                    # Faces in bottom 40% → start from bottom (show face first)
                    go_top_to_bottom = False

            if go_top_to_bottom:
                # Center the pan range: start from top portion, pan down
                pan_start = (max_pan - limited_pan) / 2.0
                pan_end = pan_start + limited_pan
            else:
                # Start from bottom, pan up
                pan_end_pos = max_pan - (max_pan - limited_pan) / 2.0
                pan_start = pan_end_pos
                pan_end = pan_end_pos - limited_pan

            return KenBurnsParams(
                zoom_start=1.0,
                zoom_end=1.0,
                pan_x_start=0.0,
                pan_x_end=0.0,
                pan_y_start=pan_start,
                pan_y_end=pan_end,
                face_center=(0.5, 0.5),
            )
        else:
            # Horizontal pan (rare case)
            max_pan = max(0.0, (canvas_w - out_w) / canvas_w)
            if segment_index % 2 == 0:
                return KenBurnsParams(
                    zoom_start=1.0, zoom_end=1.0,
                    pan_x_start=0.0, pan_x_end=max_pan,
                    pan_y_start=0.0, pan_y_end=0.0,
                    face_center=(0.5, 0.5),
                )
            else:
                return KenBurnsParams(
                    zoom_start=1.0, zoom_end=1.0,
                    pan_x_start=max_pan, pan_x_end=0.0,
                    pan_y_start=0.0, pan_y_end=0.0,
                    face_center=(0.5, 0.5),
                )

    def upload_source(self, source: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(source.copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return t.to(_device)

    def render_frame_gpu(
        self,
        source_gpu: torch.Tensor,
        params: KenBurnsParams,
        progress: float,
    ) -> np.ndarray:
        """Render a frame by extracting a viewport from the source using zoom/pan.

        For zoom: viewport shrinks/grows centered on content_center.
        For pan: viewport moves across the oversized canvas.
        """
        _, _, src_h, src_w = source_gpu.shape
        out_w, out_h = self.output_w, self.output_h

        t = _cubic_ease_in_out(progress)

        # Interpolate zoom
        zoom = params.zoom_start + (params.zoom_end - params.zoom_start) * t

        # Interpolate pan
        pan_x = params.pan_x_start + (params.pan_x_end - params.pan_x_start) * t
        pan_y = params.pan_y_start + (params.pan_y_end - params.pan_y_start) * t

        # Viewport size in source pixels (inverse of zoom)
        vp_w = out_w / zoom
        vp_h = out_h / zoom

        # Viewport center: start from content center, offset by pan
        cx = params.face_center[0] if params.face_center else 0.5
        cy = params.face_center[1] if params.face_center else 0.5

        # Pan offsets are in fraction of source dimensions
        vp_center_x = cx * src_w + pan_x * src_w
        vp_center_y = cy * src_h + pan_y * src_h

        # Viewport bounds
        vp_left = vp_center_x - vp_w / 2
        vp_top = vp_center_y - vp_h / 2

        # Clamp to source bounds
        vp_left = max(0, min(vp_left, src_w - vp_w))
        vp_top = max(0, min(vp_top, src_h - vp_h))
        vp_right = vp_left + vp_w
        vp_bottom = vp_top + vp_h

        # Build sampling grid
        gy = torch.linspace(0, 1, out_h, device=source_gpu.device)
        gx = torch.linspace(0, 1, out_w, device=source_gpu.device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

        # Map to source pixel coordinates within viewport
        src_x = vp_left + grid_x * (vp_right - vp_left)
        src_y = vp_top + grid_y * (vp_bottom - vp_top)

        # Normalize to [-1, 1] for grid_sample
        norm_x = 2.0 * src_x / (src_w - 1) - 1.0
        norm_y = 2.0 * src_y / (src_h - 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

        with torch.no_grad():
            out = F.grid_sample(
                source_gpu, grid,
                mode="bilinear", padding_mode="zeros", align_corners=True,
            )

        return (out[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()

    def render_frame(
        self,
        source: np.ndarray,
        params: KenBurnsParams,
        progress: float,
    ) -> np.ndarray:
        """CPU path: extract viewport via crop + LANCZOS4 resize."""
        src_h, src_w = source.shape[:2]
        out_w, out_h = self.output_w, self.output_h

        t = _cubic_ease_in_out(progress)

        zoom = params.zoom_start + (params.zoom_end - params.zoom_start) * t
        pan_x = params.pan_x_start + (params.pan_x_end - params.pan_x_start) * t
        pan_y = params.pan_y_start + (params.pan_y_end - params.pan_y_start) * t

        vp_w = out_w / zoom
        vp_h = out_h / zoom

        cx = params.face_center[0] if params.face_center else 0.5
        cy = params.face_center[1] if params.face_center else 0.5

        vp_center_x = cx * src_w + pan_x * src_w
        vp_center_y = cy * src_h + pan_y * src_h

        vp_left = vp_center_x - vp_w / 2
        vp_top = vp_center_y - vp_h / 2

        vp_left = max(0, min(vp_left, src_w - vp_w))
        vp_top = max(0, min(vp_top, src_h - vp_h))

        x1 = int(vp_left)
        y1 = int(vp_top)
        x2 = int(min(x1 + vp_w, src_w))
        y2 = int(min(y1 + vp_h, src_h))

        crop = source[y1:y2, x1:x2]

        if crop.shape[1] == out_w and crop.shape[0] == out_h:
            return crop.copy()

        return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
