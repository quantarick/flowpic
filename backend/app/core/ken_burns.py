"""Static frame renderer: display images without zoom or pan.

Images are shown at their original composition with no camera movement.
GPU path uses grid_sample for consistent quality; CPU path uses simple resize.
"""

import numpy as np
import torch
import torch.nn.functional as F

import cv2

from app.models import FaceRegion, KenBurnsParams

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_use_gpu = torch.cuda.is_available()


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
    ) -> KenBurnsParams:
        """Generate static params — no zoom, no pan."""
        return KenBurnsParams(
            zoom_start=1.0,
            zoom_end=1.0,
            pan_x_start=0.0,
            pan_x_end=0.0,
            pan_y_start=0.0,
            pan_y_end=0.0,
            face_center=content_center,
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
        """Static render — just resize source to output size via grid_sample."""
        _, _, src_h, src_w = source_gpu.shape
        out_w, out_h = self.output_w, self.output_h

        # Identity mapping — source maps 1:1 to output
        gy = torch.linspace(0, 1, out_h, device=source_gpu.device)
        gx = torch.linspace(0, 1, out_w, device=source_gpu.device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

        # Map to source coordinates
        src_x = grid_x * (src_w - 1)
        src_y = grid_y * (src_h - 1)
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
        """Static render — resize source to output size."""
        out_w, out_h = self.output_w, self.output_h
        src_h, src_w = source.shape[:2]

        if src_w == out_w and src_h == out_h:
            return source.copy()

        return cv2.resize(source, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
