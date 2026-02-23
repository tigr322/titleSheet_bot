from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import statistics

import numpy as np

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class PostprocessConfig:
    tile_width: int
    tile_height: int
    # Alpha threshold for the primary foreground mask.
    # A slightly higher default is more robust to soft halos from background removal.
    alpha_threshold: int = 24
    # A denser threshold used only for computing a stable "mass center".
    dense_threshold: int = 96
    # How many bottom pixels to consider as "feet zone" for baseline detection.
    foot_window_px: int = 14
    # Fraction of width around the character center used to compute body bbox.
    # This avoids tools/axe/outstretched hands from affecting scale.
    central_x_ratio: float = 0.55
    margin_bottom: int = 8
    clamp_min: float = 0.85
    clamp_max: float = 1.15
    # For true pixel art, NEAREST is usually preferred.
    resample: str = "nearest"
    debug_dir: Path | None = None


@dataclass(frozen=True)
class FrameMetrics:
    full_bbox: tuple[int, int, int, int] | None
    body_bbox: tuple[int, int, int, int] | None
    body_h: int
    baseline_y: int
    baseline_offset: int
    empty: bool
    center_x_offset: float = 0.0


class AlphaMaskExtractor:
    def extract_alpha(self, img: Image.Image) -> np.ndarray:
        rgba = img.convert("RGBA")
        alpha = rgba.getchannel("A")
        return np.asarray(alpha, dtype=np.uint8)

    def build_masks(
        self, alpha: np.ndarray, threshold: int, dense_threshold: int
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = alpha > threshold
        dense = alpha > dense_threshold
        return mask, dense


class EntityBBoxFinder:
    def find(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        return (x0, y0, x1, y1)


class BaselineDetector:
    def detect_feet_median(self, mask: np.ndarray, foot_window_px: int) -> int | None:
        """AoE-style baseline: median Y of alpha pixels in the bottom "feet zone".

        Using median (not max) prevents visible baseline jitter across walk frames,
        where a single toe pixel could be the lowest point.
        """

        h, _w = mask.shape
        if h <= 0:
            return None

        y0 = max(0, h - foot_window_px)
        ys, _xs = np.where(mask[y0:h, :])
        if len(ys) > 0:
            return int(np.median(ys + y0))

        # Fallback: last row containing any alpha.
        rows = np.where(mask.any(axis=1))[0]
        if len(rows) == 0:
            return None
        return int(rows.max())


class BatchScaleTargetCalculator:
    def compute(self, heights: list[int]) -> int:
        """Robust target height (median with MAD outlier filtering)."""

        if not heights:
            return 0

        arr = np.asarray(heights, dtype=np.float32)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) + 1e-6

        # Keep values within ~2.5 MAD. This is robust for small batches.
        keep = np.abs(arr - med) < (2.5 * mad)
        arr2 = arr[keep] if bool(np.any(keep)) else arr
        return int(round(float(np.median(arr2))))


class FrameNormalizer:
    def __init__(self, config: PostprocessConfig) -> None:
        self._config = config
        self._extractor = AlphaMaskExtractor()
        self._bbox_finder = EntityBBoxFinder()
        self._baseline_detector = BaselineDetector()
        self._scale_calculator = BatchScaleTargetCalculator()
        self._resample = self._resolve_resample(config.resample)

    def normalize(self, frames: list[Image.Image]) -> list[Image.Image]:
        metrics: list[FrameMetrics] = []
        for index, frame in enumerate(frames):
            metrics.append(self._analyze_frame(frame, index))

        target_body_h = self._scale_calculator.compute(
            [m.body_h for m in metrics if not m.empty]
        )
        if target_body_h <= 0:
            target_body_h = self._config.tile_height

        tiles: list[Image.Image] = []
        for index, (frame, meta) in enumerate(zip(frames, metrics)):
            tiles.append(self._normalize_frame(frame, meta, target_body_h, index))
        return tiles

    def _analyze_frame(self, img: Image.Image, index: int) -> FrameMetrics:
        alpha = self._extractor.extract_alpha(img)
        mask, dense = self._extractor.build_masks(
            alpha, self._config.alpha_threshold, self._config.dense_threshold
        )

        # Compute a stable mass center. Dense mask is preferred to avoid halo influence.
        d_ys, d_xs = np.where(dense)
        if len(d_xs) > 0:
            center_x = float(np.mean(d_xs))
        else:
            ys, xs = np.where(mask)
            center_x = float(np.mean(xs)) if len(xs) > 0 else 0.0

        full_bbox = self._bbox_finder.find(mask)
        if not full_bbox:
            logging.info("Frame %s is empty after thresholding.", index)
            return FrameMetrics(
                full_bbox=None,
                body_bbox=None,
                body_h=0,
                baseline_y=0,
                baseline_offset=0,
                center_x_offset=0.0,
                empty=True,
            )

        # Compute body bbox using only a central X-window around the character.
        # This excludes weapon/axe/outstretched hands from affecting height.
        h, w = mask.shape
        half = int(round((w * self._config.central_x_ratio) / 2.0))
        cx_int = int(round(center_x))
        xL = max(0, cx_int - half)
        xR = min(w, cx_int + half)
        body_mask = mask[:, xL:xR] if xR > xL else mask
        bbox_local = self._bbox_finder.find(body_mask)

        if bbox_local is None:
            body_bbox = full_bbox
        else:
            x0, y0, x1, y1 = bbox_local
            body_bbox = (xL + x0, y0, xL + x1, y1)

        baseline_y = self._baseline_detector.detect_feet_median(
            mask, self._config.foot_window_px
        )
        if baseline_y is None:
            baseline_y = full_bbox[3] - 1
        body_h = body_bbox[3] - body_bbox[1]
        baseline_offset = baseline_y - full_bbox[1]
        body_center_x = (body_bbox[0] + body_bbox[2] - 1) / 2.0
        return FrameMetrics(
            full_bbox=full_bbox,
            body_bbox=body_bbox,
            body_h=body_h,
            baseline_y=baseline_y,
            baseline_offset=baseline_offset,
            center_x_offset=body_center_x - float(full_bbox[0]),
            empty=False,
        )

    def _normalize_frame(
        self,
        img: Image.Image,
        meta: FrameMetrics,
        target_body_h: int,
        index: int,
    ) -> Image.Image:
        tile = Image.new("RGBA", (self._config.tile_width, self._config.tile_height), (0, 0, 0, 0))
        if meta.empty or not meta.full_bbox:
            self._save_debug(index, img, None, None, None, tile, None, None)
            return tile

        cropped = img.convert("RGBA").crop(meta.full_bbox)
        scale = target_body_h / max(1, meta.body_h)
        scale = min(self._config.clamp_max, max(self._config.clamp_min, scale))
        new_w = max(1, int(round(cropped.width * scale)))
        new_h = max(1, int(round(cropped.height * scale)))
        resized = cropped.resize((new_w, new_h), self._resample)

        # Recompute baseline on the resized crop using the feet-median method.
        alpha_resized = np.asarray(resized.getchannel("A"), dtype=np.uint8)
        mask_resized = alpha_resized > self._config.alpha_threshold
        baseline_resized = self._baseline_detector.detect_feet_median(
            mask_resized, self._config.foot_window_px
        )
        if baseline_resized is None:
            baseline_resized = int(round(meta.baseline_offset * scale))

        target_baseline_y = self._config.tile_height - self._config.margin_bottom
        paste_y = target_baseline_y - baseline_resized

        # Stable X centering by body center (weapon-independent).
        cx_resized = meta.center_x_offset * scale
        paste_x = int(round((self._config.tile_width / 2.0) - cx_resized))

        self._paste_with_clip(tile, resized, paste_x, paste_y)
        self._save_debug(
            index,
            img,
            meta.full_bbox,
            meta.body_bbox,
            cropped,
            resized,
            tile,
            target_baseline_y,
            baseline_resized,
        )
        return tile

    def _paste_with_clip(self, canvas: Image.Image, img: Image.Image, x: int, y: int) -> None:
        if x >= canvas.width or y >= canvas.height:
            return
        if x + img.width <= 0 or y + img.height <= 0:
            return
        left = max(0, -x)
        top = max(0, -y)
        right = min(img.width, canvas.width - x)
        bottom = min(img.height, canvas.height - y)
        if left or top or right < img.width or bottom < img.height:
            img = img.crop((left, top, right, bottom))
            x = max(0, x)
            y = max(0, y)
        canvas.paste(img, (x, y), img)

    def _save_debug(
        self,
        index: int,
        original: Image.Image,
        full_bbox: tuple[int, int, int, int] | None,
        body_bbox: tuple[int, int, int, int] | None,
        cropped: Image.Image | None,
        resized: Image.Image | None,
        tile: Image.Image,
        target_baseline_y: int | None,
        baseline_resized: int | None,
    ) -> None:
        if not self._config.debug_dir:
            return
        frame_dir = self._config.debug_dir / f"frame_{index:03d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        rgba = original.convert("RGBA")
        rgba.save(frame_dir / "00_input.png")
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
        mask = alpha > self._config.alpha_threshold
        Image.fromarray((mask.astype(np.uint8) * 255)).save(frame_dir / "01_mask.png")
        overlay = rgba.copy()
        draw = ImageDraw.Draw(overlay)
        if full_bbox:
            draw.rectangle(full_bbox, outline=(0, 255, 0, 255), width=1)
        if body_bbox:
            draw.rectangle(body_bbox, outline=(255, 0, 0, 255), width=1)
        overlay.save(frame_dir / "02_bbox.png")
        if cropped:
            cropped.save(frame_dir / "03_cropped.png")
        if resized:
            resized.save(frame_dir / "04_resized.png")
        tile.save(frame_dir / "05_tile.png")
        if target_baseline_y is not None:
            debug_tile = tile.copy()
            draw = ImageDraw.Draw(debug_tile)
            draw.line(
                [(0, target_baseline_y), (self._config.tile_width, target_baseline_y)],
                fill=(255, 0, 0, 128),
                width=1,
            )
            debug_tile.save(frame_dir / "06_tile_baseline.png")

    @staticmethod
    def _resolve_resample(mode: str) -> int:
        if mode == "nearest":
            return Image.NEAREST
        if mode == "bicubic":
            return Image.BICUBIC
        raise ValueError("resample must be 'nearest' or 'bicubic'")
