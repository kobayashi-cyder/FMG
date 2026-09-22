from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QUALITY_VERSION = "FMG-QUALITY-REFLEX-2.2"


@dataclass(frozen=True)
class QualityReport:
    score: float
    entropy: float
    detail: float
    contrast: float
    clipping: float
    resolution_ok: bool
    weak_regions: tuple[tuple[int, int, int, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": QUALITY_VERSION,
            "score": round(self.score, 4),
            "entropy": round(self.entropy, 4),
            "detail": round(self.detail, 4),
            "contrast": round(self.contrast, 4),
            "clipping": round(self.clipping, 4),
            "resolution_ok": self.resolution_ok,
            "weak_regions": [list(x) for x in self.weak_regions],
        }


class LightweightImageEvaluator:
    """Small local evaluator with no second vision model.

    Measures technical image quality only: entropy, local edge energy,
    contrast, clipping, and fixed-resolution compliance.
    """

    def __init__(self, *, width: int = 1024, height: int = 1024, grid: int = 4, max_regions: int = 3):
        self.width = int(width)
        self.height = int(height)
        self.grid = int(grid)
        self.max_regions = int(max_regions)

    @staticmethod
    def _entropy(gray) -> float:
        hist = gray.histogram()
        total = max(1, sum(hist))
        value = 0.0
        for count in hist:
            if count:
                p = count / total
                value -= p * math.log2(p)
        return max(0.0, min(1.0, value / 8.0))

    @staticmethod
    def _stats(gray) -> tuple[float, float, float]:
        from PIL import ImageFilter, ImageStat

        stat = ImageStat.Stat(gray)
        contrast = max(0.0, min(1.0, float(stat.stddev[0]) / 64.0))
        edge = gray.filter(ImageFilter.FIND_EDGES)
        detail = max(0.0, min(1.0, float(ImageStat.Stat(edge).rms[0]) / 72.0))

        hist = gray.histogram()
        total = max(1, sum(hist))
        clipped = (sum(hist[:4]) + sum(hist[252:])) / total
        clipping = max(0.0, min(1.0, clipped / 0.22))
        return detail, contrast, clipping

    def evaluate(self, path: str | Path) -> QualityReport:
        from PIL import Image

        with Image.open(Path(path)) as image:
            image = image.convert("RGB")
            resolution_ok = image.size == (self.width, self.height)
            gray = image.convert("L")
            entropy = self._entropy(gray)
            detail, contrast, clipping = self._stats(gray)
            score = (
                0.30 * entropy
                + 0.34 * detail
                + 0.26 * contrast
                + 0.10 * (1.0 if resolution_ok else 0.0)
                - 0.18 * clipping
            )
            score = max(0.0, min(1.0, score))
            weak = self._weak_regions(gray, detail, contrast)

        return QualityReport(
            score=score,
            entropy=entropy,
            detail=detail,
            contrast=contrast,
            clipping=clipping,
            resolution_ok=resolution_ok,
            weak_regions=tuple(weak),
        )

    def _weak_regions(self, gray, global_detail: float, global_contrast: float):
        from PIL import ImageFilter, ImageStat

        w, h = gray.size
        tw = w // self.grid
        th = h // self.grid
        rows: list[tuple[float, tuple[int, int, int, int]]] = []

        for gy in range(self.grid):
            for gx in range(self.grid):
                x0 = gx * tw
                y0 = gy * th
                x1 = w if gx == self.grid - 1 else (gx + 1) * tw
                y1 = h if gy == self.grid - 1 else (gy + 1) * th
                tile = gray.crop((x0, y0, x1, y1))
                stat = ImageStat.Stat(tile)
                local_contrast = max(0.0, min(1.0, float(stat.stddev[0]) / 64.0))
                edge = tile.filter(ImageFilter.FIND_EDGES)
                local_detail = max(0.0, min(1.0, float(ImageStat.Stat(edge).rms[0]) / 72.0))
                local = 0.58 * local_detail + 0.42 * local_contrast
                rows.append((local, (x0, y0, x1, y1)))

        rows.sort(key=lambda x: x[0])
        global_ref = 0.58 * global_detail + 0.42 * global_contrast
        cutoff = min(0.24, global_ref * 0.46)
        return [box for local, box in rows if local < cutoff][: self.max_regions]

    def build_mask(self, report: QualityReport, *, padding: int = 48, blur: int = 18):
        from PIL import Image, ImageDraw, ImageFilter

        mask = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(mask)
        for x0, y0, x1, y1 in report.weak_regions:
            draw.rectangle(
                (
                    max(0, x0 - padding),
                    max(0, y0 - padding),
                    min(self.width, x1 + padding),
                    min(self.height, y1 + padding),
                ),
                fill=255,
            )
        if blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(blur))
        return mask
