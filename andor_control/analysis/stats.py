"""Small numerical utilities for frame, ROI and histogram diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(slots=True)
class FrameStatistics:
    shape: tuple[int, int]
    minimum: int
    maximum: int
    mean: float
    median: float
    std: float
    p01: float
    p99: float
    total: int
    saturated_14bit: int
    saturated_16bit: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data


def frame_statistics(frame: np.ndarray) -> FrameStatistics:
    arr = np.asarray(frame)
    if arr.size == 0:
        raise ValueError("Cannot compute statistics for an empty frame")
    p01, p99 = np.percentile(arr, [1, 99])
    return FrameStatistics(
        shape=tuple(int(v) for v in arr.shape),
        minimum=int(arr.min()),
        maximum=int(arr.max()),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        std=float(arr.std()),
        p01=float(p01),
        p99=float(p99),
        total=int(np.asarray(arr, dtype=np.int64).sum()),
        saturated_14bit=int(np.count_nonzero(arr >= 16383)),
        saturated_16bit=int(np.count_nonzero(arr >= 65535)),
    )


def clip_roi_to_frame(frame: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> tuple[slice, slice]:
    height, width = frame.shape[:2]
    xa = max(0, min(int(x0), width))
    xb = max(0, min(int(x1), width))
    ya = max(0, min(int(y0), height))
    yb = max(0, min(int(y1), height))
    if xb <= xa:
        xb = min(width, xa + 1)
    if yb <= ya:
        yb = min(height, ya + 1)
    return slice(ya, yb), slice(xa, xb)


def roi_frame(frame: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    ys, xs = clip_roi_to_frame(frame, x0, x1, y0, y1)
    return np.asarray(frame)[ys, xs]
