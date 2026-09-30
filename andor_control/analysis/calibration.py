"""Bias, dark and background correction utilities for scientific acquisition."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class CalibrationFrame:
    kind: str
    frame: np.ndarray
    n_frames: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("frame", None)
        data["shape"] = list(self.frame.shape)
        data["dtype"] = str(self.frame.dtype)
        data["mean"] = float(np.mean(self.frame))
        data["std"] = float(np.std(self.frame))
        return data


@dataclass(slots=True)
class CalibrationSet:
    bias: CalibrationFrame | None = None
    dark: CalibrationFrame | None = None
    background: CalibrationFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bias": self.bias.to_dict() if self.bias is not None else None,
            "dark": self.dark.to_dict() if self.dark is not None else None,
            "background": self.background.to_dict() if self.background is not None else None,
        }

    def available(self) -> list[str]:
        out: list[str] = []
        if self.bias is not None:
            out.append("bias")
        if self.dark is not None:
            out.append("dark")
        if self.background is not None:
            out.append("background")
        return out


def average_frames(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=True)
    if arr.ndim != 3 or arr.shape[0] < 1:
        raise ValueError("Expected a frame stack with shape (n, height, width)")
    return arr.astype(np.float32).mean(axis=0)


def make_calibration_frame(kind: str, frames: np.ndarray, metadata: dict[str, Any] | None = None) -> CalibrationFrame:
    arr = np.asarray(frames)
    n_frames = int(arr.shape[0]) if arr.ndim == 3 else 1
    return CalibrationFrame(kind=kind, frame=average_frames(arr), n_frames=n_frames, metadata=metadata or {})


def _validated_frame(cal_frame: CalibrationFrame | None, raw_shape: tuple[int, ...], name: str) -> np.ndarray | None:
    if cal_frame is None:
        return None
    arr = np.asarray(cal_frame.frame, dtype=np.float32)
    if arr.shape != raw_shape:
        raise ValueError(
            f"Calibration frame '{name}' has shape {arr.shape}, but the current raw frame has shape {raw_shape}. "
            "Acquire/load calibration frames with the same camera ROI and binning as the current acquisition."
        )
    return arr


def apply_calibration(
    raw: np.ndarray,
    calibration: CalibrationSet,
    *,
    subtract_bias: bool = False,
    subtract_dark: bool = False,
    subtract_background: bool = False,
    clip_negative: bool = False,
) -> np.ndarray:
    """Apply bias/dark/background correction without silently destroying signal.

    The calculation is deliberately performed in float32 so corrected ADU may be
    negative.  Background is treated as an experimental light-background frame.
    If dark/bias subtraction is also enabled, the same electronic/dark offset is
    removed from the background before subtracting it, avoiding double
    subtraction:

        processed = raw - offset - (background - offset)

    where offset is the selected bias/dark contribution. If background is used
    alone, the operation is simply raw - background.
    """
    raw_arr = np.asarray(raw, dtype=np.float32)
    out = raw_arr.copy()
    raw_shape = raw_arr.shape

    bias = _validated_frame(calibration.bias, raw_shape, "bias") if subtract_bias else None
    dark = _validated_frame(calibration.dark, raw_shape, "dark") if subtract_dark else None
    background = _validated_frame(calibration.background, raw_shape, "background") if subtract_background else None

    offset = np.zeros(raw_shape, dtype=np.float32)
    if bias is not None:
        offset += bias
    if dark is not None:
        offset += dark

    if subtract_bias or subtract_dark:
        out -= offset

    if background is not None:
        bg = background.copy()
        if subtract_bias or subtract_dark:
            bg -= offset
        out -= bg

    if clip_negative:
        np.maximum(out, 0, out=out)
    return out


def save_calibration_frame(path: str | Path, calibration_frame: CalibrationFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        kind=calibration_frame.kind,
        frame=calibration_frame.frame,
        n_frames=calibration_frame.n_frames,
        metadata=np.array([calibration_frame.metadata], dtype=object),
    )


def load_calibration_frame(path: str | Path) -> CalibrationFrame:
    data = np.load(Path(path), allow_pickle=True)
    metadata_arr = data.get("metadata")
    metadata: dict[str, Any] = {}
    if metadata_arr is not None and len(metadata_arr) > 0:
        loaded = metadata_arr[0]
        if isinstance(loaded, dict):
            metadata = loaded
    return CalibrationFrame(
        kind=str(data["kind"]),
        frame=np.asarray(data["frame"], dtype=np.float32),
        n_frames=int(data["n_frames"]),
        metadata=metadata,
    )
