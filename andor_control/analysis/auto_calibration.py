"""Quick auto-calibration utilities for EMCCD operation.

This is intentionally a practical laboratory assistant, not a replacement for a
complete photon-transfer calibration.  It estimates bias/dark statistics, hot
pixels and an initial photon-counting threshold from shutter-closed frames.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

import numpy as np

from .calibration import CalibrationFrame


@dataclass(slots=True)
class AutoCalibrationProfile:
    camera_model: str | None
    serial_number: int | None
    timestamp_unix_s: float
    n_bias_frames: int
    n_dark_frames: int
    bias_mean_adu: float | None
    bias_std_adu: float | None
    dark_mean_adu: float | None
    dark_std_adu: float | None
    corrected_dark_median_adu: float | None
    corrected_dark_sigma_adu: float | None
    suggested_threshold_adu: float | None
    threshold_sigma: float
    false_event_fraction_dark: float | None
    hot_pixel_count: int
    hot_pixel_fraction: float
    hot_pixel_threshold_adu: float | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def robust_sigma(frame: np.ndarray) -> float:
    arr = np.asarray(frame, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad > 0:
        return float(1.4826 * mad)
    return float(np.std(arr))


def build_auto_calibration_profile(
    *,
    bias: CalibrationFrame | None,
    dark: CalibrationFrame | None,
    camera_model: str | None = None,
    serial_number: int | None = None,
    threshold_sigma: float = 5.0,
    metadata: dict[str, Any] | None = None,
    timestamp_unix_s: float = 0.0,
) -> AutoCalibrationProfile:
    bias_frame = None if bias is None else np.asarray(bias.frame, dtype=np.float32)
    dark_frame = None if dark is None else np.asarray(dark.frame, dtype=np.float32)

    if dark_frame is not None and bias_frame is not None and dark_frame.shape == bias_frame.shape:
        corrected_dark = dark_frame - bias_frame
    elif dark_frame is not None:
        corrected_dark = dark_frame
    elif bias_frame is not None:
        corrected_dark = bias_frame - np.median(bias_frame)
    else:
        corrected_dark = None

    if corrected_dark is not None:
        median = float(np.median(corrected_dark))
        sigma = robust_sigma(corrected_dark)
        suggested = float(median + float(threshold_sigma) * sigma)
        false_fraction = float(np.mean(corrected_dark >= suggested)) if corrected_dark.size else 0.0
        hot_threshold = float(median + 8.0 * sigma)
        hot_mask = corrected_dark >= hot_threshold
        hot_count = int(np.count_nonzero(hot_mask))
        hot_fraction = float(hot_count / corrected_dark.size) if corrected_dark.size else 0.0
    else:
        median = sigma = suggested = false_fraction = hot_threshold = None
        hot_count = 0
        hot_fraction = 0.0

    return AutoCalibrationProfile(
        camera_model=camera_model,
        serial_number=serial_number,
        timestamp_unix_s=float(timestamp_unix_s),
        n_bias_frames=0 if bias is None else int(bias.n_frames),
        n_dark_frames=0 if dark is None else int(dark.n_frames),
        bias_mean_adu=None if bias_frame is None else float(np.mean(bias_frame)),
        bias_std_adu=None if bias_frame is None else float(np.std(bias_frame)),
        dark_mean_adu=None if dark_frame is None else float(np.mean(dark_frame)),
        dark_std_adu=None if dark_frame is None else float(np.std(dark_frame)),
        corrected_dark_median_adu=median,
        corrected_dark_sigma_adu=sigma,
        suggested_threshold_adu=suggested,
        threshold_sigma=float(threshold_sigma),
        false_event_fraction_dark=false_fraction,
        hot_pixel_count=hot_count,
        hot_pixel_fraction=hot_fraction,
        hot_pixel_threshold_adu=hot_threshold,
        metadata=metadata or {},
    )


def save_auto_calibration_profile(path: str | Path, profile: AutoCalibrationProfile) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(profile.to_dict(), fh, indent=2)
    return path
