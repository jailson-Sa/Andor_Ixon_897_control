"""Photon-counting post-processing for EMCCD frames.

The functions here intentionally keep the raw ADU frame separate from the
thresholded/event-counted result so the experiment can always be reprocessed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np


@dataclass(slots=True)
class PhotonCountingSettings:
    enabled: bool = False
    thresholds_adu: tuple[float, ...] = (5.0,)
    use_corrected_frame: bool = True
    use_sdk_postprocess: bool = False

    def validate(self) -> None:
        if not self.thresholds_adu:
            raise ValueError("At least one threshold is required")
        previous = -np.inf
        for threshold in self.thresholds_adu:
            if threshold < 0:
                raise ValueError("Photon-counting thresholds must be >= 0 ADU")
            if threshold <= previous:
                raise ValueError("Photon-counting thresholds must be strictly increasing")
            previous = threshold

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["thresholds_adu"] = list(self.thresholds_adu)
        return data


@dataclass(slots=True)
class PhotonCountingResult:
    counted_frame: np.ndarray
    total_events: int
    active_pixels: int
    active_fraction: float
    mean_events_per_pixel: float
    max_events_per_pixel: int
    settings: PhotonCountingSettings

    def to_metadata(self) -> dict[str, Any]:
        return {
            "settings": self.settings.to_dict(),
            "shape": list(self.counted_frame.shape),
            "total_events": self.total_events,
            "active_pixels": self.active_pixels,
            "active_fraction": self.active_fraction,
            "mean_events_per_pixel": self.mean_events_per_pixel,
            "max_events_per_pixel": self.max_events_per_pixel,
        }


def normalize_thresholds(thresholds: Iterable[float]) -> tuple[float, ...]:
    vals = tuple(float(t) for t in thresholds if t is not None)
    settings = PhotonCountingSettings(enabled=True, thresholds_adu=vals)
    settings.validate()
    return vals


def python_threshold_count(frame: np.ndarray, thresholds_adu: Iterable[float]) -> np.ndarray:
    """Return a photon-counted image using single or multiple thresholds.

    For thresholds [T1, T2, T3], a pixel receives count 0 below T1, count 1
    between T1 and T2, count 2 between T2 and T3, and count 3 above T3.
    """
    thresholds = normalize_thresholds(thresholds_adu)
    arr = np.asarray(frame, dtype=np.float32)
    out = np.zeros(arr.shape, dtype=np.int32)
    for threshold in thresholds:
        out += arr >= threshold
    return out


def photon_count_frame(
    frame: np.ndarray,
    settings: PhotonCountingSettings,
    *,
    sdk: object | None = None,
) -> PhotonCountingResult:
    settings.validate()
    if settings.use_sdk_postprocess and sdk is not None and hasattr(sdk, "postprocess_photon_counting"):
        try:
            counted = sdk.postprocess_photon_counting(np.asarray(frame, dtype=np.int32), list(settings.thresholds_adu))
        except Exception:
            # The SDK postprocess is useful when available, but the Python path
            # is deterministic and safer for keeping development portable.
            counted = python_threshold_count(frame, settings.thresholds_adu)
    else:
        counted = python_threshold_count(frame, settings.thresholds_adu)

    total = int(np.asarray(counted, dtype=np.int64).sum())
    active = int(np.count_nonzero(counted))
    pixels = int(counted.size)
    return PhotonCountingResult(
        counted_frame=counted.astype(np.int32, copy=False),
        total_events=total,
        active_pixels=active,
        active_fraction=float(active / pixels) if pixels else 0.0,
        mean_events_per_pixel=float(total / pixels) if pixels else 0.0,
        max_events_per_pixel=int(counted.max()) if counted.size else 0,
        settings=settings,
    )
