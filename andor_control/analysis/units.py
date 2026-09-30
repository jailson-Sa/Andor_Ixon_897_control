"""Scientific display-unit conversion helpers for EMCCD frames.

The camera delivers raw ADU/counts.  This module keeps unit conversion explicit:
raw/corrected ADU are direct numerical transforms, SDK count-convert values are
labeled as SDK estimates, and manual photons/electrons remain calibrated
estimates rather than absolute sample emission rates.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(slots=True)
class UnitConversionSettings:
    display_unit: str = "Raw ADU"
    wavelength_nm: float = 650.0
    manual_electrons_per_adu: float = 1.0
    manual_qe_fraction: float = 0.90
    use_em_gain_for_manual: bool = True
    prefer_sdk_count_convert: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UnitConversionResult:
    frame: np.ndarray
    unit_label: str
    method: str
    metadata: dict[str, Any]


def _safe_em_gain(metadata: dict[str, Any] | None) -> float:
    if not metadata:
        return 1.0
    try:
        state = metadata.get("camera_state", {}) if isinstance(metadata, dict) else {}
        emccd = state.get("emccd", {}) if isinstance(state, dict) else {}
        gain = float(emccd.get("em_gain") or 1.0)
        return max(gain, 1.0)
    except Exception:
        return 1.0


def convert_display_frame(
    *,
    raw: np.ndarray | None,
    corrected: np.ndarray | None,
    photon_counted: np.ndarray | None,
    settings: UnitConversionSettings,
    sdk: object | None = None,
    metadata: dict[str, Any] | None = None,
) -> UnitConversionResult:
    """Return the frame to display together with unit metadata.

    Supported display units:
    - Raw ADU
    - Corrected ADU
    - Photon-counted events
    - Manual estimated electrons
    - Manual estimated photons at sensor
    - SDK estimated electrons
    - SDK estimated photons at sensor
    """
    unit = settings.display_unit
    if raw is None:
        raise ValueError("No raw frame is available")

    base_corr = corrected if corrected is not None else raw

    if unit == "Raw ADU":
        arr = np.asarray(raw)
        return UnitConversionResult(arr, "ADU", "raw_camera_counts", {"display_unit": unit})

    if unit == "Corrected ADU":
        arr = np.asarray(base_corr, dtype=np.float32)
        return UnitConversionResult(arr, "corrected ADU", "bias_dark_background_corrected", {"display_unit": unit})

    if unit == "Photon-counted events":
        if photon_counted is None:
            raise ValueError("Photon-counted frame is not available. Enable/process photon counting first.")
        arr = np.asarray(photon_counted, dtype=np.int32)
        return UnitConversionResult(arr, "events / pixel / frame", "threshold_photon_counting", {"display_unit": unit})

    if unit == "SDK estimated electrons":
        if sdk is not None and hasattr(sdk, "count_convert_frame"):
            arr = sdk.count_convert_frame(np.asarray(raw, dtype=np.int32), mode=1, wavelength_nm=float(settings.wavelength_nm))
            return UnitConversionResult(
                np.asarray(arr, dtype=np.float32),
                "SDK estimated electrons",
                "andor_sdk_count_convert_electrons",
                {"display_unit": unit, "wavelength_nm": settings.wavelength_nm},
            )
        # Manual fallback: corrected ADU * e-/ADU, with optional EM-gain compensation.
        gain = _safe_em_gain(metadata) if settings.use_em_gain_for_manual else 1.0
        arr = np.asarray(base_corr, dtype=np.float32) * float(settings.manual_electrons_per_adu) / gain
        return UnitConversionResult(
            arr,
            "estimated e-",
            "manual_electrons_per_adu_fallback",
            {
                "display_unit": unit,
                "manual_electrons_per_adu": settings.manual_electrons_per_adu,
                "em_gain_used": gain,
                "warning": "Manual estimate; not an absolute calibration.",
            },
        )

    if unit == "SDK estimated photons":
        if sdk is not None and hasattr(sdk, "count_convert_frame"):
            arr = sdk.count_convert_frame(np.asarray(raw, dtype=np.int32), mode=2, wavelength_nm=float(settings.wavelength_nm))
            return UnitConversionResult(
                np.asarray(arr, dtype=np.float32),
                "SDK estimated photons at sensor",
                "andor_sdk_count_convert_photons",
                {"display_unit": unit, "wavelength_nm": settings.wavelength_nm},
            )
        gain = _safe_em_gain(metadata) if settings.use_em_gain_for_manual else 1.0
        qe = max(float(settings.manual_qe_fraction), 1e-9)
        electrons = np.asarray(base_corr, dtype=np.float32) * float(settings.manual_electrons_per_adu) / gain
        arr = electrons / qe
        return UnitConversionResult(
            arr,
            "estimated photons at sensor",
            "manual_electrons_per_adu_qe_fallback",
            {
                "display_unit": unit,
                "manual_electrons_per_adu": settings.manual_electrons_per_adu,
                "manual_qe_fraction": settings.manual_qe_fraction,
                "em_gain_used": gain,
                "wavelength_nm": settings.wavelength_nm,
                "warning": "Manual estimate; not an absolute calibration of emitted/scattered photons.",
            },
        )

    if unit == "Manual estimated electrons":
        gain = _safe_em_gain(metadata) if settings.use_em_gain_for_manual else 1.0
        arr = np.asarray(base_corr, dtype=np.float32) * float(settings.manual_electrons_per_adu) / gain
        return UnitConversionResult(
            arr,
            "estimated e-",
            "manual_electrons_per_adu",
            {"display_unit": unit, "manual_electrons_per_adu": settings.manual_electrons_per_adu, "em_gain_used": gain},
        )

    if unit == "Manual estimated photons":
        gain = _safe_em_gain(metadata) if settings.use_em_gain_for_manual else 1.0
        qe = max(float(settings.manual_qe_fraction), 1e-9)
        electrons = np.asarray(base_corr, dtype=np.float32) * float(settings.manual_electrons_per_adu) / gain
        arr = electrons / qe
        return UnitConversionResult(
            arr,
            "estimated photons at sensor",
            "manual_electrons_per_adu_qe",
            {
                "display_unit": unit,
                "manual_electrons_per_adu": settings.manual_electrons_per_adu,
                "manual_qe_fraction": settings.manual_qe_fraction,
                "em_gain_used": gain,
                "wavelength_nm": settings.wavelength_nm,
            },
        )

    raise ValueError(f"Unknown display unit: {unit}")
