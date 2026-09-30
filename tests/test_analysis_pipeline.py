import numpy as np

from andor_control.analysis.calibration import CalibrationSet, make_calibration_frame, apply_calibration
from andor_control.analysis.photon_counting import PhotonCountingSettings, photon_count_frame, python_threshold_count
from andor_control.analysis.stats import frame_statistics, roi_frame


def test_calibration_and_photon_counting_pipeline():
    raw = np.array([[100, 105, 120], [95, 150, 250]], dtype=np.int32)
    dark_stack = np.stack([np.full(raw.shape, 100, dtype=np.int32), np.full(raw.shape, 102, dtype=np.int32)])
    calibration = CalibrationSet(dark=make_calibration_frame("dark", dark_stack))
    corrected = apply_calibration(raw, calibration, subtract_dark=True, clip_negative=True)
    assert corrected.shape == raw.shape
    assert corrected[0, 0] == 0
    counted = python_threshold_count(corrected, [10, 50, 100])
    assert counted.tolist() == [[0, 0, 1], [0, 1, 3]]
    result = photon_count_frame(corrected, PhotonCountingSettings(enabled=True, thresholds_adu=(10, 50, 100)))
    assert result.total_events == 5
    assert result.active_pixels == 3


def test_stats_and_roi():
    frame = np.arange(25, dtype=np.int32).reshape(5, 5)
    sub = roi_frame(frame, 1, 4, 2, 5)
    assert sub.shape == (3, 3)
    stats = frame_statistics(sub)
    assert stats.minimum == 11
    assert stats.maximum == 23
    assert stats.total == int(sub.sum())

from andor_control.analysis.auto_calibration import build_auto_calibration_profile
from andor_control.analysis.units import UnitConversionSettings, convert_display_frame


def test_auto_calibration_profile_and_units():
    raw = np.full((4, 4), 110, dtype=np.int32)
    bias = make_calibration_frame("bias", np.stack([np.full((4, 4), 100, dtype=np.int32) for _ in range(3)]))
    dark = make_calibration_frame("dark", np.stack([np.full((4, 4), 102, dtype=np.int32) for _ in range(3)]))
    profile = build_auto_calibration_profile(bias=bias, dark=dark, threshold_sigma=5.0)
    assert profile.suggested_threshold_adu is not None
    corrected = apply_calibration(raw, CalibrationSet(bias=bias), subtract_bias=True)
    result = convert_display_frame(raw=raw, corrected=corrected, photon_counted=None, settings=UnitConversionSettings(display_unit="Corrected ADU"))
    assert result.unit_label == "corrected ADU"
    assert float(result.frame.mean()) == 10.0
