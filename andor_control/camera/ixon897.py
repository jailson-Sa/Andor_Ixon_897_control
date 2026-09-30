"""High-level Andor iXon 897 camera controller built on SDK2."""
from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from andor_control.camera.settings import AcquisitionSettings, EMCCDSettings, ROI, ShutterSettings, TemperatureSettings
from andor_control.sdk2.bindings import AndorSDK2
from andor_control.sdk2.enums import AcquisitionMode, EMGainMode, ReadMode, ShutterMode, TriggerMode
from andor_control.sdk2.errors import DRV_ACQUIRING, DRV_IDLE


@dataclass(slots=True)
class CameraIdentity:
    model: str
    serial_number: int
    detector_shape: tuple[int, int]
    pixel_size_um: tuple[float, float]
    hardware_version: dict[str, int]
    software_version: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "serial_number": self.serial_number,
            "detector_shape": self.detector_shape,
            "pixel_size_um": self.pixel_size_um,
            "hardware_version": self.hardware_version,
            "software_version": self.software_version,
        }


@dataclass(slots=True)
class TemperatureState:
    temperature_c: float
    status: str
    sdk_code: int
    cooler_on: bool

    @property
    def stabilized(self) -> bool:
        return self.status == "stabilized"

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature_c": self.temperature_c,
            "status": self.status,
            "sdk_code": self.sdk_code,
            "cooler_on": self.cooler_on,
        }


class AndorIXon897:
    """Professional-grade high-level control layer for an iXon 897.

    This class does not own a GUI.  It is meant to be safe to use from scripts,
    tests, a CLI diagnostic tool, and later from a PySide/Qt worker thread.
    """

    def __init__(self, sdk: AndorSDK2, *, camera_index: int = 0) -> None:
        self.sdk = sdk
        self.camera_index = camera_index
        self.identity: CameraIdentity | None = None
        self.current_roi: ROI | None = None
        self.current_acquisition: AcquisitionSettings | None = None
        self.current_emccd: EMCCDSettings | None = None
        self.current_shutter: ShutterSettings | None = None
        self._initialized = False

    @classmethod
    def from_dll(cls, dll_path: str | Path | None = None, *, initialization_directory: str = "", camera_index: int = 0) -> "AndorIXon897":
        return cls(AndorSDK2(dll_path=dll_path, initialization_directory=initialization_directory), camera_index=camera_index)

    def initialize(self) -> CameraIdentity:
        self.sdk.initialize()
        n_cameras = self.sdk.get_available_cameras()
        if n_cameras <= 0:
            raise RuntimeError("No Andor camera detected by SDK2")
        if self.camera_index >= n_cameras:
            raise RuntimeError(f"Requested camera_index={self.camera_index}, but only {n_cameras} camera(s) found")
        handle = self.sdk.get_camera_handle(self.camera_index)
        self.sdk.set_current_camera(handle)
        self.identity = CameraIdentity(
            model=self.sdk.get_head_model(),
            serial_number=self.sdk.get_camera_serial_number(),
            detector_shape=self.sdk.get_detector(),
            pixel_size_um=self.sdk.get_pixel_size_um(),
            hardware_version=self.sdk.get_hardware_version(),
            software_version=self.sdk.get_software_version(),
        )
        self._initialized = True
        return self.identity

    def shutdown(self, *, turn_cooler_off: bool = False, warmup_wait_s: float = 0.0) -> None:
        """Stop acquisition and shut down the SDK.

        By default the cooler is not forcibly turned off.  This is safer in a
        lab environment where controlled warm-up may be managed externally.
        """
        with contextlib.suppress(Exception):
            if self.sdk.is_acquiring():
                self.sdk.abort_acquisition()
        with contextlib.suppress(Exception):
            self.sdk.free_internal_memory()
        if turn_cooler_off:
            with contextlib.suppress(Exception):
                self.sdk.cooler_off()
            if warmup_wait_s > 0:
                time.sleep(warmup_wait_s)
        self.sdk.shutdown()
        self._initialized = False

    def __enter__(self) -> "AndorIXon897":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(turn_cooler_off=False)

    # Diagnostics ----------------------------------------------------------
    def require_initialized(self) -> None:
        if not self._initialized or self.identity is None:
            raise RuntimeError("Camera is not initialized")

    def get_identity(self) -> CameraIdentity:
        self.require_initialized()
        assert self.identity is not None
        return self.identity

    def get_temperature_state(self) -> TemperatureState:
        temp, status, code = self.sdk.get_temperature_c()
        return TemperatureState(temp, status, code, self.sdk.is_cooler_on())

    def get_status(self) -> str:
        return self.sdk.get_status()[0]

    def get_capability_summary(self) -> dict[str, Any]:
        self.require_initialized()
        ad_channels = self.sdk.get_number_ad_channels()
        preamps = self.sdk.get_number_preamp_gains()
        hs_speeds = []
        for i in range(self.sdk.get_number_hs_speeds(0, 0)):
            hs_speeds.append(self.sdk.get_hs_speed_mhz(i, 0, 0))
        vs_speeds = []
        for i in range(self.sdk.get_number_vs_speeds()):
            vs_speeds.append(self.sdk.get_vs_speed_us(i))
        summary = {
            "ad_channels": ad_channels,
            "preamp_gains": [self.sdk.get_preamp_gain(i) for i in range(preamps)],
            "hs_speeds_mhz_output_amp_0": hs_speeds,
            "vs_speeds_us": vs_speeds,
            "em_gain_range": self.sdk.get_em_gain_range(),
            "bit_depth_ad0": self.sdk.get_bit_depth(0),
            "temperature_range_c": self.sdk.get_temperature_range_c(),
        }
        if hasattr(self.sdk, "has_internal_mechanical_shutter"):
            try:
                summary["has_internal_mechanical_shutter"] = self.sdk.has_internal_mechanical_shutter()
            except Exception:
                summary["has_internal_mechanical_shutter"] = None
        if hasattr(self.sdk, "get_shutter_min_times_ms"):
            try:
                summary["shutter_min_times_ms"] = self.sdk.get_shutter_min_times_ms()
            except Exception:
                summary["shutter_min_times_ms"] = None
        return summary

    # Safe configuration ---------------------------------------------------
    def configure_temperature(self, settings: TemperatureSettings) -> TemperatureState:
        self.require_initialized()
        low, high = self.sdk.get_temperature_range_c()
        if not (low <= settings.setpoint_c <= high):
            raise ValueError(f"Requested setpoint {settings.setpoint_c} °C outside SDK range {low}..{high} °C")
        self.sdk.set_cooler_mode(settings.keep_cooler_on_at_shutdown)
        self.sdk.set_fan_mode(int(settings.fan_mode))
        self.sdk.set_temperature_c(settings.setpoint_c)
        if settings.cooler_on:
            self.sdk.cooler_on()
        else:
            self.sdk.cooler_off()
        return self.get_temperature_state()

    def wait_until_temperature_stable(self, *, timeout_s: float = 600.0, poll_s: float = 5.0) -> TemperatureState:
        self.require_initialized()
        start = time.monotonic()
        last = self.get_temperature_state()
        while time.monotonic() - start < timeout_s:
            last = self.get_temperature_state()
            if last.stabilized:
                return last
            time.sleep(poll_s)
        raise TimeoutError(f"Temperature did not stabilize within {timeout_s:.1f} s. Last state: {last.to_dict()}")

    def configure_emccd(
        self,
        settings: EMCCDSettings,
        *,
        allow_high_gain_without_stable_temp: bool = False,
        clamp_gain_to_sdk_range: bool = False,
    ) -> EMCCDSettings:
        self.require_initialized()
        settings.validate()

        self.sdk.set_em_advanced(settings.em_advanced)
        self.sdk.set_em_gain_mode(int(settings.em_gain_mode))
        gain_min, gain_max = self.sdk.get_em_gain_range()

        requested_gain = settings.em_gain
        if requested_gain is None:
            actual_gain = gain_min
        else:
            actual_gain = int(requested_gain)

        if clamp_gain_to_sdk_range:
            actual_gain = max(gain_min, min(actual_gain, gain_max))
        elif not (gain_min <= actual_gain <= gain_max):
            raise ValueError(f"Requested EM gain {actual_gain} outside SDK range {gain_min}..{gain_max}")

        # The SDK minimum in REAL_GAIN mode may be greater than 1 on real iXon
        # cameras. Treat that minimum as the safest EM setting for the selected
        # mode. Only values above the SDK minimum are considered high gain here.
        if actual_gain > gain_min and not allow_high_gain_without_stable_temp:
            temp = self.get_temperature_state()
            if not temp.stabilized:
                raise RuntimeError(
                    f"Refusing to set EM gain {actual_gain} above the SDK minimum {gain_min} "
                    "before temperature is stabilized. "
                    f"Current temperature state: {temp.to_dict()}"
                )

        self.sdk.set_preamp_gain(settings.preamp_gain_index)
        self.sdk.set_hs_speed(settings.hs_speed_index, output_amplifier=settings.output_amplifier)
        self.sdk.set_vs_speed(settings.vs_speed_index)
        self.sdk.set_baseline_clamp(settings.baseline_clamp)
        if settings.baseline_offset is not None:
            self.sdk.set_baseline_offset(settings.baseline_offset)
        self.sdk.set_emccd_gain(actual_gain)
        self.current_emccd = replace(settings, em_gain=actual_gain)
        return self.current_emccd

    def configure_shutter(self, settings: ShutterSettings) -> ShutterSettings:
        ttl = 1 if settings.ttl_high_opens else 0
        self.sdk.set_shutter_ex(ttl, int(settings.internal_mode), settings.closing_ms, settings.opening_ms, int(settings.external_mode))
        self.current_shutter = settings
        return settings

    def configure_acquisition(self, settings: AcquisitionSettings) -> dict[str, Any]:
        self.require_initialized()
        settings.validate()
        identity = self.get_identity()
        roi = settings.roi or ROI(hstart=1, hend=identity.detector_shape[0], vstart=1, vend=identity.detector_shape[1])
        roi.validate(identity.detector_shape)

        # Defensive: do not change acquisition geometry while acquiring.
        if self.sdk.is_acquiring():
            raise RuntimeError("Cannot reconfigure acquisition while camera is acquiring")

        self.sdk.set_read_mode(int(settings.read_mode))
        self.sdk.set_acquisition_mode(int(settings.acquisition_mode))
        self.sdk.set_trigger_mode(int(settings.trigger_mode))
        self.sdk.set_frame_transfer_mode(settings.frame_transfer)
        self.sdk.set_exposure_time_s(settings.exposure_s)
        if settings.accumulation_cycle_s is not None:
            self.sdk.set_accumulation_cycle_time_s(settings.accumulation_cycle_s)
        if settings.kinetic_cycle_s is not None:
            self.sdk.set_kinetic_cycle_time_s(settings.kinetic_cycle_s)
        if settings.number_kinetics is not None:
            self.sdk.set_number_kinetics(settings.number_kinetics)
        if settings.number_accumulations is not None:
            self.sdk.set_number_accumulations(settings.number_accumulations)
        self.sdk.set_image(
            hbin=roi.hbin,
            vbin=roi.vbin,
            hstart=roi.hstart,
            hend=roi.hend,
            vstart=roi.vstart,
            vend=roi.vend,
        )
        timings = self.sdk.get_acquisition_timings_s()
        self.current_roi = roi
        self.current_acquisition = settings
        return {
            "requested": settings.to_metadata(),
            "actual_timings_s": {
                "exposure": timings[0],
                "accumulation_cycle": timings[1],
                "kinetic_cycle": timings[2],
            },
            "readout_time_s": self.sdk.get_readout_time_s(),
            "roi_output_shape": roi.output_shape,
        }

    # Acquisition ----------------------------------------------------------
    def acquire_single_frame(self, *, timeout_ms: int = 30000) -> np.ndarray:
        self.require_initialized()
        if self.current_roi is None:
            self.configure_acquisition(AcquisitionSettings(acquisition_mode=AcquisitionMode.SINGLE_SCAN))
        assert self.current_roi is not None
        if self.sdk.is_acquiring():
            raise RuntimeError("Camera is already acquiring")
        self.sdk.prepare_acquisition()
        self.sdk.start_acquisition()
        try:
            self.sdk.wait_for_acquisition(timeout_ms=timeout_ms)
            frame = self.sdk.get_acquired_data16(self.current_roi.output_shape)
            return frame.astype(np.int32, copy=False)
        finally:
            with contextlib.suppress(Exception):
                if self.sdk.is_acquiring():
                    self.sdk.abort_acquisition()

    def start_continuous(self) -> None:
        self.require_initialized()
        if self.current_acquisition is None or self.current_acquisition.acquisition_mode != AcquisitionMode.RUN_TILL_ABORT:
            self.configure_acquisition(AcquisitionSettings(acquisition_mode=AcquisitionMode.RUN_TILL_ABORT))
        if self.sdk.is_acquiring():
            return
        self.sdk.prepare_acquisition()
        self.sdk.start_acquisition()

    def stop_continuous(self) -> None:
        if self.sdk.is_acquiring():
            self.sdk.abort_acquisition()

    def get_latest_frame(self) -> np.ndarray:
        self.require_initialized()
        if self.current_roi is None:
            raise RuntimeError("No ROI configured")
        return self.sdk.get_most_recent_image16(self.current_roi.output_shape).astype(np.int32, copy=False)

    def get_new_frames(self) -> np.ndarray:
        self.require_initialized()
        if self.current_roi is None:
            raise RuntimeError("No ROI configured")
        first, last = self.sdk.get_number_new_images()
        if last < first:
            return np.empty((0, self.current_roi.output_shape[1], self.current_roi.output_shape[0]), dtype=np.int32)
        frames, _valid_first, _valid_last = self.sdk.get_images16(first, last, self.current_roi.output_shape)
        return frames.astype(np.int32, copy=False)


    def get_runtime_status(self) -> dict[str, Any]:
        """Return non-invasive live diagnostic values from the SDK."""
        self.require_initialized()
        data: dict[str, Any] = {
            "sdk_status": self.get_status(),
            "temperature": self.get_temperature_state().to_dict(),
        }
        for key, getter in [
            ("total_images_acquired", getattr(self.sdk, "get_total_number_images_acquired", None)),
            ("circular_buffer_size", getattr(self.sdk, "get_circular_buffer_size", None)),
            ("new_images_index", getattr(self.sdk, "get_number_new_images", None)),
            ("available_images_index", getattr(self.sdk, "get_number_available_images", None)),
        ]:
            if getter is None:
                continue
            try:
                data[key] = getter()
            except Exception as exc:
                data[key] = f"unavailable: {exc}"
        return data

    def postprocess_photon_counting(self, frame: np.ndarray, thresholds_adu: list[float] | tuple[float, ...]) -> np.ndarray:
        """Run SDK photon-counting post-processing when available."""
        self.require_initialized()
        if not hasattr(self.sdk, "postprocess_photon_counting"):
            raise RuntimeError("The active SDK backend does not provide photon-counting postprocess")
        return self.sdk.postprocess_photon_counting(np.asarray(frame, dtype=np.int32), thresholds_adu)

    # Metadata -------------------------------------------------------------
    def build_metadata(self) -> dict[str, Any]:
        self.require_initialized()
        return {
            "camera": self.get_identity().to_dict(),
            "temperature": self.get_temperature_state().to_dict(),
            "acquisition": self.current_acquisition.to_metadata() if self.current_acquisition else None,
            "emccd": self.current_emccd.to_metadata() if self.current_emccd else None,
            "shutter": self.current_shutter.to_metadata() if self.current_shutter else None,
            "roi": self.current_roi.to_metadata() if self.current_roi else None,
            "sdk_status": self.get_status(),
            "timestamp_unix_s": time.time(),
        }

    def save_metadata_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.build_metadata(), indent=2), encoding="utf-8")
