"""Mock implementation of the Andor SDK2 interface for development/tests.

This backend implements the same subset of methods used by AndorIXon897, but
never loads the vendor DLL and never talks to hardware.  It is useful on Linux,
macOS, or any Windows machine without the camera connected.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .errors import DRV_ACQUIRING, DRV_IDLE


@dataclass(slots=True)
class MockState:
    initialized: bool = False
    acquiring: bool = False
    exposure_s: float = 0.03
    acquisition_mode: int = 1
    read_mode: int = 4
    trigger_mode: int = 0
    detector_shape: tuple[int, int] = (512, 512)
    roi: tuple[int, int, int, int, int, int] = (1, 1, 1, 512, 1, 512)  # hbin, vbin, hstart, hend, vstart, vend
    cooler_on: bool = False
    temperature_setpoint_c: int = -70
    temperature_c: float = 20.0
    em_gain: int = 1
    em_gain_mode: int = 3
    em_advanced: bool = False
    frame_counter: int = 0
    started_at: float = 0.0
    shutter_internal_mode: int = 1
    shutter_external_mode: int = 1
    shutter_ttl_high_opens: bool = False


class MockAndorSDK2:
    """Drop-in development replacement for AndorSDK2."""

    def __init__(self, *, noise_sigma_adu: float = 8.0, seed: int = 12345) -> None:
        self.state = MockState()
        self.noise_sigma_adu = noise_sigma_adu
        self.rng = np.random.default_rng(seed)

    def initialize(self) -> None:
        self.state.initialized = True

    def shutdown(self) -> None:
        self.state.acquiring = False
        self.state.initialized = False

    def get_available_cameras(self) -> int:
        return 1

    def get_camera_handle(self, index: int = 0) -> int:
        if index != 0:
            raise RuntimeError("Mock backend has only one camera")
        return 1

    def set_current_camera(self, handle: int) -> None:
        if handle != 1:
            raise RuntimeError("Invalid mock camera handle")

    def get_current_camera(self) -> int:
        return 1

    def get_head_model(self) -> str:
        return "Mock Andor iXon Ultra 897"

    def get_camera_serial_number(self) -> int:
        return 897000

    def get_detector(self) -> tuple[int, int]:
        return self.state.detector_shape

    def get_pixel_size_um(self) -> tuple[float, float]:
        return (16.0, 16.0)

    def get_hardware_version(self) -> dict[str, int]:
        return {"pcb": 0, "decode": 0, "firmware": 0, "firmware_build": 0}

    def get_software_version(self) -> dict[str, int]:
        return {"eprom": 0, "coffile": 0, "driver_rev": 0, "driver_ver": 0, "dll_rev": 0, "dll_ver": 0}

    def get_temperature_range_c(self) -> tuple[int, int]:
        return (-100, 20)

    def set_temperature_c(self, setpoint_c: int) -> None:
        self.state.temperature_setpoint_c = int(setpoint_c)

    def get_temperature_c(self) -> tuple[float, str, int]:
        # Simple exponential cooling model for test realism.
        if self.state.cooler_on:
            self.state.temperature_c += 0.15 * (self.state.temperature_setpoint_c - self.state.temperature_c)
        else:
            self.state.temperature_c += 0.08 * (20.0 - self.state.temperature_c)
        status = "stabilized" if abs(self.state.temperature_c - self.state.temperature_setpoint_c) < 0.5 and self.state.cooler_on else "not_reached"
        code = 20036 if status == "stabilized" else 20037
        return float(self.state.temperature_c), status, code

    def cooler_on(self) -> None:
        self.state.cooler_on = True

    def cooler_off(self) -> None:
        self.state.cooler_on = False

    def is_cooler_on(self) -> bool:
        return self.state.cooler_on

    def set_cooler_mode(self, keep_cooler_on_at_shutdown: bool) -> None:
        return None

    def set_fan_mode(self, mode: int) -> None:
        return None

    def set_acquisition_mode(self, mode: int) -> None:
        self.state.acquisition_mode = int(mode)

    def set_read_mode(self, mode: int) -> None:
        self.state.read_mode = int(mode)

    def set_trigger_mode(self, mode: int) -> None:
        self.state.trigger_mode = int(mode)

    def is_trigger_mode_available(self, mode: int) -> bool:
        return int(mode) in {0, 1, 6, 7, 10}

    def set_exposure_time_s(self, seconds: float) -> None:
        self.state.exposure_s = float(seconds)

    def set_accumulation_cycle_time_s(self, seconds: float) -> None:
        return None

    def set_kinetic_cycle_time_s(self, seconds: float) -> None:
        return None

    def set_number_kinetics(self, count: int) -> None:
        return None

    def set_number_accumulations(self, count: int) -> None:
        return None

    def get_acquisition_timings_s(self) -> tuple[float, float, float]:
        return self.state.exposure_s, max(self.state.exposure_s, 0.01), max(self.state.exposure_s, 0.01)

    def get_readout_time_s(self) -> float:
        shape = self._roi_output_shape()
        return shape[0] * shape[1] / (10_000_000.0)

    def set_image(self, *, hbin: int = 1, vbin: int = 1, hstart: int = 1, hend: int, vstart: int = 1, vend: int) -> None:
        self.state.roi = (hbin, vbin, hstart, hend, vstart, vend)

    def prepare_acquisition(self) -> None:
        return None

    def start_acquisition(self) -> None:
        self.state.acquiring = True
        self.state.started_at = time.monotonic()

    def abort_acquisition(self) -> None:
        self.state.acquiring = False

    def wait_for_acquisition(self, timeout_ms: int | None = None) -> None:
        sleep_s = min(self.state.exposure_s, 0.05)
        if timeout_ms is not None and sleep_s * 1000.0 > timeout_ms:
            raise TimeoutError("Mock acquisition timeout")
        time.sleep(sleep_s)
        self.state.frame_counter += 1

    def cancel_wait(self) -> None:
        return None

    def get_status(self) -> tuple[str, int]:
        return ("acquiring", DRV_ACQUIRING) if self.state.acquiring else ("idle", DRV_IDLE)

    def is_acquiring(self) -> bool:
        return self.state.acquiring

    def get_number_new_images(self) -> tuple[int, int]:
        elapsed = max(0.0, time.monotonic() - self.state.started_at)
        available = max(1, int(elapsed / max(self.state.exposure_s, 0.001))) if self.state.acquiring else 0
        first = self.state.frame_counter + 1
        last = self.state.frame_counter + available
        self.state.frame_counter = last
        return first, last

    def get_number_available_images(self) -> tuple[int, int]:
        return self.get_number_new_images()

    def get_total_number_images_acquired(self) -> int:
        return self.state.frame_counter

    def get_circular_buffer_size(self) -> int:
        return 1000

    def free_internal_memory(self) -> None:
        return None

    def _roi_output_shape(self) -> tuple[int, int]:
        hbin, vbin, hstart, hend, vstart, vend = self.state.roi
        return ((hend - hstart + 1) // hbin, (vend - vstart + 1) // vbin)

    def _synthetic_frame(self) -> np.ndarray:
        width, height = self._roi_output_shape()
        y, x = np.mgrid[0:height, 0:width]
        if self.state.shutter_internal_mode == 2 or self.state.shutter_external_mode == 2:
            dark = 100.0 + self.rng.normal(0, self.noise_sigma_adu, size=(height, width))
            return np.clip(dark, 0, 65535).astype(np.int16)
        t = self.state.frame_counter
        cx = width * (0.5 + 0.1 * math.sin(0.05 * t))
        cy = height * (0.5 + 0.1 * math.cos(0.043 * t))
        sigma = max(width, height) / 12.0
        spot = 1800.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        background = 100.0 + 15.0 * np.sin(2 * np.pi * x / max(width, 1))
        em_factor = max(1, self.state.em_gain)
        noise = self.rng.normal(0, self.noise_sigma_adu * math.sqrt(em_factor), size=(height, width))
        frame = background + spot + noise
        return np.clip(frame, 0, 65535).astype(np.int16)

    def get_most_recent_image16(self, shape: tuple[int, int]) -> np.ndarray:
        self.state.frame_counter += 1
        return self._synthetic_frame()

    def get_acquired_data16(self, shape: tuple[int, int]) -> np.ndarray:
        self.state.frame_counter += 1
        return self._synthetic_frame()

    def get_oldest_image16(self, shape: tuple[int, int]) -> np.ndarray:
        self.state.frame_counter += 1
        return self._synthetic_frame()

    def get_images16(self, first: int, last: int, shape: tuple[int, int]) -> tuple[np.ndarray, int, int]:
        n = max(0, last - first + 1)
        frames = []
        for _ in range(n):
            self.state.frame_counter += 1
            frames.append(self._synthetic_frame())
        return np.stack(frames, axis=0), first, last

    def set_em_gain_mode(self, mode: int) -> None:
        self.state.em_gain_mode = int(mode)

    def get_em_gain_range(self) -> tuple[int, int]:
        return (1, 300) if not self.state.em_advanced else (1, 1000)

    def set_emccd_gain(self, gain: int) -> None:
        self.state.em_gain = int(gain)

    def get_emccd_gain(self) -> int:
        return self.state.em_gain

    def set_em_advanced(self, enabled: bool) -> None:
        self.state.em_advanced = bool(enabled)

    def get_em_advanced(self) -> bool:
        return self.state.em_advanced

    def get_number_preamp_gains(self) -> int:
        return 3

    def get_preamp_gain(self, index: int) -> float:
        return [1.0, 2.4, 5.1][index]

    def set_preamp_gain(self, index: int) -> None:
        return None

    def get_number_ad_channels(self) -> int:
        return 1

    def get_number_hs_speeds(self, ad_channel: int = 0, output_amplifier: int = 0) -> int:
        return 3

    def get_hs_speed_mhz(self, index: int, ad_channel: int = 0, output_amplifier: int = 0) -> float:
        return [10.0, 5.0, 1.0][index]

    def set_hs_speed(self, index: int, output_amplifier: int = 0) -> None:
        return None

    def get_number_vs_speeds(self) -> int:
        return 3

    def get_vs_speed_us(self, index: int) -> float:
        return [0.3, 0.5, 1.0][index]

    def set_vs_speed(self, index: int) -> None:
        return None

    def set_baseline_clamp(self, enabled: bool) -> None:
        return None

    def set_baseline_offset(self, value: int) -> None:
        return None

    def get_bit_depth(self, ad_channel: int = 0) -> int:
        return 16

    def set_shutter_ex(self, typ: int, internal_mode: int, closing_ms: int, opening_ms: int, external_mode: int) -> None:
        self.state.shutter_ttl_high_opens = bool(typ)
        self.state.shutter_internal_mode = int(internal_mode)
        self.state.shutter_external_mode = int(external_mode)
        return None

    def get_shutter_min_times_ms(self) -> tuple[int, int]:
        return (0, 0)

    def has_internal_mechanical_shutter(self) -> bool:
        return True

    def send_software_trigger(self) -> None:
        self.state.frame_counter += 1

    def set_frame_transfer_mode(self, enabled: bool) -> None:
        return None
    def postprocess_photon_counting(self, frame: np.ndarray, thresholds_adu: list[float] | tuple[float, ...]) -> np.ndarray:
        arr = np.asarray(frame, dtype=np.float32)
        out = np.zeros(arr.shape, dtype=np.int32)
        for threshold in thresholds_adu:
            out += arr >= float(threshold)
        return out


# Count-convert helpers are attached to the mock class for GUI development.
def _mock_count_convert_mode_available(self, mode: int) -> bool:
    return int(mode) in {1, 2}


def _mock_get_count_convert_wavelength_range_nm(self) -> tuple[float, float]:
    return (300.0, 1100.0)


def _mock_get_qe_percent(self, wavelength_nm: float) -> float:
    wl = float(wavelength_nm)
    # Smooth approximate QE curve used only for simulation.
    return float(20.0 + 75.0 * np.exp(-((wl - 650.0) / 260.0) ** 2))


def _mock_get_sensitivity_adu_per_electron(self, ad_channel: int = 0, hs_speed_index: int = 0, output_amplifier: int = 0, preamp_index: int = 0) -> float:
    return 1.0


def _mock_count_convert_frame(self, frame: np.ndarray, *, mode: int, wavelength_nm: float, em_gain: int | None = None, sensitivity: float | None = None, qe_percent: float | None = None, baseline: int = 0, ad_channel: int = 0, hs_speed_index: int = 0, output_amplifier: int = 0, preamp_index: int = 0) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32) - float(baseline)
    gain = max(float(self.state.em_gain if em_gain is None else em_gain), 1.0)
    sens = float(1.0 if sensitivity is None else sensitivity)
    electrons = arr / max(gain * sens, 1e-9)
    if int(mode) == 1:
        return np.maximum(electrons, 0).astype(np.float32)
    qe = (self.get_qe_percent(wavelength_nm) if qe_percent is None else qe_percent) / 100.0
    return np.maximum(electrons / max(qe, 1e-9), 0).astype(np.float32)

MockAndorSDK2.count_convert_mode_available = _mock_count_convert_mode_available
MockAndorSDK2.get_count_convert_wavelength_range_nm = _mock_get_count_convert_wavelength_range_nm
MockAndorSDK2.get_qe_percent = _mock_get_qe_percent
MockAndorSDK2.get_sensitivity_adu_per_electron = _mock_get_sensitivity_adu_per_electron
MockAndorSDK2.count_convert_frame = _mock_count_convert_frame
