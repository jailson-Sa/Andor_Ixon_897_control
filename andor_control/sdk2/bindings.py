"""Minimal, explicit ctypes binding for Andor SDK2/ATMCD.

The original source in this project used Lantz.  This module deliberately uses
plain ctypes and explicitly patches the signatures required by the higher-level
camera layer.  Keep this module narrow: it should expose SDK calls safely, but
not implement experiment logic or GUI behavior.
"""
from __future__ import annotations

import ctypes as ct
import os
import platform
from pathlib import Path
from typing import Iterable

import numpy as np

from .errors import (
    DRV_ACQUIRING,
    DRV_IDLE,
    DRV_TEMP_DRIFT,
    DRV_TEMP_NOT_REACHED,
    DRV_TEMP_NOT_STABILIZED,
    DRV_TEMP_OFF,
    DRV_TEMP_OUT_RANGE,
    DRV_TEMP_STABILIZED,
    DRV_TEMP_NOT_SUPPORTED,
    STATUS,
    TEMPERATURE_STATUS,
    check_success,
    code_name,
)

TEMP_RETURN_CODES = {
    DRV_TEMP_OFF,
    DRV_TEMP_NOT_STABILIZED,
    DRV_TEMP_STABILIZED,
    DRV_TEMP_NOT_REACHED,
    DRV_TEMP_OUT_RANGE,
    DRV_TEMP_NOT_SUPPORTED,
    DRV_TEMP_DRIFT,
}


class SDK2LibraryNotFound(FileNotFoundError):
    """Raised when atmcd64d.dll cannot be located."""


class AndorSDK2:
    """Thin wrapper around the Andor SDK2 ATMCD DLL.

    Parameters
    ----------
    dll_path:
        Explicit path to ``atmcd64d.dll``.  If omitted, the loader checks the
        ``ANDOR_SDK2_DLL`` environment variable and then the normal DLL search
        path.
    initialization_directory:
        Directory passed to ``Initialize``.  Empty string lets the SDK use its
        default configuration.  Some older setups require a directory containing
        DETECTOR.INI.
    """

    def __init__(self, dll_path: str | os.PathLike[str] | None = None, initialization_directory: str = "") -> None:
        self.dll_path = self._resolve_dll_path(dll_path)
        self.initialization_directory = initialization_directory
        self.lib = self._load_library(self.dll_path)
        self._patch_signatures()
        self.initialized = False

    @staticmethod
    def _resolve_dll_path(dll_path: str | os.PathLike[str] | None) -> str:
        if dll_path is not None:
            path = Path(dll_path)
            if not path.exists():
                raise SDK2LibraryNotFound(f"DLL not found: {path}")
            return str(path)

        env = os.environ.get("ANDOR_SDK2_DLL")
        if env:
            path = Path(env)
            if not path.exists():
                raise SDK2LibraryNotFound(f"ANDOR_SDK2_DLL points to a missing file: {path}")
            return str(path)

        # Fall back to DLL search path on Windows.  This succeeds when the
        # Andor Driver Pack/SDK has installed ATMCD64D.DLL into the system path.
        return "atmcd64d.dll"

    @staticmethod
    def _load_library(path: str) -> ct.CDLL:
        if platform.system().lower() != "windows" and Path(path).name.lower().endswith(".dll"):
            raise OSError(
                "Andor SDK2 DLL loading is only supported on Windows. "
                "Use MockAndorSDK2 for development/tests without hardware."
            )
        try:
            return ct.WinDLL(path)  # type: ignore[attr-defined]
        except AttributeError:
            return ct.CDLL(path)
        except OSError as exc:
            raise SDK2LibraryNotFound(
                f"Could not load Andor SDK2 DLL '{path}'. Confirm Python is 64-bit, "
                "the Andor Driver Pack is installed, and no incompatible DLL is being loaded."
            ) from exc

    def _set_signature(self, name: str, argtypes: Iterable[object], restype: object = ct.c_uint) -> None:
        func = getattr(self.lib, name)
        func.argtypes = list(argtypes)
        func.restype = restype

    def _patch_signatures(self) -> None:
        c_int_p = ct.POINTER(ct.c_int)
        c_uint_p = ct.POINTER(ct.c_uint)
        c_long_p = ct.POINTER(ct.c_long)
        c_float_p = ct.POINTER(ct.c_float)

        # System and identification
        self._set_signature("Initialize", [ct.c_char_p])
        self._set_signature("ShutDown", [])
        self._set_signature("GetAvailableCameras", [c_long_p])
        self._set_signature("GetCameraHandle", [ct.c_long, c_long_p])
        self._set_signature("SetCurrentCamera", [ct.c_long])
        self._set_signature("GetCurrentCamera", [c_long_p])
        self._set_signature("GetHeadModel", [ct.c_char_p])
        self._set_signature("GetCameraSerialNumber", [c_uint_p])
        self._set_signature("GetDetector", [c_int_p, c_int_p])
        self._set_signature("GetPixelSize", [c_float_p, c_float_p])
        self._set_signature("GetHardwareVersion", [c_uint_p, c_uint_p, c_uint_p, c_uint_p, c_uint_p, c_uint_p])
        self._set_signature("GetSoftwareVersion", [c_uint_p, c_uint_p, c_uint_p, c_uint_p, c_uint_p, c_uint_p])

        # Temperature and cooler
        self._set_signature("GetTemperatureRange", [c_int_p, c_int_p])
        self._set_signature("SetTemperature", [ct.c_int])
        self._set_signature("GetTemperatureF", [c_float_p])
        self._set_signature("CoolerON", [])
        self._set_signature("CoolerOFF", [])
        self._set_signature("IsCoolerOn", [c_int_p])
        self._set_signature("SetCoolerMode", [ct.c_int])
        self._set_signature("SetFanMode", [ct.c_int])

        # Acquisition and image configuration
        self._set_signature("SetAcquisitionMode", [ct.c_int])
        self._set_signature("SetReadMode", [ct.c_int])
        self._set_signature("SetTriggerMode", [ct.c_int])
        self._set_signature("SetExposureTime", [ct.c_float])
        self._set_signature("SetAccumulationCycleTime", [ct.c_float])
        self._set_signature("SetKineticCycleTime", [ct.c_float])
        self._set_signature("SetNumberKinetics", [ct.c_int])
        self._set_signature("SetNumberAccumulations", [ct.c_int])
        self._set_signature("GetAcquisitionTimings", [c_float_p, c_float_p, c_float_p])
        self._set_signature("GetReadOutTime", [c_float_p])
        self._set_signature("GetKeepCleanTime", [c_float_p])
        self._set_signature("SetImage", [ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int])
        self._set_signature("PrepareAcquisition", [])
        self._set_signature("StartAcquisition", [])
        self._set_signature("AbortAcquisition", [])
        self._set_signature("WaitForAcquisition", [])
        if hasattr(self.lib, "WaitForAcquisitionTimeOut"):
            self._set_signature("WaitForAcquisitionTimeOut", [ct.c_int])
        self._set_signature("CancelWait", [])
        self._set_signature("GetStatus", [c_int_p])
        self._set_signature("GetAcquisitionProgress", [c_long_p, c_long_p])
        self._set_signature("GetTotalNumberImagesAcquired", [c_long_p])
        self._set_signature("GetSizeOfCircularBuffer", [c_long_p])
        self._set_signature("GetNumberNewImages", [c_long_p, c_long_p])
        self._set_signature("GetNumberAvailableImages", [c_long_p, c_long_p])
        self._set_signature("FreeInternalMemory", [])

        # Data retrieval.  SDK takes pointer to data + total number of pixels.
        self._set_signature("GetAcquiredData", [ct.POINTER(ct.c_int32), ct.c_ulong])
        self._set_signature("GetAcquiredData16", [ct.POINTER(ct.c_int16), ct.c_ulong])
        self._set_signature("GetMostRecentImage", [ct.POINTER(ct.c_int32), ct.c_ulong])
        self._set_signature("GetMostRecentImage16", [ct.POINTER(ct.c_int16), ct.c_ulong])
        self._set_signature("GetOldestImage", [ct.POINTER(ct.c_int32), ct.c_ulong])
        self._set_signature("GetOldestImage16", [ct.POINTER(ct.c_int16), ct.c_ulong])
        self._set_signature("GetImages", [ct.c_long, ct.c_long, ct.POINTER(ct.c_int32), ct.c_ulong, c_long_p, c_long_p])
        self._set_signature("GetImages16", [ct.c_long, ct.c_long, ct.POINTER(ct.c_int16), ct.c_ulong, c_long_p, c_long_p])

        # Gain, readout, baseline
        self._set_signature("SetEMGainMode", [ct.c_int])
        self._set_signature("GetEMGainRange", [c_int_p, c_int_p])
        self._set_signature("SetEMCCDGain", [ct.c_int])
        self._set_signature("GetEMCCDGain", [c_int_p])
        self._set_signature("SetEMAdvanced", [ct.c_int])
        self._set_signature("GetEMAdvanced", [c_int_p])
        self._set_signature("GetNumberPreAmpGains", [c_int_p])
        self._set_signature("GetPreAmpGain", [ct.c_int, c_float_p])
        self._set_signature("SetPreAmpGain", [ct.c_int])
        self._set_signature("GetNumberADChannels", [c_int_p])
        self._set_signature("GetNumberAmp", [c_int_p])
        self._set_signature("GetNumberHSSpeeds", [ct.c_int, ct.c_int, c_int_p])
        self._set_signature("GetHSSpeed", [ct.c_int, ct.c_int, ct.c_int, c_float_p])
        self._set_signature("SetHSSpeed", [ct.c_int, ct.c_int])
        self._set_signature("GetNumberVSSpeeds", [c_int_p])
        self._set_signature("GetVSSpeed", [ct.c_int, c_float_p])
        self._set_signature("SetVSSpeed", [ct.c_int])
        self._set_signature("GetFastestRecommendedVSSpeed", [c_int_p, c_float_p])
        self._set_signature("SetBaselineClamp", [ct.c_int])
        self._set_signature("GetBaselineClamp", [c_int_p])
        self._set_signature("SetBaselineOffset", [ct.c_int])
        self._set_signature("GetBitDepth", [ct.c_int, c_uint_p])

        # Shutter and trigger helpers
        self._set_signature("SetShutterEx", [ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int])
        self._set_signature("GetShutterMinTimes", [c_int_p, c_int_p])
        self._set_signature("IsInternalMechanicalShutter", [c_int_p])
        self._set_signature("SendSoftwareTrigger", [])
        self._set_signature("IsTriggerModeAvailable", [ct.c_int])
        self._set_signature("SetFrameTransferMode", [ct.c_int])
        # Photon-counting and count-convert post-processing. Available in SDK2 examples/headers.
        if hasattr(self.lib, "PostProcessPhotonCounting"):
            self._set_signature(
                "PostProcessPhotonCounting",
                [
                    ct.POINTER(ct.c_int32),
                    ct.POINTER(ct.c_int32),
                    ct.c_int,
                    ct.c_int,
                    ct.c_int,
                    ct.c_int,
                    ct.POINTER(ct.c_float),
                    ct.c_int,
                    ct.c_int,
                ],
            )
        if hasattr(self.lib, "PostProcessCountConvert"):
            self._set_signature(
                "PostProcessCountConvert",
                [
                    ct.POINTER(ct.c_int32),
                    ct.POINTER(ct.c_int32),
                    ct.c_int,
                    ct.c_int,
                    ct.c_int,
                    ct.c_int,
                    ct.c_int,
                    ct.c_float,
                    ct.c_float,
                    ct.c_int,
                    ct.c_int,
                ],
            )
        if hasattr(self.lib, "SetCountConvertMode"):
            self._set_signature("SetCountConvertMode", [ct.c_int])
        if hasattr(self.lib, "SetCountConvertWavelength"):
            self._set_signature("SetCountConvertWavelength", [ct.c_float])
        if hasattr(self.lib, "GetCountConvertWavelengthRange"):
            self._set_signature("GetCountConvertWavelengthRange", [c_float_p, c_float_p])
        if hasattr(self.lib, "IsCountConvertModeAvailable"):
            self._set_signature("IsCountConvertModeAvailable", [ct.c_int])
        if hasattr(self.lib, "GetQE"):
            self._set_signature("GetQE", [ct.c_char_p, ct.c_float, ct.c_uint, c_float_p])
        if hasattr(self.lib, "GetSensitivity"):
            self._set_signature("GetSensitivity", [ct.c_int, ct.c_int, ct.c_int, ct.c_int, c_float_p])

    # Generic call helpers -------------------------------------------------
    def _check(self, code: int, function: str, *, allowed: set[int] | None = None) -> int:
        return check_success(code, function, allowed=allowed)

    # Lifecycle ------------------------------------------------------------
    def initialize(self) -> None:
        init_dir = self.initialization_directory.encode("utf-8")
        self._check(self.lib.Initialize(init_dir), "Initialize")
        self.initialized = True

    def shutdown(self) -> None:
        if self.initialized:
            self.lib.ShutDown()
            self.initialized = False

    def __enter__(self) -> "AndorSDK2":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # System info ----------------------------------------------------------
    def get_available_cameras(self) -> int:
        n = ct.c_long()
        self._check(self.lib.GetAvailableCameras(ct.byref(n)), "GetAvailableCameras")
        return int(n.value)

    def get_camera_handle(self, index: int = 0) -> int:
        handle = ct.c_long()
        self._check(self.lib.GetCameraHandle(ct.c_long(index), ct.byref(handle)), "GetCameraHandle")
        return int(handle.value)

    def set_current_camera(self, handle: int) -> None:
        self._check(self.lib.SetCurrentCamera(ct.c_long(handle)), "SetCurrentCamera")

    def get_current_camera(self) -> int:
        handle = ct.c_long()
        self._check(self.lib.GetCurrentCamera(ct.byref(handle)), "GetCurrentCamera")
        return int(handle.value)

    def get_head_model(self) -> str:
        buf = ct.create_string_buffer(128)
        self._check(self.lib.GetHeadModel(buf), "GetHeadModel")
        return buf.value.decode(errors="replace")

    def get_camera_serial_number(self) -> int:
        serial = ct.c_uint()
        self._check(self.lib.GetCameraSerialNumber(ct.byref(serial)), "GetCameraSerialNumber")
        return int(serial.value)

    def get_detector(self) -> tuple[int, int]:
        xp = ct.c_int()
        yp = ct.c_int()
        self._check(self.lib.GetDetector(ct.byref(xp), ct.byref(yp)), "GetDetector")
        return int(xp.value), int(yp.value)

    def get_pixel_size_um(self) -> tuple[float, float]:
        xp = ct.c_float()
        yp = ct.c_float()
        self._check(self.lib.GetPixelSize(ct.byref(xp), ct.byref(yp)), "GetPixelSize")
        return float(xp.value), float(yp.value)

    def get_hardware_version(self) -> dict[str, int]:
        pcb, decode, dummy1, dummy2, firmware, build = [ct.c_uint() for _ in range(6)]
        self._check(
            self.lib.GetHardwareVersion(
                ct.byref(pcb), ct.byref(decode), ct.byref(dummy1), ct.byref(dummy2), ct.byref(firmware), ct.byref(build)
            ),
            "GetHardwareVersion",
        )
        return {"pcb": pcb.value, "decode": decode.value, "firmware": firmware.value, "firmware_build": build.value}

    def get_software_version(self) -> dict[str, int]:
        eprom, coffile, vxdrev, vxdver, dllrev, dllver = [ct.c_uint() for _ in range(6)]
        self._check(
            self.lib.GetSoftwareVersion(
                ct.byref(eprom), ct.byref(coffile), ct.byref(vxdrev), ct.byref(vxdver), ct.byref(dllrev), ct.byref(dllver)
            ),
            "GetSoftwareVersion",
        )
        return {
            "eprom": eprom.value,
            "coffile": coffile.value,
            "driver_rev": vxdrev.value,
            "driver_ver": vxdver.value,
            "dll_rev": dllrev.value,
            "dll_ver": dllver.value,
        }

    # Temperature ----------------------------------------------------------
    def get_temperature_range_c(self) -> tuple[int, int]:
        low = ct.c_int()
        high = ct.c_int()
        self._check(self.lib.GetTemperatureRange(ct.byref(low), ct.byref(high)), "GetTemperatureRange")
        return int(low.value), int(high.value)

    def set_temperature_c(self, setpoint_c: int) -> None:
        self._check(self.lib.SetTemperature(ct.c_int(int(setpoint_c))), "SetTemperature")

    def get_temperature_c(self) -> tuple[float, str, int]:
        temp = ct.c_float()
        code = self.lib.GetTemperatureF(ct.byref(temp))
        self._check(code, "GetTemperatureF", allowed=TEMP_RETURN_CODES)
        return float(temp.value), TEMPERATURE_STATUS.get(int(code), code_name(int(code))), int(code)

    def cooler_on(self) -> None:
        self._check(self.lib.CoolerON(), "CoolerON")

    def cooler_off(self) -> None:
        self._check(self.lib.CoolerOFF(), "CoolerOFF")

    def is_cooler_on(self) -> bool:
        state = ct.c_int()
        self._check(self.lib.IsCoolerOn(ct.byref(state)), "IsCoolerOn")
        return bool(state.value)

    def set_cooler_mode(self, keep_cooler_on_at_shutdown: bool) -> None:
        self._check(self.lib.SetCoolerMode(ct.c_int(int(bool(keep_cooler_on_at_shutdown)))), "SetCoolerMode")

    def set_fan_mode(self, mode: int) -> None:
        self._check(self.lib.SetFanMode(ct.c_int(int(mode))), "SetFanMode")

    # Acquisition configuration ------------------------------------------
    def set_acquisition_mode(self, mode: int) -> None:
        self._check(self.lib.SetAcquisitionMode(ct.c_int(int(mode))), "SetAcquisitionMode")

    def set_read_mode(self, mode: int) -> None:
        self._check(self.lib.SetReadMode(ct.c_int(int(mode))), "SetReadMode")

    def set_trigger_mode(self, mode: int) -> None:
        self._check(self.lib.SetTriggerMode(ct.c_int(int(mode))), "SetTriggerMode")

    def is_trigger_mode_available(self, mode: int) -> bool:
        code = self.lib.IsTriggerModeAvailable(ct.c_int(int(mode)))
        return code == 20002

    def set_exposure_time_s(self, seconds: float) -> None:
        self._check(self.lib.SetExposureTime(ct.c_float(float(seconds))), "SetExposureTime")

    def set_accumulation_cycle_time_s(self, seconds: float) -> None:
        self._check(self.lib.SetAccumulationCycleTime(ct.c_float(float(seconds))), "SetAccumulationCycleTime")

    def set_kinetic_cycle_time_s(self, seconds: float) -> None:
        self._check(self.lib.SetKineticCycleTime(ct.c_float(float(seconds))), "SetKineticCycleTime")

    def set_number_kinetics(self, count: int) -> None:
        self._check(self.lib.SetNumberKinetics(ct.c_int(int(count))), "SetNumberKinetics")

    def set_number_accumulations(self, count: int) -> None:
        self._check(self.lib.SetNumberAccumulations(ct.c_int(int(count))), "SetNumberAccumulations")

    def get_acquisition_timings_s(self) -> tuple[float, float, float]:
        exposure = ct.c_float()
        accumulation = ct.c_float()
        kinetic = ct.c_float()
        self._check(self.lib.GetAcquisitionTimings(ct.byref(exposure), ct.byref(accumulation), ct.byref(kinetic)), "GetAcquisitionTimings")
        return float(exposure.value), float(accumulation.value), float(kinetic.value)

    def get_readout_time_s(self) -> float:
        value = ct.c_float()
        self._check(self.lib.GetReadOutTime(ct.byref(value)), "GetReadOutTime")
        return float(value.value)

    def set_image(self, *, hbin: int = 1, vbin: int = 1, hstart: int = 1, hend: int, vstart: int = 1, vend: int) -> None:
        self._check(
            self.lib.SetImage(
                ct.c_int(hbin), ct.c_int(vbin), ct.c_int(hstart), ct.c_int(hend), ct.c_int(vstart), ct.c_int(vend)
            ),
            "SetImage",
        )

    def prepare_acquisition(self) -> None:
        self._check(self.lib.PrepareAcquisition(), "PrepareAcquisition")

    def start_acquisition(self) -> None:
        self._check(self.lib.StartAcquisition(), "StartAcquisition")

    def abort_acquisition(self) -> None:
        code = self.lib.AbortAcquisition()
        # AbortAcquisition can be called defensively.  Only raise for real errors.
        self._check(code, "AbortAcquisition")

    def wait_for_acquisition(self, timeout_ms: int | None = None) -> None:
        if timeout_ms is None or not hasattr(self.lib, "WaitForAcquisitionTimeOut"):
            self._check(self.lib.WaitForAcquisition(), "WaitForAcquisition")
        else:
            self._check(self.lib.WaitForAcquisitionTimeOut(ct.c_int(int(timeout_ms))), "WaitForAcquisitionTimeOut")

    def cancel_wait(self) -> None:
        self._check(self.lib.CancelWait(), "CancelWait")

    def get_status(self) -> tuple[str, int]:
        status = ct.c_int()
        code = self.lib.GetStatus(ct.byref(status))
        self._check(code, "GetStatus")
        return STATUS.get(int(status.value), code_name(int(status.value))), int(status.value)

    def is_acquiring(self) -> bool:
        return self.get_status()[1] == DRV_ACQUIRING

    def get_number_new_images(self) -> tuple[int, int]:
        first = ct.c_long()
        last = ct.c_long()
        self._check(self.lib.GetNumberNewImages(ct.byref(first), ct.byref(last)), "GetNumberNewImages")
        return int(first.value), int(last.value)

    def get_number_available_images(self) -> tuple[int, int]:
        first = ct.c_long()
        last = ct.c_long()
        self._check(self.lib.GetNumberAvailableImages(ct.byref(first), ct.byref(last)), "GetNumberAvailableImages")
        return int(first.value), int(last.value)

    def get_total_number_images_acquired(self) -> int:
        n = ct.c_long()
        self._check(self.lib.GetTotalNumberImagesAcquired(ct.byref(n)), "GetTotalNumberImagesAcquired")
        return int(n.value)

    def get_circular_buffer_size(self) -> int:
        n = ct.c_long()
        self._check(self.lib.GetSizeOfCircularBuffer(ct.byref(n)), "GetSizeOfCircularBuffer")
        return int(n.value)

    def free_internal_memory(self) -> None:
        self._check(self.lib.FreeInternalMemory(), "FreeInternalMemory")

    # Data retrieval -------------------------------------------------------
    @staticmethod
    def _empty_frame(shape: tuple[int, int], dtype: np.dtype) -> np.ndarray:
        return np.ascontiguousarray(np.zeros(int(shape[0]) * int(shape[1]), dtype=dtype))

    def get_most_recent_image16(self, shape: tuple[int, int]) -> np.ndarray:
        arr = self._empty_frame(shape, np.dtype(np.int16))
        self._check(self.lib.GetMostRecentImage16(arr.ctypes.data_as(ct.POINTER(ct.c_int16)), ct.c_ulong(arr.size)), "GetMostRecentImage16")
        return arr.reshape((shape[1], shape[0]))

    def get_acquired_data16(self, shape: tuple[int, int]) -> np.ndarray:
        arr = self._empty_frame(shape, np.dtype(np.int16))
        self._check(self.lib.GetAcquiredData16(arr.ctypes.data_as(ct.POINTER(ct.c_int16)), ct.c_ulong(arr.size)), "GetAcquiredData16")
        return arr.reshape((shape[1], shape[0]))

    def get_oldest_image16(self, shape: tuple[int, int]) -> np.ndarray:
        arr = self._empty_frame(shape, np.dtype(np.int16))
        self._check(self.lib.GetOldestImage16(arr.ctypes.data_as(ct.POINTER(ct.c_int16)), ct.c_ulong(arr.size)), "GetOldestImage16")
        return arr.reshape((shape[1], shape[0]))

    def get_images16(self, first: int, last: int, shape: tuple[int, int]) -> tuple[np.ndarray, int, int]:
        n_frames = int(last) - int(first) + 1
        if n_frames <= 0:
            raise ValueError("last image index must be >= first image index")
        arr = np.ascontiguousarray(np.zeros(int(shape[0]) * int(shape[1]) * n_frames, dtype=np.int16))
        valid_first = ct.c_long()
        valid_last = ct.c_long()
        self._check(
            self.lib.GetImages16(
                ct.c_long(first),
                ct.c_long(last),
                arr.ctypes.data_as(ct.POINTER(ct.c_int16)),
                ct.c_ulong(arr.size),
                ct.byref(valid_first),
                ct.byref(valid_last),
            ),
            "GetImages16",
        )
        return arr.reshape((n_frames, shape[1], shape[0])), int(valid_first.value), int(valid_last.value)

    def postprocess_photon_counting(self, frame: np.ndarray, thresholds_adu: list[float] | tuple[float, ...]) -> np.ndarray:
        """Run Andor SDK2 PostProcessPhotonCounting on an int32 2D frame."""
        if not hasattr(self.lib, "PostProcessPhotonCounting"):
            raise RuntimeError("This SDK2 DLL does not export PostProcessPhotonCounting")
        arr = np.ascontiguousarray(frame, dtype=np.int32)
        if arr.ndim != 2:
            raise ValueError("PostProcessPhotonCounting expects one 2D frame")
        thresholds = np.ascontiguousarray(np.asarray(thresholds_adu, dtype=np.float32))
        if thresholds.size < 1:
            raise ValueError("At least one photon-counting threshold is required")
        out = np.ascontiguousarray(np.zeros(arr.size, dtype=np.int32))
        height, width = int(arr.shape[0]), int(arr.shape[1])
        self._check(
            self.lib.PostProcessPhotonCounting(
                arr.ravel().ctypes.data_as(ct.POINTER(ct.c_int32)),
                out.ctypes.data_as(ct.POINTER(ct.c_int32)),
                ct.c_int(out.size),
                ct.c_int(1),
                ct.c_int(1),
                ct.c_int(thresholds.size),
                thresholds.ctypes.data_as(ct.POINTER(ct.c_float)),
                ct.c_int(height),
                ct.c_int(width),
            ),
            "PostProcessPhotonCounting",
        )
        return out.reshape(arr.shape)

    def count_convert_mode_available(self, mode: int) -> bool:
        """Return whether SDK count-convert mode is available for current settings."""
        if not hasattr(self.lib, "IsCountConvertModeAvailable"):
            return False
        code = int(self.lib.IsCountConvertModeAvailable(ct.c_int(int(mode))))
        return code == 20002

    def get_count_convert_wavelength_range_nm(self) -> tuple[float, float]:
        if not hasattr(self.lib, "GetCountConvertWavelengthRange"):
            raise RuntimeError("This SDK2 DLL does not export GetCountConvertWavelengthRange")
        low = ct.c_float()
        high = ct.c_float()
        self._check(self.lib.GetCountConvertWavelengthRange(ct.byref(low), ct.byref(high)), "GetCountConvertWavelengthRange")
        return float(low.value), float(high.value)

    def set_count_convert(self, mode: int, wavelength_nm: float) -> None:
        if not hasattr(self.lib, "SetCountConvertMode") or not hasattr(self.lib, "SetCountConvertWavelength"):
            raise RuntimeError("This SDK2 DLL does not export SetCountConvertMode/SetCountConvertWavelength")
        self._check(self.lib.SetCountConvertWavelength(ct.c_float(float(wavelength_nm))), "SetCountConvertWavelength")
        self._check(self.lib.SetCountConvertMode(ct.c_int(int(mode))), "SetCountConvertMode")

    def get_qe_percent(self, wavelength_nm: float) -> float:
        if not hasattr(self.lib, "GetQE"):
            raise RuntimeError("This SDK2 DLL does not export GetQE")
        sensor = self.get_head_model().encode("ascii", errors="ignore")
        value = ct.c_float()
        self._check(self.lib.GetQE(ct.c_char_p(sensor), ct.c_float(float(wavelength_nm)), ct.c_uint(0), ct.byref(value)), "GetQE")
        return float(value.value)

    def get_sensitivity_adu_per_electron(self, ad_channel: int = 0, hs_speed_index: int = 0, output_amplifier: int = 0, preamp_index: int = 0) -> float:
        if not hasattr(self.lib, "GetSensitivity"):
            raise RuntimeError("This SDK2 DLL does not export GetSensitivity")
        value = ct.c_float()
        self._check(
            self.lib.GetSensitivity(
                ct.c_int(int(ad_channel)),
                ct.c_int(int(hs_speed_index)),
                ct.c_int(int(output_amplifier)),
                ct.c_int(int(preamp_index)),
                ct.byref(value),
            ),
            "GetSensitivity",
        )
        return float(value.value)

    def count_convert_frame(
        self,
        frame: np.ndarray,
        *,
        mode: int,
        wavelength_nm: float,
        em_gain: int | None = None,
        sensitivity: float | None = None,
        qe_percent: float | None = None,
        baseline: int = 0,
        ad_channel: int = 0,
        hs_speed_index: int = 0,
        output_amplifier: int = 0,
        preamp_index: int = 0,
    ) -> np.ndarray:
        """Convert a raw ADU frame to SDK-estimated electrons (mode=1) or photons (mode=2).

        Uses PostProcessCountConvert when exported by the DLL. This remains an
        SDK estimate and should be recorded as such in metadata.
        """
        if int(mode) not in {1, 2}:
            raise ValueError("count-convert mode must be 1 (electrons) or 2 (photons)")
        arr = np.ascontiguousarray(frame, dtype=np.int32)
        if arr.ndim != 2:
            raise ValueError("count_convert_frame expects one 2D frame")

        if not hasattr(self.lib, "PostProcessCountConvert"):
            raise RuntimeError("This SDK2 DLL does not export PostProcessCountConvert")

        em = int(self.get_emccd_gain() if em_gain is None else em_gain)
        sens = float(self.get_sensitivity_adu_per_electron(ad_channel, hs_speed_index, output_amplifier, preamp_index) if sensitivity is None else sensitivity)
        qe = float(self.get_qe_percent(wavelength_nm) if qe_percent is None else qe_percent)
        out = np.ascontiguousarray(np.zeros(arr.size, dtype=np.int32))
        height, width = int(arr.shape[0]), int(arr.shape[1])
        self._check(
            self.lib.PostProcessCountConvert(
                arr.ravel().ctypes.data_as(ct.POINTER(ct.c_int32)),
                out.ctypes.data_as(ct.POINTER(ct.c_int32)),
                ct.c_int(out.size),
                ct.c_int(1),
                ct.c_int(int(baseline)),
                ct.c_int(int(mode)),
                ct.c_int(em),
                ct.c_float(qe),
                ct.c_float(sens),
                ct.c_int(height),
                ct.c_int(width),
            ),
            "PostProcessCountConvert",
        )
        return out.reshape(arr.shape).astype(np.float32, copy=False)

    # Gain/readout ---------------------------------------------------------
    def set_em_gain_mode(self, mode: int) -> None:
        self._check(self.lib.SetEMGainMode(ct.c_int(int(mode))), "SetEMGainMode")

    def get_em_gain_range(self) -> tuple[int, int]:
        low = ct.c_int()
        high = ct.c_int()
        self._check(self.lib.GetEMGainRange(ct.byref(low), ct.byref(high)), "GetEMGainRange")
        return int(low.value), int(high.value)

    def set_emccd_gain(self, gain: int) -> None:
        self._check(self.lib.SetEMCCDGain(ct.c_int(int(gain))), "SetEMCCDGain")

    def get_emccd_gain(self) -> int:
        value = ct.c_int()
        self._check(self.lib.GetEMCCDGain(ct.byref(value)), "GetEMCCDGain")
        return int(value.value)

    def set_em_advanced(self, enabled: bool) -> None:
        self._check(self.lib.SetEMAdvanced(ct.c_int(int(bool(enabled)))), "SetEMAdvanced")

    def get_em_advanced(self) -> bool:
        value = ct.c_int()
        self._check(self.lib.GetEMAdvanced(ct.byref(value)), "GetEMAdvanced")
        return bool(value.value)

    def get_number_preamp_gains(self) -> int:
        n = ct.c_int()
        self._check(self.lib.GetNumberPreAmpGains(ct.byref(n)), "GetNumberPreAmpGains")
        return int(n.value)

    def get_preamp_gain(self, index: int) -> float:
        value = ct.c_float()
        self._check(self.lib.GetPreAmpGain(ct.c_int(int(index)), ct.byref(value)), "GetPreAmpGain")
        return float(value.value)

    def set_preamp_gain(self, index: int) -> None:
        self._check(self.lib.SetPreAmpGain(ct.c_int(int(index))), "SetPreAmpGain")

    def get_number_ad_channels(self) -> int:
        n = ct.c_int()
        self._check(self.lib.GetNumberADChannels(ct.byref(n)), "GetNumberADChannels")
        return int(n.value)

    def get_number_hs_speeds(self, ad_channel: int = 0, output_amplifier: int = 0) -> int:
        n = ct.c_int()
        self._check(self.lib.GetNumberHSSpeeds(ct.c_int(ad_channel), ct.c_int(output_amplifier), ct.byref(n)), "GetNumberHSSpeeds")
        return int(n.value)

    def get_hs_speed_mhz(self, index: int, ad_channel: int = 0, output_amplifier: int = 0) -> float:
        value = ct.c_float()
        self._check(
            self.lib.GetHSSpeed(ct.c_int(ad_channel), ct.c_int(output_amplifier), ct.c_int(index), ct.byref(value)),
            "GetHSSpeed",
        )
        return float(value.value)

    def set_hs_speed(self, index: int, output_amplifier: int = 0) -> None:
        self._check(self.lib.SetHSSpeed(ct.c_int(output_amplifier), ct.c_int(index)), "SetHSSpeed")

    def get_number_vs_speeds(self) -> int:
        n = ct.c_int()
        self._check(self.lib.GetNumberVSSpeeds(ct.byref(n)), "GetNumberVSSpeeds")
        return int(n.value)

    def get_vs_speed_us(self, index: int) -> float:
        value = ct.c_float()
        self._check(self.lib.GetVSSpeed(ct.c_int(int(index)), ct.byref(value)), "GetVSSpeed")
        return float(value.value)

    def set_vs_speed(self, index: int) -> None:
        self._check(self.lib.SetVSSpeed(ct.c_int(int(index))), "SetVSSpeed")

    def set_baseline_clamp(self, enabled: bool) -> None:
        self._check(self.lib.SetBaselineClamp(ct.c_int(int(bool(enabled)))), "SetBaselineClamp")

    def set_baseline_offset(self, value: int) -> None:
        self._check(self.lib.SetBaselineOffset(ct.c_int(int(value))), "SetBaselineOffset")

    def get_bit_depth(self, ad_channel: int = 0) -> int:
        depth = ct.c_uint()
        self._check(self.lib.GetBitDepth(ct.c_int(ad_channel), ct.byref(depth)), "GetBitDepth")
        return int(depth.value)

    # Shutter --------------------------------------------------------------
    def set_shutter_ex(self, typ: int, internal_mode: int, closing_ms: int, opening_ms: int, external_mode: int) -> None:
        self._check(
            self.lib.SetShutterEx(ct.c_int(typ), ct.c_int(internal_mode), ct.c_int(closing_ms), ct.c_int(opening_ms), ct.c_int(external_mode)),
            "SetShutterEx",
        )

    def get_shutter_min_times_ms(self) -> tuple[int, int]:
        closing = ct.c_int()
        opening = ct.c_int()
        self._check(self.lib.GetShutterMinTimes(ct.byref(closing), ct.byref(opening)), "GetShutterMinTimes")
        return int(opening.value), int(closing.value)

    def has_internal_mechanical_shutter(self) -> bool:
        state = ct.c_int()
        self._check(self.lib.IsInternalMechanicalShutter(ct.byref(state)), "IsInternalMechanicalShutter")
        return bool(state.value)

    def send_software_trigger(self) -> None:
        self._check(self.lib.SendSoftwareTrigger(), "SendSoftwareTrigger")

    def set_frame_transfer_mode(self, enabled: bool) -> None:
        self._check(self.lib.SetFrameTransferMode(ct.c_int(int(bool(enabled)))), "SetFrameTransferMode")
