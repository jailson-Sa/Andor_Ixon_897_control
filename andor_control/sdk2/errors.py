"""Andor SDK2 return codes and exceptions.

This module intentionally contains no Andor SDK calls.  It is safe to import
on any operating system and is used by the real SDK wrapper and by tests.
"""
from __future__ import annotations

from dataclasses import dataclass


DRV_SUCCESS = 20002
DRV_IDLE = 20073
DRV_ACQUIRING = 20072
DRV_TEMP_OFF = 20034
DRV_TEMP_NOT_STABILIZED = 20035
DRV_TEMP_STABILIZED = 20036
DRV_TEMP_NOT_REACHED = 20037
DRV_TEMP_OUT_RANGE = 20038
DRV_TEMP_NOT_SUPPORTED = 20039
DRV_TEMP_DRIFT = 20040

ERRORS: dict[int, str] = {
    20002: "DRV_SUCCESS",
    20003: "DRV_VXDNOTINSTALLED",
    20004: "DRV_ERROR_SCAN",
    20005: "DRV_ERROR_CHECK_SUM",
    20006: "DRV_ERROR_FILELOAD",
    20007: "DRV_UNKNOWN_FUNCTION",
    20008: "DRV_ERROR_VXD_INIT",
    20009: "DRV_ERROR_ADDRESS",
    20010: "DRV_ERROR_PAGELOCK",
    20011: "DRV_ERROR_PAGE_UNLOCK",
    20012: "DRV_ERROR_BOARDTEST",
    20013: "DRV_ERROR_ACK: unable to communicate with card/camera",
    20014: "DRV_ERROR_UP_FIFO",
    20015: "DRV_ERROR_PATTERN",
    20017: "DRV_ACQUISITION_ERRORS",
    20018: "DRV_ACQ_BUFFER: computer unable to read the data at required rate",
    20019: "DRV_ACQ_DOWNFIFO_FULL",
    20020: "DRV_PROC_UNKNOWN_INSTRUCTION",
    20021: "DRV_ILLEGAL_OP_CODE",
    20022: "DRV_KINETIC_TIME_NOT_MET",
    20023: "DRV_ACCUM_TIME_NOT_MET",
    20024: "DRV_NO_NEW_DATA",
    20026: "DRV_SPOOLERROR: overflow of spool buffer",
    20027: "DRV_SPOOLSETUPERROR",
    20033: "DRV_TEMPERATURE_CODES",
    20034: "DRV_TEMP_OFF: temperature is OFF",
    20035: "DRV_TEMP_NOT_STABILIZED: reached but not stabilized",
    20036: "DRV_TEMP_STABILIZED: stabilized at set point",
    20037: "DRV_TEMP_NOT_REACHED: has not reached set point",
    20038: "DRV_TEMP_OUT_RANGE",
    20039: "DRV_TEMP_NOT_SUPPORTED",
    20040: "DRV_TEMP_DRIFT: stabilized but drifted",
    20049: "DRV_GENERAL_ERRORS",
    20050: "DRV_INVALID_AUX",
    20051: "DRV_COF_NOTLOADED",
    20052: "DRV_FPGAPROG",
    20053: "DRV_FLEXERROR",
    20054: "DRV_GPIBERROR",
    20064: "DRV_DATATYPE",
    20065: "DRV_DRIVER_ERRORS",
    20066: "DRV_P1INVALID",
    20067: "DRV_P2INVALID",
    20068: "DRV_P3INVALID",
    20069: "DRV_P4INVALID",
    20070: "DRV_INIERROR",
    20071: "DRV_COFERROR",
    20072: "DRV_ACQUIRING",
    20073: "DRV_IDLE",
    20074: "DRV_TEMPCYCLE",
    20075: "DRV_NOT_INITIALIZED",
    20076: "DRV_P5INVALID",
    20077: "DRV_P6INVALID",
    20078: "DRV_INVALID_MODE",
    20079: "DRV_INVALID_FILTER",
    20080: "DRV_I2CERRORS",
    20081: "DRV_I2CDEVNOTFOUND",
    20082: "DRV_I2CTIMEOUT",
    20083: "DRV_P7INVALID",
    20089: "DRV_USBERROR",
    20090: "DRV_IOCERROR",
    20091: "DRV_VRMVERSIONERROR",
    20093: "DRV_USB_INTERRUPT_ENDPOINT_ERROR",
    20094: "DRV_RANDOM_TRACK_ERROR",
    20095: "DRV_INVALID_TRIGGER_MODE",
    20096: "DRV_LOAD_FIRMWARE_ERROR",
    20097: "DRV_DIVIDE_BY_ZERO_ERROR",
    20098: "DRV_INVALID_RINGEXPOSURES",
    20099: "DRV_BINNING_ERROR",
    20100: "DRV_INVALID_AMPLIFIER",
    20101: "DRV_INVALID_COUNTCONVERT_MODE",
    20115: "DRV_ERROR_MAP",
    20116: "DRV_ERROR_UNMAP",
    20117: "DRV_ERROR_MDL",
    20118: "DRV_ERROR_UNMDL",
    20119: "DRV_ERROR_BUFFSIZE",
    20121: "DRV_ERROR_NOHANDLE",
    20130: "DRV_GATING_NOT_AVAILABLE",
    20131: "DRV_FPGA_VOLTAGE_ERROR",
    20990: "DRV_NOCAMERA: no camera present",
    20991: "DRV_NOT_SUPPORTED: feature not supported on this camera",
    20992: "DRV_NOT_AVAILABLE: feature not available at the moment",
}

TEMPERATURE_STATUS: dict[int, str] = {
    DRV_TEMP_OFF: "off",
    DRV_TEMP_NOT_STABILIZED: "not_stabilized",
    DRV_TEMP_STABILIZED: "stabilized",
    DRV_TEMP_NOT_REACHED: "not_reached",
    DRV_TEMP_OUT_RANGE: "out_of_range",
    DRV_TEMP_NOT_SUPPORTED: "not_supported",
    DRV_TEMP_DRIFT: "drifted",
}

STATUS: dict[int, str] = {
    DRV_IDLE: "idle",
    DRV_ACQUIRING: "acquiring",
    20022: "kinetic_time_not_met",
    20023: "accum_time_not_met",
    20013: "communication_error",
    20018: "acquisition_buffer_error",
    20026: "spool_overflow",
    20074: "temperature_cycle",
}


@dataclass(slots=True)
class SDK2Error(RuntimeError):
    """Raised when an SDK2 function returns a non-success error code."""

    function: str
    code: int

    def __str__(self) -> str:
        return f"{self.function} failed with {self.code}: {ERRORS.get(self.code, 'UNKNOWN_ERROR')}"


def code_name(code: int) -> str:
    """Return a readable SDK2 code name."""
    return ERRORS.get(int(code), f"UNKNOWN_ERROR_{int(code)}")


def check_success(code: int, function: str, *, allowed: set[int] | None = None) -> int:
    """Validate an SDK2 return code.

    Parameters
    ----------
    code:
        Integer return code from the SDK function.
    function:
        Human-readable function name used in exceptions.
    allowed:
        Extra accepted non-success return codes.  This is necessary for SDK2
        functions such as GetTemperatureF, which return temperature status as
        their function result.
    """
    code = int(code)
    if code == DRV_SUCCESS or (allowed is not None and code in allowed):
        return code
    raise SDK2Error(function=function, code=code)
