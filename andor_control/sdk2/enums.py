"""Small typed constants for the Andor SDK2/ATMCD interface."""
from __future__ import annotations

from enum import IntEnum


class AcquisitionMode(IntEnum):
    SINGLE_SCAN = 1
    ACCUMULATE = 2
    KINETICS = 3
    FAST_KINETICS = 4
    RUN_TILL_ABORT = 5


class ReadMode(IntEnum):
    FVB = 0
    MULTI_TRACK = 1
    RANDOM_TRACK = 2
    SINGLE_TRACK = 3
    IMAGE = 4


class TriggerMode(IntEnum):
    INTERNAL = 0
    EXTERNAL = 1
    EXTERNAL_START = 6
    EXTERNAL_EXPOSURE = 7
    EXTERNAL_FVB_EM = 9
    SOFTWARE = 10
    EXTERNAL_CHARGE_SHIFTING = 12


class EMGainMode(IntEnum):
    DAC_255 = 0
    DAC_4095 = 1
    LINEAR = 2
    REAL_GAIN = 3


class ShutterMode(IntEnum):
    FULLY_AUTO = 0
    PERMANENTLY_OPEN = 1
    PERMANENTLY_CLOSED = 2
    OPEN_FOR_FVB_SERIES = 4
    OPEN_FOR_ANY_SERIES = 5


class FanMode(IntEnum):
    FULL = 0
    LOW = 1
    OFF = 2
