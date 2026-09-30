"""Typed configuration objects for Andor iXon acquisition."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from andor_control.sdk2.enums import AcquisitionMode, EMGainMode, FanMode, ReadMode, ShutterMode, TriggerMode


@dataclass(slots=True)
class ROI:
    """Andor SDK2 image ROI.

    The SDK uses 1-based inclusive coordinates.  This class keeps that convention
    explicit to avoid off-by-one errors when calling SetImage.
    """

    hstart: int = 1
    hend: int = 512
    vstart: int = 1
    vend: int = 512
    hbin: int = 1
    vbin: int = 1

    def validate(self, detector_shape: tuple[int, int]) -> None:
        width, height = detector_shape
        if self.hbin < 1 or self.vbin < 1:
            raise ValueError("ROI binning must be >= 1")
        if not (1 <= self.hstart <= self.hend <= width):
            raise ValueError(f"Invalid horizontal ROI limits {self.hstart}:{self.hend} for detector width {width}")
        if not (1 <= self.vstart <= self.vend <= height):
            raise ValueError(f"Invalid vertical ROI limits {self.vstart}:{self.vend} for detector height {height}")
        if (self.hend - self.hstart + 1) % self.hbin != 0:
            raise ValueError("Horizontal ROI width must be divisible by hbin")
        if (self.vend - self.vstart + 1) % self.vbin != 0:
            raise ValueError("Vertical ROI height must be divisible by vbin")

    @property
    def output_shape(self) -> tuple[int, int]:
        """Return shape as (width, height) after binning."""
        return ((self.hend - self.hstart + 1) // self.hbin, (self.vend - self.vstart + 1) // self.vbin)

    def to_metadata(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class AcquisitionSettings:
    acquisition_mode: AcquisitionMode = AcquisitionMode.SINGLE_SCAN
    read_mode: ReadMode = ReadMode.IMAGE
    trigger_mode: TriggerMode = TriggerMode.INTERNAL
    exposure_s: float = 0.03
    accumulation_cycle_s: float | None = None
    kinetic_cycle_s: float | None = None
    number_kinetics: int | None = None
    number_accumulations: int | None = None
    frame_transfer: bool = False
    roi: ROI | None = None

    def validate(self) -> None:
        if self.exposure_s <= 0:
            raise ValueError("Exposure time must be > 0")
        if self.number_kinetics is not None and self.number_kinetics < 1:
            raise ValueError("number_kinetics must be >= 1")
        if self.number_accumulations is not None and self.number_accumulations < 1:
            raise ValueError("number_accumulations must be >= 1")
        if self.kinetic_cycle_s is not None and self.kinetic_cycle_s < 0:
            raise ValueError("kinetic_cycle_s must be >= 0")
        if self.accumulation_cycle_s is not None and self.accumulation_cycle_s < 0:
            raise ValueError("accumulation_cycle_s must be >= 0")

    def to_metadata(self) -> dict[str, object]:
        data = asdict(self)
        data["acquisition_mode"] = self.acquisition_mode.name
        data["read_mode"] = self.read_mode.name
        data["trigger_mode"] = self.trigger_mode.name
        if self.roi is not None:
            data["roi"] = self.roi.to_metadata()
        return data


@dataclass(slots=True)
class EMCCDSettings:
    output_amplifier: int = 0  # 0: EM amplifier on most iXon systems
    ad_channel: int = 0
    hs_speed_index: int = 0
    vs_speed_index: int = 0
    preamp_gain_index: int = 0
    em_gain_mode: EMGainMode = EMGainMode.REAL_GAIN
    em_gain: int | None = None  # None means: use the minimum valid SDK gain for the selected EM gain mode
    em_advanced: bool = False
    baseline_clamp: bool = True
    baseline_offset: int | None = None

    def validate(self) -> None:
        if self.output_amplifier < 0:
            raise ValueError("output_amplifier must be >= 0")
        if self.ad_channel < 0:
            raise ValueError("ad_channel must be >= 0")
        if self.hs_speed_index < 0 or self.vs_speed_index < 0:
            raise ValueError("speed indices must be >= 0")
        if self.preamp_gain_index < 0:
            raise ValueError("preamp_gain_index must be >= 0")
        if self.em_gain is not None and self.em_gain < 0:
            raise ValueError("em_gain must be >= 0")
        if self.baseline_offset is not None and self.baseline_offset % 100 != 0:
            raise ValueError("baseline_offset should be a multiple of 100")

    def to_metadata(self) -> dict[str, object]:
        data = asdict(self)
        data["em_gain_mode"] = self.em_gain_mode.name
        return data


@dataclass(slots=True)
class TemperatureSettings:
    setpoint_c: int = -70
    cooler_on: bool = True
    keep_cooler_on_at_shutdown: bool = False
    fan_mode: FanMode = FanMode.FULL
    require_stable_before_high_gain: bool = True

    def to_metadata(self) -> dict[str, object]:
        data = asdict(self)
        data["fan_mode"] = self.fan_mode.name
        return data


@dataclass(slots=True)
class ShutterSettings:
    ttl_high_opens: bool = False
    internal_mode: ShutterMode = ShutterMode.FULLY_AUTO
    external_mode: ShutterMode = ShutterMode.FULLY_AUTO
    closing_ms: int = 0
    opening_ms: int = 0

    def to_metadata(self) -> dict[str, object]:
        data = asdict(self)
        data["internal_mode"] = self.internal_mode.name
        data["external_mode"] = self.external_mode.name
        return data
