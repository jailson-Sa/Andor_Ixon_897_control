"""Diagnostic script for Andor iXon 897 over SDK2.

Examples
--------
Real camera on Windows:
    python scripts/diagnose_ixon897.py --dll "C:\\path\\to\\atmcd64d.dll" --output out

Development test without hardware:
    python scripts/diagnose_ixon897.py --mock --output out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from andor_control.camera.ixon897 import AndorIXon897
from andor_control.camera.settings import AcquisitionSettings, EMCCDSettings, ROI, ShutterSettings, TemperatureSettings
from andor_control.sdk2.enums import AcquisitionMode, ReadMode, ShutterMode, TriggerMode
from andor_control.sdk2.mock import MockAndorSDK2
from andor_control.storage.simple_io import save_frame_npy, save_metadata_json


def build_camera(args: argparse.Namespace) -> AndorIXon897:
    if args.mock:
        return AndorIXon897(MockAndorSDK2(), camera_index=0)
    return AndorIXon897.from_dll(args.dll, initialization_directory=args.init_dir, camera_index=args.camera_index)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Andor iXon 897 SDK2 communication and capture one test frame.")
    parser.add_argument("--dll", default=None, help="Path to atmcd64d.dll. If omitted, uses ANDOR_SDK2_DLL or Windows DLL search path.")
    parser.add_argument("--init-dir", default="", help="SDK2 Initialize directory. Usually empty. Older systems may need a DETECTOR.INI directory.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--mock", action="store_true", help="Use synthetic mock backend instead of the real SDK/DLL.")
    parser.add_argument("--output", default="andor_diagnostic_output", help="Output directory.")
    parser.add_argument("--exposure", type=float, default=0.03, help="Exposure time in seconds.")
    parser.add_argument("--temperature", type=int, default=-70, help="Temperature setpoint in Celsius.")
    parser.add_argument("--no-cooler", action="store_true", help="Do not turn the cooler on during this diagnostic.")
    parser.add_argument(
        "--em-gain",
        type=int,
        default=None,
        help="EM gain for the test. If omitted, the diagnostic uses the SDK minimum valid gain for the selected mode.",
    )
    parser.add_argument("--allow-gain-before-stable", action="store_true", help="Allow EM gain > 1 before temperature is stabilized.")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("HSTART", "HEND", "VSTART", "VEND"), help="1-based inclusive SDK2 ROI.")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    camera = build_camera(args)
    try:
        identity = camera.initialize()
        print("Camera identity:")
        print(json.dumps(identity.to_dict(), indent=2))

        print("\nCapabilities:")
        print(json.dumps(camera.get_capability_summary(), indent=2))

        temp_settings = TemperatureSettings(setpoint_c=args.temperature, cooler_on=not args.no_cooler)
        temp_state = camera.configure_temperature(temp_settings)
        print("\nTemperature state after configuration:")
        print(json.dumps(temp_state.to_dict(), indent=2))

        # Safe default: internal shutter auto.  The user can change this later
        # after validating their lab shutter wiring.
        camera.configure_shutter(
            ShutterSettings(internal_mode=ShutterMode.FULLY_AUTO, external_mode=ShutterMode.FULLY_AUTO)
        )

        em_settings = camera.configure_emccd(
            EMCCDSettings(em_gain=args.em_gain),
            allow_high_gain_without_stable_temp=args.allow_gain_before_stable or args.mock,
            clamp_gain_to_sdk_range=True,
        )
        if args.em_gain is None:
            print(f"\nEM gain not specified. Using SDK minimum valid gain: {em_settings.em_gain}")
        elif em_settings.em_gain != args.em_gain:
            print(f"\nRequested EM gain {args.em_gain} is outside the current SDK range. Using {em_settings.em_gain} instead.")

        detector_width, detector_height = identity.detector_shape
        if args.roi:
            roi = ROI(hstart=args.roi[0], hend=args.roi[1], vstart=args.roi[2], vend=args.roi[3])
        else:
            roi = ROI(hstart=1, hend=detector_width, vstart=1, vend=detector_height)

        acq_info = camera.configure_acquisition(
            AcquisitionSettings(
                acquisition_mode=AcquisitionMode.SINGLE_SCAN,
                read_mode=ReadMode.IMAGE,
                trigger_mode=TriggerMode.INTERNAL,
                exposure_s=args.exposure,
                roi=roi,
            )
        )
        print("\nAcquisition setup:")
        print(json.dumps(acq_info, indent=2))

        frame = camera.acquire_single_frame()
        print("\nFrame statistics:")
        print(
            json.dumps(
                {
                    "shape": list(frame.shape),
                    "dtype": str(frame.dtype),
                    "min": int(frame.min()),
                    "max": int(frame.max()),
                    "mean": float(np.mean(frame)),
                    "std": float(np.std(frame)),
                },
                indent=2,
            )
        )

        frame_path = save_frame_npy(out / "diagnostic_frame.npy", frame)
        metadata_path = save_metadata_json(out / "diagnostic_metadata.json", camera.build_metadata())
        print(f"\nSaved frame: {frame_path}")
        print(f"Saved metadata: {metadata_path}")
        return 0
    finally:
        camera.shutdown(turn_cooler_off=False)


if __name__ == "__main__":
    raise SystemExit(main())
