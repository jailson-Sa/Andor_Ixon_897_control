"""Minimal continuous acquisition smoke test.

This is not a GUI.  It verifies that continuous acquisition, the worker thread,
and frame queue work before building the professional graphical interface.
"""
from __future__ import annotations

import argparse
import time

from andor_control.camera.acquisition_worker import ContinuousAcquisitionWorker
from andor_control.camera.ixon897 import AndorIXon897
from andor_control.camera.settings import AcquisitionSettings, ROI
from andor_control.sdk2.enums import AcquisitionMode, ReadMode, TriggerMode
from andor_control.sdk2.mock import MockAndorSDK2


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous acquisition CLI smoke test for Andor iXon 897.")
    parser.add_argument("--dll", default=None)
    parser.add_argument("--init-dir", default="")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--exposure", type=float, default=0.03)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("HSTART", "HEND", "VSTART", "VEND"))
    args = parser.parse_args()

    camera = AndorIXon897(MockAndorSDK2()) if args.mock else AndorIXon897.from_dll(args.dll, initialization_directory=args.init_dir)
    try:
        identity = camera.initialize()
        width, height = identity.detector_shape
        roi = ROI(args.roi[0], args.roi[1], args.roi[2], args.roi[3]) if args.roi else ROI(1, width, 1, height)
        camera.configure_acquisition(
            AcquisitionSettings(
                acquisition_mode=AcquisitionMode.RUN_TILL_ABORT,
                read_mode=ReadMode.IMAGE,
                trigger_mode=TriggerMode.INTERNAL,
                exposure_s=args.exposure,
                roi=roi,
            )
        )
        worker = ContinuousAcquisitionWorker(camera)
        worker.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            try:
                packet = worker.frames.get(timeout=0.5)
                print(
                    f"frame={packet.index:06d} shape={packet.frame.shape} "
                    f"min={int(packet.frame.min())} max={int(packet.frame.max())} mean={float(packet.frame.mean()):.2f}"
                )
            except Exception:
                pass
        worker.stop()
        print(f"acquired={worker.acquired_frames} dropped={worker.dropped_frames} last_error={worker.last_error}")
        return 0 if worker.last_error is None else 1
    finally:
        camera.shutdown(turn_cooler_off=False)


if __name__ == "__main__":
    raise SystemExit(main())
