"""Threaded acquisition worker independent from any GUI framework."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from andor_control.camera.ixon897 import AndorIXon897
from andor_control.camera.settings import AcquisitionSettings
from andor_control.sdk2.enums import AcquisitionMode


@dataclass(slots=True)
class FramePacket:
    frame: np.ndarray
    index: int
    timestamp_s: float


class ContinuousAcquisitionWorker:
    """Continuously acquire frames into a bounded queue.

    A GUI can read from ``frames`` without calling the SDK directly.  Dropping
    frames is intentional when the GUI cannot keep up; the latest data remains
    responsive while the acquisition thread records how many frames were dropped.
    """

    def __init__(self, camera: AndorIXon897, *, max_queue: int = 16, wait_timeout_ms: int = 1000) -> None:
        self.camera = camera
        self.frames: queue.Queue[FramePacket] = queue.Queue(maxsize=max_queue)
        self.wait_timeout_ms = wait_timeout_ms
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.dropped_frames = 0
        self.acquired_frames = 0
        self.last_error: BaseException | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.last_error = None
        self._thread = threading.Thread(target=self._run, name="AndorContinuousAcquisition", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 5.0) -> None:
        self._stop.set()
        try:
            self.camera.sdk.cancel_wait()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(join_timeout_s)
        self.camera.stop_continuous()

    def _put_frame(self, packet: FramePacket) -> None:
        try:
            self.frames.put_nowait(packet)
        except queue.Full:
            self.dropped_frames += 1
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(packet)

    def _run(self) -> None:
        try:
            if self.camera.current_acquisition is None or self.camera.current_acquisition.acquisition_mode != AcquisitionMode.RUN_TILL_ABORT:
                self.camera.configure_acquisition(AcquisitionSettings(acquisition_mode=AcquisitionMode.RUN_TILL_ABORT))
            self.camera.start_continuous()
            while not self._stop.is_set():
                try:
                    self.camera.sdk.wait_for_acquisition(timeout_ms=self.wait_timeout_ms)
                except Exception:
                    # Timeout can happen with external triggers.  Keep thread alive
                    # unless stop was requested.
                    if self._stop.is_set():
                        break
                frames = self.camera.get_new_frames()
                for frame in frames:
                    self.acquired_frames += 1
                    self._put_frame(FramePacket(frame=frame, index=self.acquired_frames, timestamp_s=time.time()))
        except BaseException as exc:  # intentionally broad: propagate to GUI/status panel
            self.last_error = exc
        finally:
            try:
                self.camera.stop_continuous()
            except Exception:
                pass
