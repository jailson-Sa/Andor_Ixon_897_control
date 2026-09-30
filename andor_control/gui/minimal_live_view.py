"""Diagnostic PySide6/pyqtgraph GUI for the Andor iXon 897.

This is still intentionally compact, but it exposes the controls that matter
for diagnosing whether image acquisition is really working: shutter state,
trigger mode, exposure, ROI/binning, EM gain, readout speeds, cooler/fan and
basic live-view status.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

try:  # pragma: no cover - exercised manually on the camera PC
    from PySide6 import QtCore, QtWidgets
    import pyqtgraph as pg
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The GUI dependencies are not installed.\n"
        "Install them from the project directory with:\n\n"
        "  py -3.11 -m pip install -e .[gui]\n\n"
        "or:\n\n"
        "  py -3.11 -m pip install -r requirements-gui.txt\n"
    ) from exc

from andor_control.camera.ixon897 import AndorIXon897
from andor_control.camera.settings import AcquisitionSettings, EMCCDSettings, ROI, ShutterSettings, TemperatureSettings
from andor_control.sdk2.enums import AcquisitionMode, EMGainMode, FanMode, ReadMode, ShutterMode, TriggerMode
from andor_control.sdk2.mock import MockAndorSDK2
from andor_control.storage.simple_io import save_frame_npy, save_metadata_json

E = TypeVar("E")


class DiagnosticLiveViewWindow(QtWidgets.QMainWindow):  # pragma: no cover - GUI runtime
    """Practical diagnostic GUI for image detection with the iXon 897."""

    def __init__(self, *, dll_path: str | None = None, mock: bool = False, output_dir: str | Path = "gui_output") -> None:
        super().__init__()
        self.setWindowTitle("Andor iXon 897 — Diagnostic Live View")
        self.resize(1450, 900)

        self.dll_path = dll_path
        self.mock = mock
        self.output_dir = Path(output_dir)
        self.camera: AndorIXon897 | None = None
        self.last_frame: np.ndarray | None = None
        self.live_running = False
        self.frame_counter = 0
        self.last_live_t = time.monotonic()
        self.live_fps = 0.0
        self.last_acq_info: dict[str, Any] | None = None
        self.capability_summary: dict[str, Any] = {}

        self._build_ui()
        self._connect_signals()

        self.live_timer = QtCore.QTimer(self)
        self.live_timer.setInterval(80)
        self.live_timer.timeout.connect(self._update_live_frame)

        self.status_timer = QtCore.QTimer(self)
        self.status_timer.setInterval(1500)
        self.status_timer.timeout.connect(self._refresh_camera_status)

        if self.mock:
            self.dll_edit.setText("MOCK BACKEND")
        elif self.dll_path:
            self.dll_edit.setText(self.dll_path)

        self._set_connected_ui(False)

    # UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)

        controls_container = QtWidgets.QWidget(self)
        controls_layout = QtWidgets.QVBoxLayout(controls_container)
        controls_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls_container)
        scroll.setMinimumWidth(430)
        scroll.setMaximumWidth(470)

        # Connection ------------------------------------------------------
        connection_box = QtWidgets.QGroupBox("Connection")
        connection_layout = QtWidgets.QGridLayout(connection_box)
        self.dll_edit = QtWidgets.QLineEdit()
        self.dll_edit.setPlaceholderText(r"C:\Path\To\atmcd64d.dll or leave empty if in PATH")
        self.browse_button = QtWidgets.QPushButton("Browse DLL")
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.disconnect_button = QtWidgets.QPushButton("Disconnect")
        self.identity_label = QtWidgets.QLabel("Not connected")
        self.identity_label.setWordWrap(True)
        connection_layout.addWidget(QtWidgets.QLabel("DLL:"), 0, 0)
        connection_layout.addWidget(self.dll_edit, 0, 1, 1, 2)
        connection_layout.addWidget(self.browse_button, 1, 1)
        connection_layout.addWidget(self.connect_button, 2, 1)
        connection_layout.addWidget(self.disconnect_button, 2, 2)
        connection_layout.addWidget(self.identity_label, 3, 0, 1, 3)
        controls_layout.addWidget(connection_box)

        # Temperature -----------------------------------------------------
        temp_box = QtWidgets.QGroupBox("Temperature / cooler")
        temp_layout = QtWidgets.QGridLayout(temp_box)
        self.temp_setpoint_spin = QtWidgets.QSpinBox()
        self.temp_setpoint_spin.setRange(-120, 20)
        self.temp_setpoint_spin.setValue(-70)
        self.cooler_check = QtWidgets.QCheckBox("Cooler ON")
        self.cooler_check.setChecked(True)
        self.keep_cooler_check = QtWidgets.QCheckBox("Keep cooler ON at SDK shutdown")
        self.keep_cooler_check.setChecked(False)
        self.fan_combo = self._enum_combo(FanMode, FanMode.FULL)
        self.apply_temp_button = QtWidgets.QPushButton("Apply temperature")
        self.temp_status_label = QtWidgets.QLabel("Temperature: --")
        self.temp_status_label.setWordWrap(True)
        temp_layout.addWidget(QtWidgets.QLabel("Setpoint (°C):"), 0, 0)
        temp_layout.addWidget(self.temp_setpoint_spin, 0, 1)
        temp_layout.addWidget(QtWidgets.QLabel("Fan:"), 1, 0)
        temp_layout.addWidget(self.fan_combo, 1, 1)
        temp_layout.addWidget(self.cooler_check, 2, 0, 1, 2)
        temp_layout.addWidget(self.keep_cooler_check, 3, 0, 1, 2)
        temp_layout.addWidget(self.apply_temp_button, 4, 0, 1, 2)
        temp_layout.addWidget(self.temp_status_label, 5, 0, 1, 2)
        controls_layout.addWidget(temp_box)

        # Shutter ---------------------------------------------------------
        shutter_box = QtWidgets.QGroupBox("Shutter diagnostic")
        shutter_layout = QtWidgets.QGridLayout(shutter_box)
        self.internal_shutter_combo = self._enum_combo(ShutterMode, ShutterMode.PERMANENTLY_OPEN)
        self.external_shutter_combo = self._enum_combo(ShutterMode, ShutterMode.PERMANENTLY_OPEN)
        self.ttl_high_check = QtWidgets.QCheckBox("TTL high opens external shutter")
        self.ttl_high_check.setChecked(False)
        self.opening_ms_spin = QtWidgets.QSpinBox(); self.opening_ms_spin.setRange(0, 10000); self.opening_ms_spin.setValue(0)
        self.closing_ms_spin = QtWidgets.QSpinBox(); self.closing_ms_spin.setRange(0, 10000); self.closing_ms_spin.setValue(0)
        self.apply_shutter_button = QtWidgets.QPushButton("Apply shutter")
        self.force_open_button = QtWidgets.QPushButton("Force open both")
        self.auto_internal_button = QtWidgets.QPushButton("Auto internal")
        self.force_closed_button = QtWidgets.QPushButton("Force closed both")
        self.shutter_info_label = QtWidgets.QLabel("Shutter info: --")
        self.shutter_info_label.setWordWrap(True)
        shutter_layout.addWidget(QtWidgets.QLabel("Internal:"), 0, 0)
        shutter_layout.addWidget(self.internal_shutter_combo, 0, 1)
        shutter_layout.addWidget(QtWidgets.QLabel("External/TTL:"), 1, 0)
        shutter_layout.addWidget(self.external_shutter_combo, 1, 1)
        shutter_layout.addWidget(self.ttl_high_check, 2, 0, 1, 2)
        shutter_layout.addWidget(QtWidgets.QLabel("Open/close ms:"), 3, 0)
        shutter_layout.addWidget(self.opening_ms_spin, 3, 1)
        shutter_layout.addWidget(self.closing_ms_spin, 3, 2)
        shutter_layout.addWidget(self.apply_shutter_button, 4, 0, 1, 3)
        shutter_layout.addWidget(self.force_open_button, 5, 0)
        shutter_layout.addWidget(self.auto_internal_button, 5, 1)
        shutter_layout.addWidget(self.force_closed_button, 5, 2)
        shutter_layout.addWidget(self.shutter_info_label, 6, 0, 1, 3)
        controls_layout.addWidget(shutter_box)

        # Acquisition and trigger ----------------------------------------
        acq_box = QtWidgets.QGroupBox("Acquisition / trigger")
        acq_layout = QtWidgets.QGridLayout(acq_box)
        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setRange(0.000001, 3600.0)
        self.exposure_spin.setDecimals(6)
        self.exposure_spin.setSingleStep(0.01)
        self.exposure_spin.setValue(0.03)
        self.trigger_combo = self._enum_combo(TriggerMode, TriggerMode.INTERNAL)
        self.read_mode_combo = self._enum_combo(ReadMode, ReadMode.IMAGE, allowed=[ReadMode.IMAGE])
        self.frame_transfer_check = QtWidgets.QCheckBox("Frame transfer")
        self.apply_settings_button = QtWidgets.QPushButton("Apply all camera settings")
        acq_layout.addWidget(QtWidgets.QLabel("Exposure (s):"), 0, 0)
        acq_layout.addWidget(self.exposure_spin, 0, 1)
        acq_layout.addWidget(QtWidgets.QLabel("Trigger:"), 1, 0)
        acq_layout.addWidget(self.trigger_combo, 1, 1)
        acq_layout.addWidget(QtWidgets.QLabel("Readout:"), 2, 0)
        acq_layout.addWidget(self.read_mode_combo, 2, 1)
        acq_layout.addWidget(self.frame_transfer_check, 3, 0, 1, 2)
        acq_layout.addWidget(self.apply_settings_button, 4, 0, 1, 2)
        controls_layout.addWidget(acq_box)

        # EMCCD/readout ---------------------------------------------------
        em_box = QtWidgets.QGroupBox("EMCCD / readout")
        em_layout = QtWidgets.QGridLayout(em_box)
        self.em_gain_mode_combo = self._enum_combo(EMGainMode, EMGainMode.REAL_GAIN)
        self.em_gain_spin = QtWidgets.QSpinBox(); self.em_gain_spin.setRange(1, 300); self.em_gain_spin.setValue(7)
        self.em_advanced_check = QtWidgets.QCheckBox("EM advanced")
        self.allow_gain_check = QtWidgets.QCheckBox("Allow gain above minimum before stable")
        self.allow_gain_check.setToolTip("Use only during controlled tests. High EM gain should normally wait for cold/stable temperature.")
        self.preamp_combo = QtWidgets.QComboBox()
        self.hs_speed_combo = QtWidgets.QComboBox()
        self.vs_speed_combo = QtWidgets.QComboBox()
        self.baseline_clamp_check = QtWidgets.QCheckBox("Baseline clamp")
        self.baseline_clamp_check.setChecked(True)
        em_layout.addWidget(QtWidgets.QLabel("EM gain mode:"), 0, 0)
        em_layout.addWidget(self.em_gain_mode_combo, 0, 1)
        em_layout.addWidget(QtWidgets.QLabel("EM gain:"), 1, 0)
        em_layout.addWidget(self.em_gain_spin, 1, 1)
        em_layout.addWidget(self.em_advanced_check, 2, 0, 1, 2)
        em_layout.addWidget(self.allow_gain_check, 3, 0, 1, 2)
        em_layout.addWidget(QtWidgets.QLabel("Preamp:"), 4, 0)
        em_layout.addWidget(self.preamp_combo, 4, 1)
        em_layout.addWidget(QtWidgets.QLabel("HS speed:"), 5, 0)
        em_layout.addWidget(self.hs_speed_combo, 5, 1)
        em_layout.addWidget(QtWidgets.QLabel("VS speed:"), 6, 0)
        em_layout.addWidget(self.vs_speed_combo, 6, 1)
        em_layout.addWidget(self.baseline_clamp_check, 7, 0, 1, 2)
        controls_layout.addWidget(em_box)

        # ROI -------------------------------------------------------------
        roi_box = QtWidgets.QGroupBox("ROI / binning")
        roi_layout = QtWidgets.QGridLayout(roi_box)
        self.hstart_spin = QtWidgets.QSpinBox(); self.hstart_spin.setRange(1, 512); self.hstart_spin.setValue(1)
        self.hend_spin = QtWidgets.QSpinBox(); self.hend_spin.setRange(1, 512); self.hend_spin.setValue(512)
        self.vstart_spin = QtWidgets.QSpinBox(); self.vstart_spin.setRange(1, 512); self.vstart_spin.setValue(1)
        self.vend_spin = QtWidgets.QSpinBox(); self.vend_spin.setRange(1, 512); self.vend_spin.setValue(512)
        self.hbin_spin = QtWidgets.QSpinBox(); self.hbin_spin.setRange(1, 16); self.hbin_spin.setValue(1)
        self.vbin_spin = QtWidgets.QSpinBox(); self.vbin_spin.setRange(1, 16); self.vbin_spin.setValue(1)
        self.full_roi_button = QtWidgets.QPushButton("Full frame")
        self.center_roi_button = QtWidgets.QPushButton("Center 128×128")
        roi_layout.addWidget(QtWidgets.QLabel("H start/end:"), 0, 0)
        roi_layout.addWidget(self.hstart_spin, 0, 1); roi_layout.addWidget(self.hend_spin, 0, 2)
        roi_layout.addWidget(QtWidgets.QLabel("V start/end:"), 1, 0)
        roi_layout.addWidget(self.vstart_spin, 1, 1); roi_layout.addWidget(self.vend_spin, 1, 2)
        roi_layout.addWidget(QtWidgets.QLabel("H/V bin:"), 2, 0)
        roi_layout.addWidget(self.hbin_spin, 2, 1); roi_layout.addWidget(self.vbin_spin, 2, 2)
        roi_layout.addWidget(self.full_roi_button, 3, 0, 1, 2)
        roi_layout.addWidget(self.center_roi_button, 3, 2)
        controls_layout.addWidget(roi_box)

        # Actions ---------------------------------------------------------
        action_box = QtWidgets.QGroupBox("Image detection")
        action_layout = QtWidgets.QGridLayout(action_box)
        self.capture_button = QtWidgets.QPushButton("Capture single frame")
        self.start_live_button = QtWidgets.QPushButton("Start live")
        self.stop_live_button = QtWidgets.QPushButton("Stop live")
        self.software_trigger_button = QtWidgets.QPushButton("Software trigger")
        self.autolevel_check = QtWidgets.QCheckBox("Auto-level display")
        self.autolevel_check.setChecked(True)
        self.save_button = QtWidgets.QPushButton("Save last frame")
        action_layout.addWidget(self.capture_button, 0, 0, 1, 2)
        action_layout.addWidget(self.start_live_button, 1, 0)
        action_layout.addWidget(self.stop_live_button, 1, 1)
        action_layout.addWidget(self.software_trigger_button, 2, 0, 1, 2)
        action_layout.addWidget(self.autolevel_check, 3, 0, 1, 2)
        action_layout.addWidget(self.save_button, 4, 0, 1, 2)
        controls_layout.addWidget(action_box)

        # Status ----------------------------------------------------------
        status_box = QtWidgets.QGroupBox("Runtime status")
        status_layout = QtWidgets.QVBoxLayout(status_box)
        self.camera_status_label = QtWidgets.QLabel("SDK status: --")
        self.camera_status_label.setWordWrap(True)
        self.frame_stats_label = QtWidgets.QLabel("Frame statistics: --")
        self.frame_stats_label.setWordWrap(True)
        status_layout.addWidget(self.camera_status_label)
        status_layout.addWidget(self.frame_stats_label)
        controls_layout.addWidget(status_box)
        controls_layout.addStretch(1)

        # Image panel -----------------------------------------------------
        right_panel = QtWidgets.QWidget(self)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.setPredefinedGradient("grey")
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(170)
        right_layout.addWidget(self.image_view, stretch=1)
        right_layout.addWidget(self.log_box)

        root_layout.addWidget(scroll)
        root_layout.addWidget(right_panel, stretch=1)
        self.statusBar().showMessage("Ready")

    def _connect_signals(self) -> None:
        self.browse_button.clicked.connect(self._browse_dll)
        self.connect_button.clicked.connect(self.connect_camera)
        self.disconnect_button.clicked.connect(self.disconnect_camera)
        self.apply_temp_button.clicked.connect(self.apply_temperature)
        self.apply_shutter_button.clicked.connect(self.apply_shutter)
        self.force_open_button.clicked.connect(self.force_shutter_open)
        self.auto_internal_button.clicked.connect(self.set_internal_shutter_auto)
        self.force_closed_button.clicked.connect(self.force_shutter_closed)
        self.apply_settings_button.clicked.connect(lambda: self.apply_camera_settings(AcquisitionMode.SINGLE_SCAN))
        self.capture_button.clicked.connect(self.capture_single_frame)
        self.start_live_button.clicked.connect(self.start_live)
        self.stop_live_button.clicked.connect(self.stop_live)
        self.software_trigger_button.clicked.connect(self.send_software_trigger)
        self.save_button.clicked.connect(self.save_last_frame)
        self.full_roi_button.clicked.connect(self._set_full_roi)
        self.center_roi_button.clicked.connect(self._set_center_roi_128)

    # UI state ------------------------------------------------------------
    def _set_connected_ui(self, connected: bool) -> None:
        widgets = [
            self.disconnect_button,
            self.apply_temp_button,
            self.apply_shutter_button,
            self.force_open_button,
            self.auto_internal_button,
            self.force_closed_button,
            self.apply_settings_button,
            self.capture_button,
            self.start_live_button,
            self.stop_live_button,
            self.software_trigger_button,
            self.save_button,
            self.temp_setpoint_spin,
            self.cooler_check,
            self.keep_cooler_check,
            self.fan_combo,
            self.internal_shutter_combo,
            self.external_shutter_combo,
            self.ttl_high_check,
            self.opening_ms_spin,
            self.closing_ms_spin,
            self.exposure_spin,
            self.trigger_combo,
            self.read_mode_combo,
            self.frame_transfer_check,
            self.em_gain_mode_combo,
            self.em_gain_spin,
            self.em_advanced_check,
            self.allow_gain_check,
            self.preamp_combo,
            self.hs_speed_combo,
            self.vs_speed_combo,
            self.baseline_clamp_check,
            self.hstart_spin,
            self.hend_spin,
            self.vstart_spin,
            self.vend_spin,
            self.hbin_spin,
            self.vbin_spin,
            self.full_roi_button,
            self.center_roi_button,
            self.autolevel_check,
        ]
        for widget in widgets:
            widget.setEnabled(connected)
        self.connect_button.setEnabled(not connected)
        self.dll_edit.setEnabled(not connected and not self.mock)
        self.browse_button.setEnabled(not connected and not self.mock)
        self.stop_live_button.setEnabled(False)
        self.save_button.setEnabled(False)

    def _set_live_ui(self, running: bool) -> None:
        self.start_live_button.setEnabled(not running and self.camera is not None)
        self.stop_live_button.setEnabled(running)
        self.capture_button.setEnabled(not running and self.camera is not None)
        self.apply_settings_button.setEnabled(not running and self.camera is not None)
        # Shutter can still be changed while live for diagnostics, but ROI/gain/readout should not.
        for widget in [
            self.exposure_spin, self.trigger_combo, self.read_mode_combo, self.frame_transfer_check,
            self.em_gain_mode_combo, self.em_gain_spin, self.em_advanced_check, self.preamp_combo,
            self.hs_speed_combo, self.vs_speed_combo, self.baseline_clamp_check,
            self.hstart_spin, self.hend_spin, self.vstart_spin, self.vend_spin, self.hbin_spin, self.vbin_spin,
            self.full_roi_button, self.center_roi_button,
        ]:
            widget.setEnabled(not running and self.camera is not None)
        self.save_button.setEnabled(self.last_frame is not None)

    # Helpers -------------------------------------------------------------
    @staticmethod
    def _enum_combo(enum_cls: type[E], default: E, allowed: list[E] | None = None) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        items = allowed if allowed is not None else list(enum_cls)  # type: ignore[arg-type]
        for item in items:
            combo.addItem(getattr(item, "name", str(item)), int(item))
        idx = combo.findData(int(default))
        if idx >= 0:
            combo.setCurrentIndex(idx)
        return combo

    @staticmethod
    def _combo_enum(combo: QtWidgets.QComboBox, enum_cls: type[E]) -> E:
        return enum_cls(int(combo.currentData()))  # type: ignore[call-arg]

    @staticmethod
    def _combo_int(combo: QtWidgets.QComboBox, default: int = 0) -> int:
        data = combo.currentData()
        return int(data) if data is not None else default

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{timestamp}] {message}")
        self.statusBar().showMessage(message, 5000)

    def _show_error(self, title: str, exc: BaseException | str) -> None:
        text = str(exc)
        self._log(f"ERROR: {text}")
        QtWidgets.QMessageBox.critical(self, title, text)

    def _browse_dll(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select atmcd64d.dll", "", "Andor SDK2 DLL (atmcd64d.dll);;DLL files (*.dll);;All files (*.*)"
        )
        if path:
            self.dll_edit.setText(path)

    def _detector_shape(self) -> tuple[int, int]:
        if self.camera is not None and self.camera.identity is not None:
            return self.camera.identity.detector_shape
        return (512, 512)

    def _set_full_roi(self) -> None:
        width, height = self._detector_shape()
        for spin, value, maximum in [
            (self.hstart_spin, 1, width),
            (self.hend_spin, width, width),
            (self.vstart_spin, 1, height),
            (self.vend_spin, height, height),
        ]:
            spin.setRange(1, maximum)
            spin.setValue(value)
        self.hbin_spin.setValue(1)
        self.vbin_spin.setValue(1)

    def _set_center_roi_128(self) -> None:
        width, height = self._detector_shape()
        size = min(128, width, height)
        h0 = width // 2 - size // 2 + 1
        v0 = height // 2 - size // 2 + 1
        self.hstart_spin.setValue(h0)
        self.hend_spin.setValue(h0 + size - 1)
        self.vstart_spin.setValue(v0)
        self.vend_spin.setValue(v0 + size - 1)
        self.hbin_spin.setValue(1)
        self.vbin_spin.setValue(1)

    def _roi_from_ui(self) -> ROI:
        return ROI(
            hstart=int(self.hstart_spin.value()),
            hend=int(self.hend_spin.value()),
            vstart=int(self.vstart_spin.value()),
            vend=int(self.vend_spin.value()),
            hbin=int(self.hbin_spin.value()),
            vbin=int(self.vbin_spin.value()),
        )

    def _temperature_from_ui(self) -> TemperatureSettings:
        return TemperatureSettings(
            setpoint_c=int(self.temp_setpoint_spin.value()),
            cooler_on=bool(self.cooler_check.isChecked()),
            keep_cooler_on_at_shutdown=bool(self.keep_cooler_check.isChecked()),
            fan_mode=self._combo_enum(self.fan_combo, FanMode),
        )

    def _shutter_from_ui(self) -> ShutterSettings:
        return ShutterSettings(
            ttl_high_opens=bool(self.ttl_high_check.isChecked()),
            internal_mode=self._combo_enum(self.internal_shutter_combo, ShutterMode),
            external_mode=self._combo_enum(self.external_shutter_combo, ShutterMode),
            opening_ms=int(self.opening_ms_spin.value()),
            closing_ms=int(self.closing_ms_spin.value()),
        )

    def _acquisition_from_ui(self, mode: AcquisitionMode) -> AcquisitionSettings:
        return AcquisitionSettings(
            acquisition_mode=mode,
            read_mode=self._combo_enum(self.read_mode_combo, ReadMode),
            trigger_mode=self._combo_enum(self.trigger_combo, TriggerMode),
            exposure_s=float(self.exposure_spin.value()),
            frame_transfer=bool(self.frame_transfer_check.isChecked()),
            roi=self._roi_from_ui(),
        )

    def _emccd_from_ui(self) -> EMCCDSettings:
        return EMCCDSettings(
            output_amplifier=0,
            ad_channel=0,
            hs_speed_index=self._combo_int(self.hs_speed_combo, 0),
            vs_speed_index=self._combo_int(self.vs_speed_combo, 0),
            preamp_gain_index=self._combo_int(self.preamp_combo, 0),
            em_gain_mode=self._combo_enum(self.em_gain_mode_combo, EMGainMode),
            em_gain=int(self.em_gain_spin.value()),
            em_advanced=bool(self.em_advanced_check.isChecked()),
            baseline_clamp=bool(self.baseline_clamp_check.isChecked()),
        )

    def _populate_capability_controls(self) -> None:
        if self.camera is None:
            return
        caps = self.camera.get_capability_summary()
        self.capability_summary = caps

        gain_min, gain_max = caps.get("em_gain_range", (1, 300))
        self.em_gain_spin.setRange(int(gain_min), int(gain_max))
        if self.em_gain_spin.value() < int(gain_min):
            self.em_gain_spin.setValue(int(gain_min))

        self.preamp_combo.clear()
        for index, value in enumerate(caps.get("preamp_gains", [])):
            self.preamp_combo.addItem(f"{index}: {value:.3g}×", index)
        if self.preamp_combo.count() == 0:
            self.preamp_combo.addItem("0", 0)

        self.hs_speed_combo.clear()
        for index, value in enumerate(caps.get("hs_speeds_mhz_output_amp_0", [])):
            self.hs_speed_combo.addItem(f"{index}: {value:.3g} MHz", index)
        if self.hs_speed_combo.count() == 0:
            self.hs_speed_combo.addItem("0", 0)

        self.vs_speed_combo.clear()
        for index, value in enumerate(caps.get("vs_speeds_us", [])):
            self.vs_speed_combo.addItem(f"{index}: {value:.3g} µs", index)
        if self.vs_speed_combo.count() == 0:
            self.vs_speed_combo.addItem("0", 0)

        info = []
        if "has_internal_mechanical_shutter" in caps:
            info.append(f"internal mechanical: {caps['has_internal_mechanical_shutter']}")
        if "shutter_min_times_ms" in caps:
            info.append(f"min open/close ms: {caps['shutter_min_times_ms']}")
        self.shutter_info_label.setText("Shutter info: " + (" | ".join(info) if info else "available controls loaded"))

    # Camera actions ------------------------------------------------------
    def connect_camera(self) -> None:
        try:
            if self.mock:
                self.camera = AndorIXon897(MockAndorSDK2())
            else:
                dll = self.dll_edit.text().strip() or None
                self.camera = AndorIXon897.from_dll(dll)
            ident = self.camera.initialize()
            self._set_full_roi()
            self._populate_capability_controls()
            self.identity_label.setText(
                f"{ident.model} | serial {ident.serial_number}\n"
                f"Detector: {ident.detector_shape[0]} × {ident.detector_shape[1]} | Pixel: {ident.pixel_size_um[0]:.1f} µm\n"
                f"HW: {ident.hardware_version} | SDK: {ident.software_version}"
            )
            self._set_connected_ui(True)
            self.apply_temperature(show_errors=False)
            self.apply_shutter(show_errors=False)
            self.apply_camera_settings(AcquisitionMode.SINGLE_SCAN, show_errors=False)
            self.status_timer.start()
            self._refresh_camera_status()
            self._log("Camera connected and diagnostic defaults applied")
        except BaseException as exc:
            self.camera = None
            self._set_connected_ui(False)
            self._show_error("Connection failed", exc)

    def disconnect_camera(self) -> None:
        self.stop_live()
        if self.camera is not None:
            try:
                self.camera.shutdown(turn_cooler_off=False)
            except BaseException as exc:
                self._log(f"Shutdown warning: {exc}")
        self.camera = None
        self.status_timer.stop()
        self.identity_label.setText("Not connected")
        self.temp_status_label.setText("Temperature: --")
        self.camera_status_label.setText("SDK status: --")
        self._set_connected_ui(False)
        self._log("Camera disconnected. Cooler was not forcibly turned off.")

    def apply_temperature(self, *, show_errors: bool = True) -> bool:
        if self.camera is None:
            return False
        try:
            state = self.camera.configure_temperature(self._temperature_from_ui())
            self._update_temperature_label(state.to_dict())
            self._log(
                f"Temperature applied: setpoint {self.temp_setpoint_spin.value()} °C, "
                f"cooler {'ON' if self.cooler_check.isChecked() else 'OFF'}, fan={self.fan_combo.currentText()}"
            )
            return True
        except BaseException as exc:
            if show_errors:
                self._show_error("Temperature error", exc)
            else:
                self._log(f"Temperature setup warning: {exc}")
            return False

    def apply_shutter(self, *, show_errors: bool = True) -> bool:
        if self.camera is None:
            return False
        try:
            settings = self.camera.configure_shutter(self._shutter_from_ui())
            self._log(
                "Shutter applied: "
                f"internal={settings.internal_mode.name}, external={settings.external_mode.name}, "
                f"TTL={'HIGH opens' if settings.ttl_high_opens else 'LOW opens'}, "
                f"open/close={settings.opening_ms}/{settings.closing_ms} ms"
            )
            return True
        except BaseException as exc:
            if show_errors:
                self._show_error("Shutter error", exc)
            else:
                self._log(f"Shutter setup warning: {exc}")
            return False

    def force_shutter_open(self) -> None:
        self._set_combo_data(self.internal_shutter_combo, int(ShutterMode.PERMANENTLY_OPEN))
        self._set_combo_data(self.external_shutter_combo, int(ShutterMode.PERMANENTLY_OPEN))
        self.apply_shutter()

    def set_internal_shutter_auto(self) -> None:
        self._set_combo_data(self.internal_shutter_combo, int(ShutterMode.FULLY_AUTO))
        self._set_combo_data(self.external_shutter_combo, int(ShutterMode.PERMANENTLY_OPEN))
        self.apply_shutter()

    def force_shutter_closed(self) -> None:
        self._set_combo_data(self.internal_shutter_combo, int(ShutterMode.PERMANENTLY_CLOSED))
        self._set_combo_data(self.external_shutter_combo, int(ShutterMode.PERMANENTLY_CLOSED))
        self.apply_shutter()

    @staticmethod
    def _set_combo_data(combo: QtWidgets.QComboBox, value: int) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def apply_camera_settings(self, mode: AcquisitionMode, *, show_errors: bool = True) -> bool:
        if self.camera is None:
            return False
        try:
            if self.live_running:
                raise RuntimeError("Stop live acquisition before changing acquisition/readout settings")
            # Apply shutter every time so live/single acquisition cannot silently inherit a closed SDK shutter state.
            self.apply_shutter(show_errors=False)
            settings = self._acquisition_from_ui(mode)
            info = self.camera.configure_acquisition(settings)
            actual_gain = self.camera.configure_emccd(
                self._emccd_from_ui(),
                allow_high_gain_without_stable_temp=bool(self.allow_gain_check.isChecked()),
                clamp_gain_to_sdk_range=True,
            )
            if actual_gain.em_gain is not None:
                self.em_gain_spin.setValue(int(actual_gain.em_gain))
            self.last_acq_info = info
            timing = info["actual_timings_s"]
            self._log(
                f"Acquisition applied: mode={mode.name}, trigger={settings.trigger_mode.name}, "
                f"exposure={timing['exposure']:.6f} s, readout={info['readout_time_s']:.6f} s, "
                f"ROI output={info['roi_output_shape']}, EM gain={actual_gain.em_gain}"
            )
            return True
        except BaseException as exc:
            if show_errors:
                self._show_error("Acquisition configuration error", exc)
            else:
                self._log(f"Acquisition setup warning: {exc}")
            return False

    def capture_single_frame(self) -> None:
        if self.camera is None:
            return
        try:
            if self.live_running:
                self.stop_live()
            if not self.apply_camera_settings(AcquisitionMode.SINGLE_SCAN):
                return
            timeout_ms = max(30000, int(self.exposure_spin.value() * 3000 + 10000))
            frame = self.camera.acquire_single_frame(timeout_ms=timeout_ms)
            self._display_frame(frame)
            self._log("Single frame acquired")
        except BaseException as exc:
            self._show_error("Single-frame acquisition failed", exc)

    def start_live(self) -> None:
        if self.camera is None:
            return
        try:
            if not self.apply_camera_settings(AcquisitionMode.RUN_TILL_ABORT):
                return
            self.camera.start_continuous()
            self.live_running = True
            self.frame_counter = 0
            self.last_live_t = time.monotonic()
            self.live_timer.start()
            self._set_live_ui(True)
            self._log("Live acquisition started")
        except BaseException as exc:
            self.live_running = False
            self._set_live_ui(False)
            self._show_error("Live acquisition failed", exc)

    def stop_live(self) -> None:
        self.live_timer.stop()
        if self.camera is not None:
            try:
                self.camera.stop_continuous()
            except BaseException as exc:
                self._log(f"Stop warning: {exc}")
        was_running = self.live_running
        self.live_running = False
        if self.camera is not None:
            self._set_live_ui(False)
        if was_running:
            self._log("Live acquisition stopped")

    def send_software_trigger(self) -> None:
        if self.camera is None:
            return
        try:
            self.camera.sdk.send_software_trigger()
            self._log("Software trigger sent")
        except BaseException as exc:
            self._show_error("Software trigger failed", exc)

    def _update_live_frame(self) -> None:
        if self.camera is None or not self.live_running:
            return
        try:
            frame = self.camera.get_latest_frame()
        except BaseException as exc:
            self._log(f"Live frame not ready: {exc}")
            return
        self._display_frame(frame)
        self.frame_counter += 1
        now = time.monotonic()
        dt = now - self.last_live_t
        if dt >= 1.0:
            self.live_fps = self.frame_counter / dt
            self.frame_counter = 0
            self.last_live_t = now
            self._update_frame_stats()

    def _display_frame(self, frame: np.ndarray) -> None:
        self.last_frame = np.asarray(frame, dtype=np.int32)
        self.image_view.setImage(self.last_frame, autoLevels=bool(self.autolevel_check.isChecked()))
        self.save_button.setEnabled(True)
        self._update_frame_stats()

    def _update_frame_stats(self) -> None:
        if self.last_frame is None:
            self.frame_stats_label.setText("Frame statistics: --")
            return
        frame = self.last_frame
        saturated14 = int(np.count_nonzero(frame >= 16383))
        saturated16 = int(np.count_nonzero(frame >= 65535))
        p01, p99 = np.percentile(frame, [1, 99])
        self.frame_stats_label.setText(
            "Frame statistics:\n"
            f"shape={frame.shape[1]}×{frame.shape[0]}, min={int(frame.min())}, max={int(frame.max())}, "
            f"mean={float(frame.mean()):.2f}, median={float(np.median(frame)):.2f}, std={float(frame.std()):.2f}\n"
            f"p1={float(p01):.1f}, p99={float(p99):.1f}, sat14={saturated14}, sat16={saturated16}, live FPS≈{self.live_fps:.1f}"
        )

    def _refresh_camera_status(self) -> None:
        if self.camera is None:
            return
        try:
            temp = self.camera.get_temperature_state().to_dict()
            self._update_temperature_label(temp)
            status = self.camera.get_status()
            total = "--"
            buf = "--"
            try:
                total = str(self.camera.sdk.get_total_number_images_acquired())
            except Exception:
                pass
            try:
                buf = str(self.camera.sdk.get_circular_buffer_size())
            except Exception:
                pass
            self.camera_status_label.setText(
                f"SDK status: {status}\n"
                f"Total images acquired: {total} | circular buffer: {buf}\n"
                f"Live running: {self.live_running} | output dir: {self.output_dir}"
            )
        except BaseException as exc:
            self._log(f"Status read warning: {exc}")

    def _update_temperature_label(self, state: dict[str, Any]) -> None:
        self.temp_status_label.setText(
            f"Temperature: {state['temperature_c']:.2f} °C\n"
            f"Status: {state['status']} | Cooler: {'ON' if state['cooler_on'] else 'OFF'}"
        )

    def save_last_frame(self) -> None:
        if self.last_frame is None or self.camera is None:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            frame_path = self.output_dir / f"andor_frame_{stamp}.npy"
            metadata_path = self.output_dir / f"andor_frame_{stamp}.json"
            metadata = self.camera.build_metadata()
            metadata["frame_statistics"] = {
                "shape": list(self.last_frame.shape),
                "min": int(self.last_frame.min()),
                "max": int(self.last_frame.max()),
                "mean": float(self.last_frame.mean()),
                "median": float(np.median(self.last_frame)),
                "std": float(self.last_frame.std()),
            }
            metadata["capability_summary"] = self.capability_summary
            metadata["last_acquisition_info"] = self.last_acq_info
            save_frame_npy(frame_path, self.last_frame)
            save_metadata_json(metadata_path, metadata)
            self._log(f"Saved {frame_path.name} and {metadata_path.name}")
        except BaseException as exc:
            self._show_error("Save failed", exc)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt method name
        try:
            self.disconnect_camera()
        finally:
            event.accept()


# Backward-compatible name used by scripts/minimal_gui.py in milestone 3 docs.
MinimalLiveViewWindow = DiagnosticLiveViewWindow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic GUI live view for Andor iXon 897")
    parser.add_argument("--dll", default=None, help="Path to atmcd64d.dll. If omitted, normal SDK DLL lookup is used.")
    parser.add_argument("--mock", action="store_true", help="Use simulated camera backend instead of real hardware")
    parser.add_argument("--output", default="gui_output", help="Directory where saved frames/metadata are written")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    window = DiagnosticLiveViewWindow(dll_path=args.dll, mock=args.mock, output_dir=args.output)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
