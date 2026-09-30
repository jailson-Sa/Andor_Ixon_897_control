"""Scientific acquisition GUI for the Andor iXon 897 EMCCD.

Milestone 5 turns the validated diagnostic GUI into a first laboratory-oriented
application: session folders, complete metadata, raw data preservation, HDF5/NPZ
stack saving, calibration frames, ROI statistics and photon-counting
post-processing.
"""
from __future__ import annotations

import argparse
import json
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

from andor_control.analysis import (
    CalibrationSet,
    PhotonCountingSettings,
    UnitConversionSettings,
    apply_calibration,
    build_auto_calibration_profile,
    convert_display_frame,
    frame_statistics,
    make_calibration_frame,
    photon_count_frame,
    roi_frame,
    save_auto_calibration_profile,
)
from andor_control.camera.ixon897 import AndorIXon897
from andor_control.camera.settings import AcquisitionSettings, EMCCDSettings, ROI, ShutterSettings, TemperatureSettings
from andor_control.sdk2.enums import AcquisitionMode, EMGainMode, FanMode, ReadMode, ShutterMode, TriggerMode
from andor_control.sdk2.mock import MockAndorSDK2
from andor_control.storage import ExperimentSession, save_frame_npy, save_metadata_json, write_hdf5_stack, write_npz_stack

E = TypeVar("E")


class SequenceWorker(QtCore.QObject):  # pragma: no cover - GUI runtime
    frame_ready = QtCore.Signal(object, int)
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(object, object, object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        camera: AndorIXon897,
        n_frames: int,
        delay_s: float,
        timeout_ms: int,
        calibration: CalibrationSet,
        correction_flags: dict[str, bool],
        photon_settings: PhotonCountingSettings,
        use_sdk_pc: bool,
    ) -> None:
        super().__init__()
        self.camera = camera
        self.n_frames = int(n_frames)
        self.delay_s = float(delay_s)
        self.timeout_ms = int(timeout_ms)
        self.calibration = calibration
        self.correction_flags = correction_flags
        self.photon_settings = photon_settings
        self.use_sdk_pc = use_sdk_pc
        self._abort = False

    @QtCore.Slot()
    def run(self) -> None:
        try:
            raw_frames: list[np.ndarray] = []
            processed_frames: list[np.ndarray] = []
            counted_frames: list[np.ndarray] = []
            for idx in range(self.n_frames):
                if self._abort:
                    break
                frame = self.camera.acquire_single_frame(timeout_ms=self.timeout_ms)
                raw_frames.append(frame.astype(np.int32, copy=False))
                processed = apply_calibration(
                    frame,
                    self.calibration,
                    subtract_bias=self.correction_flags.get("bias", False),
                    subtract_dark=self.correction_flags.get("dark", False),
                    subtract_background=self.correction_flags.get("background", False),
                    clip_negative=self.correction_flags.get("clip", False),
                )
                processed_frames.append(processed.astype(np.float32, copy=False))
                if self.photon_settings.enabled:
                    pc_settings = PhotonCountingSettings(
                        enabled=True,
                        thresholds_adu=self.photon_settings.thresholds_adu,
                        use_corrected_frame=self.photon_settings.use_corrected_frame,
                        use_sdk_postprocess=self.use_sdk_pc,
                    )
                    source = processed if pc_settings.use_corrected_frame else frame
                    result = photon_count_frame(source, pc_settings, sdk=self.camera.sdk)
                    counted_frames.append(result.counted_frame)
                self.frame_ready.emit(frame, idx + 1)
                self.progress.emit(idx + 1, self.n_frames)
                if self.delay_s > 0 and idx < self.n_frames - 1:
                    time.sleep(self.delay_s)
            raw = np.stack(raw_frames, axis=0) if raw_frames else np.empty((0, 0, 0), dtype=np.int32)
            proc = np.stack(processed_frames, axis=0) if processed_frames else None
            counted = np.stack(counted_frames, axis=0) if counted_frames else None
            self.finished.emit(raw, proc, counted)
        except BaseException as exc:
            self.error.emit(str(exc))

    @QtCore.Slot()
    def abort(self) -> None:
        self._abort = True


class ScientificAcquisitionWindow(QtWidgets.QMainWindow):  # pragma: no cover - GUI runtime
    def __init__(self, *, dll_path: str | None = None, mock: bool = False, output_dir: str | Path = "andor_data") -> None:
        super().__init__()
        self.setWindowTitle("Andor iXon 897 — Scientific Acquisition Console")
        self.setMinimumSize(1024, 660)

        # Size to 90% of the available screen, centred
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            w = int(avail.width() * 0.90)
            h = int(avail.height() * 0.90)
            self.resize(w, h)
            self.move(
                avail.x() + (avail.width() - w) // 2,
                avail.y() + (avail.height() - h) // 2,
            )

        self.dll_path = dll_path
        self.mock = mock
        self.output_base = Path(output_dir)
        self.session: ExperimentSession | None = None
        self.camera: AndorIXon897 | None = None
        self.capability_summary: dict[str, Any] = {}
        self.last_acq_info: dict[str, Any] | None = None
        self.last_raw: np.ndarray | None = None
        self.last_processed: np.ndarray | None = None
        self.last_photon_counted: np.ndarray | None = None
        self.last_pc_metadata: dict[str, Any] | None = None
        self.calibration = CalibrationSet()
        self.auto_calibration_profile = None
        self.last_unit_metadata: dict[str, Any] | None = None
        self.last_display_frame: np.ndarray | None = None
        self.requested_em_gain: int | None = None
        self.applied_em_gain: int | None = None
        self.shutter_commanded_state = "CLOSED"
        self.shutter_commanded_mode = "AUTOMATIC"
        self.controls_hidden = False
        self.clean_detection_mode = False
        self.compact_control_mode = True
        self._advanced_control_widgets: list[QtWidgets.QWidget] = []
        self._collapsible_groups: list[QtWidgets.QGroupBox] = []
        self.live_running = False
        self._live_reconfigure_pending = False
        self.frame_counter = 0
        self.live_fps = 0.0
        self._last_live_time = time.monotonic()
        self._sequence_thread: QtCore.QThread | None = None
        self._sequence_worker: SequenceWorker | None = None

        self._build_ui()
        self._connect_signals()

        self.live_timer = QtCore.QTimer(self)
        self.live_timer.setInterval(80)
        self.live_timer.timeout.connect(self._update_live_frame)

        self.status_timer = QtCore.QTimer(self)
        self.status_timer.setInterval(1500)
        self.status_timer.timeout.connect(self._refresh_status)

        if self.mock:
            self.dll_edit.setText("MOCK BACKEND")
        elif self.dll_path:
            self.dll_edit.setText(self.dll_path)
        self._set_connected_ui(False)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self._apply_professional_theme()

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        header = QtWidgets.QFrame(self)
        header.setObjectName("HeaderFrame")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(20, 12, 20, 12)
        header_layout.setSpacing(16)

        title_box = QtWidgets.QWidget(header)
        title_layout = QtWidgets.QVBoxLayout(title_box)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(3)
        title = QtWidgets.QLabel("Andor iXon 897")
        title.setObjectName("AppTitle")
        subtitle = QtWidgets.QLabel("EMCCD Scientific Acquisition  ·  SDK2  ·  Photon Counting  ·  HDF5/NPZ")
        subtitle.setObjectName("AppSubtitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header_layout.addWidget(title_box, 1)

        # Vertical separator
        sep = QtWidgets.QFrame(header)
        sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        sep.setStyleSheet("QFrame { color: #1a2c44; margin: 6px 0; }")
        header_layout.addWidget(sep)

        self.toggle_controls_button = QtWidgets.QPushButton("⊟  Hide Controls")
        self.toggle_controls_button.setToolTip("Collapse or restore the left control panel to maximise the detector view.")
        self.clean_view_button = QtWidgets.QPushButton("⬛  Full Screen View")
        self.clean_view_button.setToolTip("Hide secondary inspectors, keeping only the detector image and essential status.")
        header_layout.addWidget(self.toggle_controls_button)
        header_layout.addWidget(self.clean_view_button)

        self.connection_tile = self._status_tile("CAMERA", "Disconnected", "Neutral")
        self.temperature_tile = self._status_tile("TEMPERATURE", "—", "Neutral")
        self.acquisition_tile = self._status_tile("ACQUISITION", "Idle", "Neutral")
        self.shutter_tile = self._status_tile("SHUTTER", "AUTO · CLOSED", "Neutral")
        self.fps_tile = self._status_tile("LIVE FPS", "—", "Neutral")
        self.session_tile = self._status_tile("SESSION", "No session", "Neutral")
        for tile in [self.connection_tile, self.temperature_tile, self.acquisition_tile, self.shutter_tile, self.fps_tile, self.session_tile]:
            header_layout.addWidget(tile)
        root.addWidget(header, 0)

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.main_splitter.setChildrenCollapsible(False)
        root.addWidget(self.main_splitter, 1)

        self.left_panel = QtWidgets.QFrame(self)
        self.left_panel.setObjectName("SidePanel")
        self.left_panel.setMinimumWidth(330)
        self.left_panel.setMaximumWidth(430)
        self.left_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        left_layout = QtWidgets.QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(8, 10, 8, 10)
        left_layout.setSpacing(6)

        control_header = QtWidgets.QFrame(self.left_panel)
        control_header.setObjectName("ControlHeader")
        ch = QtWidgets.QVBoxLayout(control_header)
        ch.setContentsMargins(4, 0, 4, 4)
        ch.setSpacing(6)
        section_title = QtWidgets.QLabel("INSTRUMENT CONTROL")
        section_title.setObjectName("PanelHeader")
        ch.addWidget(section_title)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(6)
        mode_row.addWidget(QtWidgets.QLabel("Mode:"), 0)
        self.control_mode_combo = QtWidgets.QComboBox()
        self.control_mode_combo.addItems(["Basic", "Advanced"])
        self.control_mode_combo.setCurrentText("Basic")
        self.control_mode_combo.setToolTip("Basic hides rarely used controls. Advanced shows all camera parameters.")
        mode_row.addWidget(self.control_mode_combo, 1)
        ch.addLayout(mode_row)

        quick = QtWidgets.QGridLayout()
        quick.setContentsMargins(0, 0, 0, 0)
        quick.setHorizontalSpacing(6)
        quick.setVerticalSpacing(6)
        self.quick_connect_button = QtWidgets.QPushButton("Connect")
        self.quick_single_button = QtWidgets.QPushButton("Single")
        self.quick_live_button = QtWidgets.QPushButton("Live")
        self.quick_stop_button = QtWidgets.QPushButton("Stop")
        self.quick_save_button = QtWidgets.QPushButton("Save")
        for btn in [self.quick_connect_button, self.quick_single_button, self.quick_live_button, self.quick_stop_button, self.quick_save_button]:
            btn.setMinimumHeight(28)
            btn.setMaximumHeight(30)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        quick.addWidget(self.quick_connect_button, 0, 0)
        quick.addWidget(self.quick_single_button, 0, 1)
        quick.addWidget(self.quick_live_button, 1, 0)
        quick.addWidget(self.quick_stop_button, 1, 1)
        quick.addWidget(self.quick_save_button, 2, 0, 1, 2)
        ch.addLayout(quick)
        left_layout.addWidget(control_header, 0)

        self.tabs = QtWidgets.QTabWidget(self)
        self.tabs.setObjectName("ControlTabs")
        self.tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.West)
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setMinimumWidth(0)
        self.tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        left_layout.addWidget(self.tabs, 1)
        self.main_splitter.addWidget(self.left_panel)

        self._build_session_tab()
        self._build_camera_tab()
        self._build_acquisition_tab()
        self._build_calibration_tab()
        self._build_units_tab()
        self._build_photon_counting_tab()
        self._apply_compact_control_defaults()
        self._apply_control_mode()

        self.center_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        self.center_splitter.setChildrenCollapsible(False)
        image_group = QtWidgets.QGroupBox("Detector Image")
        image_group.setObjectName("ImageGroup")
        image_layout = QtWidgets.QVBoxLayout(image_group)
        image_layout.setContentsMargins(8, 18, 8, 8)
        image_layout.setSpacing(6)

        image_toolbar = QtWidgets.QFrame(self)
        image_toolbar.setObjectName("ImageToolbar")
        toolbar_layout = QtWidgets.QHBoxLayout(image_toolbar)
        toolbar_layout.setContentsMargins(10, 4, 10, 4)
        toolbar_layout.setSpacing(16)
        self.display_mode_label = QtWidgets.QLabel("Display: raw ADU")
        self.display_mode_label.setObjectName("SoftLabel")
        self.frame_shape_label = QtWidgets.QLabel("Frame: —")
        self.frame_shape_label.setObjectName("SoftLabel")
        self.saturation_label = QtWidgets.QLabel("Saturation: —")
        self.saturation_label.setObjectName("SoftLabel")
        # Thin coloured accent line under toolbar acts as a live indicator
        self._live_indicator = QtWidgets.QLabel("  ● LIVE  ")
        self._live_indicator.setStyleSheet(
            "color: #00c8d4; font-weight: 700; font-size: 8pt; "
            "letter-spacing: 1px; background: transparent;"
        )
        self._live_indicator.setVisible(False)
        toolbar_layout.addWidget(self.display_mode_label)
        toolbar_layout.addWidget(self.frame_shape_label)
        self.display_scale_label = QtWidgets.QLabel("Scale: Auto")
        self.display_scale_label.setObjectName("SoftLabel")
        self.spectroscopy_status_label = QtWidgets.QLabel("Spectroscopy: off")
        self.spectroscopy_status_label.setObjectName("SoftLabel")
        self.fit_view_button = QtWidgets.QPushButton("Fit")
        self.reset_zoom_button = QtWidgets.QPushButton("Reset Zoom")
        self.one_to_one_button = QtWidgets.QPushButton("1:1")
        for btn in [self.fit_view_button, self.reset_zoom_button, self.one_to_one_button]:
            btn.setMaximumWidth(90)
            btn.setToolTip("Image zoom/pan control")
        toolbar_layout.addWidget(self.saturation_label)
        toolbar_layout.addWidget(self.display_scale_label)
        toolbar_layout.addWidget(self.spectroscopy_status_label)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self.fit_view_button)
        toolbar_layout.addWidget(self.reset_zoom_button)
        toolbar_layout.addWidget(self.one_to_one_button)
        toolbar_layout.addWidget(self._live_indicator)
        image_layout.addWidget(image_toolbar, 0)

        self.image_view = pg.ImageView()
        self.image_view.setObjectName("DetectorImageView")
        try:
            self.image_view.view.setMouseEnabled(x=True, y=True)
            self.image_view.view.setAspectLocked(True)
        except Exception:
            pass
        image_layout.addWidget(self.image_view, 1)
        self.center_splitter.addWidget(image_group)

        self.inspector_tabs = QtWidgets.QTabWidget(self)
        self.inspector_tabs.setObjectName("InspectorTabs")
        self.hist_plot = pg.PlotWidget(title="Pixel Intensity Distribution")
        self.hist_plot.showGrid(x=True, y=True, alpha=0.15)
        self.hist_plot.setLabel("bottom", "Signal", units="ADU")
        self.hist_plot.setLabel("left", "Pixel count")
        self.hist_plot.getAxis("bottom").setTextPen(pg.mkPen("#607590"))
        self.hist_plot.getAxis("left").setTextPen(pg.mkPen("#607590"))

        self.stats_text = QtWidgets.QPlainTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setObjectName("MonospaceBox")
        self.stats_text.setMaximumBlockCount(5000)

        self.spectroscopy_plot = pg.PlotWidget(title="Spectroscopy ROI — vertical sum")
        self.spectroscopy_plot.showGrid(x=True, y=True, alpha=0.15)
        self.spectroscopy_plot.setLabel("bottom", "Horizontal pixel x")
        self.spectroscopy_plot.setLabel("left", "Vertical sum")
        self.spectroscopy_plot.getAxis("bottom").setTextPen(pg.mkPen("#607590"))
        self.spectroscopy_plot.getAxis("left").setTextPen(pg.mkPen("#607590"))

        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setObjectName("MonospaceBox")
        self.log_text.setMaximumBlockCount(5000)

        self.inspector_tabs.addTab(self.hist_plot, "Histogram")
        self.inspector_tabs.addTab(self.spectroscopy_plot, "Spectroscopy")
        self.inspector_tabs.addTab(self.stats_text, "Statistics")
        self.inspector_tabs.addTab(self.log_text, "Session Log")
        self.center_splitter.addWidget(self.inspector_tabs)
        self.main_splitter.addWidget(self.center_splitter)

        self.statusBar().setObjectName("MainStatusBar")
        self.statusBar().showMessage("Ready")
        self._update_header_status()

    def _apply_professional_theme(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setStyle("Fusion")
            app.setStyleSheet(self._professional_stylesheet())
        pg.setConfigOptions(
            background="#05080f",
            foreground="#c8d6e8",
            antialias=True,
            imageAxisOrder="row-major",
        )

    @staticmethod
    def _professional_stylesheet() -> str:
        # ── Colour palette — deep navy scientific instrument theme ──────────
        # Background layers:  #05080f → #0a0f1e → #0d1526 → #111d30
        # Accent blue:  #1a6fff (active), #2d7fff (hover), #0d4ecc (pressed)
        # Accent cyan:  #00c8d4 (live / good indicators)
        # Amber:        #f0a500 (warnings)
        # Red:          #e03050 (errors / bad)
        # Text:         #e8edf5 (primary) / #8fa3c0 (secondary) / #4e6280 (muted)
        return """
        /* ── Global reset ───────────────────────────────────────────── */
        QWidget {
            background-color: #0a0f1e;
            color: #e8edf5;
            font-family: "Segoe UI", "Inter", "SF Pro Display", Arial, sans-serif;
            font-size: 10pt;
        }
        QMainWindow, QDialog { background-color: #05080f; }

        /* ── Header & side-panel shells ─────────────────────────────── */
        #HeaderFrame {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #0e1828, stop:1 #09101e);
            border: 1px solid #1a2c44;
            border-radius: 12px;
        }
        #SidePanel {
            background-color: #0a0f1e;
            border: 1px solid #1a2c44;
            border-radius: 10px;
        }

        /* ── Typography ─────────────────────────────────────────────── */
        #AppTitle {
            font-size: 17pt;
            font-weight: 700;
            color: #e8edf5;
            letter-spacing: 0.5px;
            background: transparent;
        }
        #AppSubtitle {
            color: #5a7a9e;
            font-size: 9pt;
            letter-spacing: 0.8px;
            background: transparent;
        }
        #SoftLabel {
            color: #607590;
            font-size: 9pt;
            background: transparent;
        }
        #PanelHeader {
            color: #1a6fff;
            font-size: 8pt;
            font-weight: 700;
            letter-spacing: 2px;
            background: transparent;
        }
        QLabel { background: transparent; }

        /* ── Status tiles ───────────────────────────────────────────── */
        QFrame[cardClass="status"] {
            background-color: #0d1526;
            border: 1px solid #1e3050;
            border-radius: 8px;
            min-width: 100px;
        }
        QLabel[role="statusTitle"] {
            color: #4e6a8c;
            font-size: 7.5pt;
            font-weight: 700;
            letter-spacing: 1.5px;
            background: transparent;
        }
        QLabel[role="statusValue"] {
            color: #c8d8f0;
            font-size: 10pt;
            font-weight: 600;
            background: transparent;
        }

        /* ── Group boxes ────────────────────────────────────────────── */
        QGroupBox {
            background-color: #0d1526;
            border: 1px solid #1a2c44;
            border-radius: 8px;
            margin-top: 14px;
            padding: 14px 10px 10px 10px;
            font-size: 9.5pt;
            font-weight: 600;
            color: #c8d8f0;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            top: -1px;
            padding: 1px 8px;
            color: #1a6fff;
            font-size: 8.5pt;
            font-weight: 700;
            letter-spacing: 0.8px;
            background-color: #0d1526;
            border-radius: 4px;
        }

        /* ── Input controls ─────────────────────────────────────────── */
        QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #070c18;
            border: 1px solid #1e3050;
            border-radius: 5px;
            padding: 5px 8px;
            selection-background-color: #1a6fff;
            color: #d8e4f5;
            font-size: 10pt;
            min-height: 26px;
        }
        QLineEdit:focus, QPlainTextEdit:focus,
        QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 1px solid #1a6fff;
            background-color: #080d1c;
        }
        QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
            border-color: #2a4870;
        }
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
            background-color: #111d30;
            border: none;
            width: 16px;
        }
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #1a2c44;
        }
        QComboBox::drop-down {
            border: none;
            background-color: #111d30;
            width: 22px;
            border-top-right-radius: 5px;
            border-bottom-right-radius: 5px;
        }
        QComboBox QAbstractItemView {
            background-color: #0d1526;
            border: 1px solid #1a6fff;
            border-radius: 5px;
            selection-background-color: #1a3a6e;
            color: #d8e4f5;
            outline: none;
        }

        /* ── Buttons (default) ──────────────────────────────────────── */
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #16253d, stop:1 #0f1d30);
            border: 1px solid #2a4060;
            border-radius: 6px;
            padding: 6px 12px;
            color: #c8d8f0;
            font-weight: 600;
            font-size: 9.5pt;
            min-height: 28px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1e3558, stop:1 #162a48);
            border-color: #1a6fff;
            color: #e8f0ff;
        }
        QPushButton:pressed {
            background: #0d2a5a;
            border-color: #1a6fff;
        }
        QPushButton:disabled {
            color: #2e4060;
            background: #0a0f1e;
            border-color: #141e2e;
        }

        /* ── Checkboxes / Radio buttons ─────────────────────────────── */
        QCheckBox {
            spacing: 8px;
            background: transparent;
            color: #b8cce0;
            font-size: 9.5pt;
        }
        QCheckBox::indicator {
            width: 14px;
            height: 14px;
            border-radius: 3px;
            border: 1px solid #2a4060;
            background: #070c18;
        }
        QCheckBox::indicator:hover { border-color: #1a6fff; }
        QCheckBox::indicator:checked {
            background: #1a6fff;
            border-color: #2d7fff;
        }
        QCheckBox::indicator:checked:hover { background: #2d7fff; }
        QRadioButton {
            background: transparent;
            color: #b8cce0;
            spacing: 8px;
        }
        QRadioButton::indicator {
            width: 13px;
            height: 13px;
            border-radius: 7px;
            border: 1px solid #2a4060;
            background: #070c18;
        }
        QRadioButton::indicator:checked {
            background: #1a6fff;
            border-color: #2d7fff;
        }

        /* ── Tab widget ─────────────────────────────────────────────── */
        QTabWidget::pane {
            border: 1px solid #1a2c44;
            border-top-left-radius: 0px;
            border-top-right-radius: 8px;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
            background: #0d1526;
            top: -1px;
        }
        QTabBar { background: transparent; }
        QTabBar::tab {
            background: #070c18;
            color: #4e6a8c;
            padding: 6px 9px;
            border: 1px solid #1a2c44;
            border-bottom: none;
            border-top-left-radius: 7px;
            border-top-right-radius: 7px;
            margin-right: 2px;
            font-size: 9pt;
            font-weight: 600;
            min-width: 46px;
        }
        QTabBar::tab:selected {
            background: #0d1526;
            color: #d8eaff;
            border-color: #1a6fff;
            border-bottom-color: #0d1526;
        }
        QTabBar::tab:hover:!selected { color: #90b8e8; background: #0a1220; }

        /* ── Scroll bars ────────────────────────────────────────────── */
        QScrollArea { border: none; background: transparent; }
        QScrollBar:vertical {
            background: #070c18;
            width: 8px;
            margin: 0;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: #1e3050;
            border-radius: 4px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover { background: #2a4870; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar:horizontal {
            background: transparent;
            height: 0px;
            max-height: 0px;
        }
        QScrollBar::handle:horizontal {
            background: transparent;
            min-width: 0px;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; height: 0; }

        /* ── Splitter ───────────────────────────────────────────────── */
        QSplitter::handle { background-color: #111d30; }
        QSplitter::handle:horizontal { width: 4px; }
        QSplitter::handle:vertical   { height: 4px; }
        QSplitter::handle:hover      { background-color: #1a6fff; }

        /* ── Progress bar ───────────────────────────────────────────── */
        QProgressBar {
            border: 1px solid #1e3050;
            border-radius: 5px;
            background: #070c18;
            text-align: center;
            color: #8fa8d0;
            font-size: 8.5pt;
            max-height: 16px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #1448aa, stop:1 #1a6fff);
            border-radius: 5px;
        }

        /* ── Image panel ────────────────────────────────────────────── */
        #ImageToolbar {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #0d1526, stop:1 #09101e);
            border: 1px solid #1a2c44;
            border-radius: 6px;
        }
        #ImageGroup {
            background-color: #08101e;
            border: 1px solid #1a2c44;
            border-radius: 8px;
        }

        /* ── Monospace log / stats panels ───────────────────────────── */
        #MonospaceBox {
            font-family: "Cascadia Mono", "JetBrains Mono", Consolas, monospace;
            font-size: 8.5pt;
            background-color: #05080f;
            color: #90b8d8;
            border: 1px solid #1a2c44;
            border-radius: 5px;
            padding: 4px;
        }

        /* ── Status bar ─────────────────────────────────────────────── */
        #MainStatusBar {
            background-color: #070c18;
            border-top: 1px solid #1a2c44;
            color: #607590;
            font-size: 9pt;
        }

        /* ── Tooltips ───────────────────────────────────────────────── */
        QToolTip {
            background-color: #0d1a2e;
            color: #c8d8f0;
            border: 1px solid #1a6fff;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 9pt;
        }

        /* ── Message boxes ──────────────────────────────────────────── */
        QMessageBox { background-color: #0a0f1e; }
        QMessageBox QLabel { color: #c8d8f0; }
        """

    def _status_tile(self, title: str, value: str, state: str = "Neutral") -> QtWidgets.QFrame:
        tile = QtWidgets.QFrame(self)
        tile.setProperty("cardClass", "status")
        layout = QtWidgets.QVBoxLayout(tile)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(3)
        title_label = QtWidgets.QLabel(title)
        title_label.setProperty("role", "statusTitle")
        value_label = QtWidgets.QLabel(value)
        value_label.setProperty("role", "statusValue")
        value_label.setWordWrap(True)
        tile.value_label = value_label  # type: ignore[attr-defined]
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        self._set_tile_state(tile, state)
        return tile

    @staticmethod
    def _set_tile_state(tile: QtWidgets.QFrame, state: str) -> None:
        # border_color, value_text_color
        colors = {
            "Neutral": ("#1e3050", "#8fa8c8"),
            "Good":    ("#00c8d4", "#80f0f8"),
            "Warning": ("#f0a500", "#ffd060"),
            "Bad":     ("#e03050", "#ff8090"),
            "Info":    ("#1a6fff", "#90c0ff"),
        }
        border, value = colors.get(state, colors["Neutral"])
        tile.setStyleSheet(
            f"QFrame {{ background-color: #0d1526; border: 1px solid {border}; border-radius: 8px; }}"
            f"QLabel[role='statusTitle'] {{ color: #4e6a8c; font-size: 7.5pt; font-weight: 700; letter-spacing: 1.5px; background: transparent; }}"
            f"QLabel[role='statusValue'] {{ color: {value}; font-size: 10pt; font-weight: 600; background: transparent; }}"
        )

    def _update_header_status(self) -> None:
        if not hasattr(self, "connection_tile"):
            return
        connected = self.camera is not None
        self.connection_tile.value_label.setText("Connected" if connected else "Disconnected")  # type: ignore[attr-defined]
        self._set_tile_state(self.connection_tile, "Good" if connected else "Neutral")
        if self.session is not None:
            self.session_tile.value_label.setText(self.session.root.name)  # type: ignore[attr-defined]
            self._set_tile_state(self.session_tile, "Info")
        else:
            self.session_tile.value_label.setText("No session")  # type: ignore[attr-defined]
            self._set_tile_state(self.session_tile, "Neutral")
        self.acquisition_tile.value_label.setText("Live" if self.live_running else "Idle")  # type: ignore[attr-defined]
        self._set_tile_state(self.acquisition_tile, "Good" if self.live_running else "Neutral")
        if hasattr(self, "fps_tile"):
            fps_text = f"{self.live_fps:.1f} fps" if self.live_running else "—"
            self.fps_tile.value_label.setText(fps_text)  # type: ignore[attr-defined]
            self._set_tile_state(self.fps_tile, "Good" if self.live_running else "Neutral")

    # ── Shared helper: wraps a widget in a scrollable container ──────────
    @staticmethod
    def _scrollable_tab(content_widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        content_widget.setMinimumWidth(0)
        content_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        scroll.setWidget(content_widget)
        outer.addWidget(scroll)
        return tab

    @staticmethod
    def _make_content_widget() -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        return content, layout

    @staticmethod
    def _grid(parent: QtWidgets.QGroupBox | None = None) -> QtWidgets.QGridLayout:
        """Return a QGridLayout with consistent label/control column stretch."""
        gl = QtWidgets.QGridLayout(parent) if parent else QtWidgets.QGridLayout()
        gl.setContentsMargins(8, 5, 8, 8)
        gl.setHorizontalSpacing(6)
        gl.setVerticalSpacing(5)
        gl.setColumnStretch(0, 0)   # label column — fit to content
        gl.setColumnStretch(1, 1)   # control column — stretches
        return gl

    def _register_advanced(self, *widgets: QtWidgets.QWidget) -> None:
        for widget in widgets:
            if widget not in self._advanced_control_widgets:
                self._advanced_control_widgets.append(widget)

    def _apply_compact_control_defaults(self) -> None:
        """Keep the Instrument Control panel narrow and ergonomic.

        The left control panel must never force a horizontal scrollbar.  This
        pass reduces long controls, lets text elide/wrap where possible, and
        gives spin boxes/combos sane maximum widths inside the side panel.
        """
        for label in self.left_panel.findChildren(QtWidgets.QLabel):
            label.setWordWrap(True)
            label.setMinimumWidth(0)
        for button in self.left_panel.findChildren(QtWidgets.QPushButton):
            button.setMinimumWidth(0)
            button.setMaximumHeight(34)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
        for spin in self.left_panel.findChildren((QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            spin.setMinimumWidth(70)
            spin.setMaximumWidth(145)
            spin.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
        for combo in self.left_panel.findChildren(QtWidgets.QComboBox):
            combo.setMinimumWidth(80)
            combo.setMaximumWidth(180)
            combo.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
        for edit in self.left_panel.findChildren(QtWidgets.QLineEdit):
            edit.setMinimumWidth(0)
            edit.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
        for group in self.left_panel.findChildren(QtWidgets.QGroupBox):
            group.setMinimumWidth(0)
            group.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Maximum,
            )

    def _apply_control_mode(self) -> None:
        advanced = getattr(self, "control_mode_combo", None) is not None and self.control_mode_combo.currentText() == "Advanced"
        self.compact_control_mode = not advanced
        for widget in getattr(self, "_advanced_control_widgets", []):
            widget.setVisible(advanced)
        if hasattr(self, "tabs"):
            for idx in range(self.tabs.count()):
                name = self.tabs.tabText(idx)
                if name in {"Calibration", "Photon Counting"}:
                    self.tabs.setTabVisible(idx, advanced)
        if hasattr(self, "control_mode_combo"):
            self.control_mode_combo.setToolTip(
                "Advanced controls are visible." if advanced else
                "Basic mode: advanced camera/calibration/photon-counting controls are hidden to keep the panel compact."
            )

    def _build_session_tab(self) -> None:
        content, layout = self._make_content_widget()

        grp = QtWidgets.QGroupBox("Session")
        gl = self._grid(grp)

        self.base_dir_edit = QtWidgets.QLineEdit(str(self.output_base))
        self.browse_base_button = QtWidgets.QPushButton("Browse…")
        self.browse_base_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.experiment_edit = QtWidgets.QLineEdit("Andor_iXon897_test")
        self.sample_edit = QtWidgets.QLineEdit("")
        self.operator_edit = QtWidgets.QLineEdit("")
        self.notes_edit = QtWidgets.QPlainTextEdit()
        self.notes_edit.setPlaceholderText(
            "Session notes, alignment conditions, optics, filters, sample…")
        self.notes_edit.setMinimumHeight(80)
        self.notes_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding)

        # Row 0: output dir + browse button side-by-side
        dir_row = QtWidgets.QHBoxLayout()
        dir_row.setSpacing(6)
        dir_row.addWidget(self.base_dir_edit, 1)
        dir_row.addWidget(self.browse_base_button, 0)
        gl.addWidget(QtWidgets.QLabel("Output dir:"), 0, 0)
        gl.addLayout(dir_row, 0, 1)

        gl.addWidget(QtWidgets.QLabel("Experiment:"), 1, 0)
        gl.addWidget(self.experiment_edit, 1, 1)
        gl.addWidget(QtWidgets.QLabel("Sample:"), 2, 0)
        gl.addWidget(self.sample_edit, 2, 1)
        gl.addWidget(QtWidgets.QLabel("Operator:"), 3, 0)
        gl.addWidget(self.operator_edit, 3, 1)
        gl.addWidget(QtWidgets.QLabel("Notes:"), 4, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        gl.addWidget(self.notes_edit, 4, 1)
        gl.setRowStretch(4, 1)
        layout.addWidget(grp, 1)

        self.create_session_button = QtWidgets.QPushButton("Create Session")
        self.create_session_button.setMinimumHeight(34)
        layout.addWidget(self.create_session_button)

        self.session_label = QtWidgets.QLabel(
            "No session active. Files will be saved in the output directory.")
        self.session_label.setWordWrap(True)
        self.session_label.setStyleSheet("color: #607590; font-size: 9pt;")
        layout.addWidget(self.session_label)

        self.tabs.addTab(self._scrollable_tab(content), "Session")

    def _build_camera_tab(self) -> None:
        content, layout = self._make_content_widget()

        # ── Connection ──────────────────────────────────────────────────────
        conn = QtWidgets.QGroupBox("Connection")
        gl = self._grid(conn)
        self.dll_edit = QtWidgets.QLineEdit()
        self.dll_edit.setPlaceholderText(r"C:\Path\To\atmcd64d.dll")
        self.browse_dll_button = QtWidgets.QPushButton("Browse…")
        self.browse_dll_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        dll_row = QtWidgets.QHBoxLayout()
        dll_row.setSpacing(6)
        dll_row.addWidget(self.dll_edit, 1)
        dll_row.addWidget(self.browse_dll_button, 0)
        gl.addWidget(QtWidgets.QLabel("SDK DLL:"), 0, 0)
        gl.addLayout(dll_row, 0, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(6)
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.disconnect_button = QtWidgets.QPushButton("Disconnect")
        btn_row.addWidget(self.connect_button, 1)
        btn_row.addWidget(self.disconnect_button, 1)
        gl.addWidget(QtWidgets.QLabel(""), 1, 0)
        gl.addLayout(btn_row, 1, 1)

        self.identity_label = QtWidgets.QLabel("Not connected")
        self.identity_label.setWordWrap(True)
        self.identity_label.setStyleSheet("color: #607590; font-size: 9pt;")
        gl.addWidget(self.identity_label, 2, 0, 1, 2)
        layout.addWidget(conn)

        # ── Temperature ─────────────────────────────────────────────────────
        temp = QtWidgets.QGroupBox("Temperature")
        tl = self._grid(temp)
        self.temp_setpoint_spin = QtWidgets.QSpinBox()
        self.temp_setpoint_spin.setRange(-120, 20)
        self.temp_setpoint_spin.setValue(-70)
        self.temp_setpoint_spin.setSuffix(" °C")
        self.fan_combo = self._enum_combo(FanMode, FanMode.FULL)
        self.cooler_check = QtWidgets.QCheckBox("Cooler ON")
        self.cooler_check.setChecked(True)
        self.keep_cooler_check = QtWidgets.QCheckBox("Keep cooler ON at SDK shutdown")
        self.apply_temp_button = QtWidgets.QPushButton("Apply")
        self.apply_temp_button.setMinimumHeight(30)
        self.temp_label = QtWidgets.QLabel("Temperature: —")
        self.temp_label.setWordWrap(True)
        self.temp_label.setStyleSheet("color: #607590; font-size: 9pt;")
        tl.addWidget(QtWidgets.QLabel("Setpoint:"), 0, 0)
        tl.addWidget(self.temp_setpoint_spin, 0, 1)
        tl.addWidget(QtWidgets.QLabel("Fan:"), 1, 0)
        tl.addWidget(self.fan_combo, 1, 1)
        tl.addWidget(self.cooler_check, 2, 0, 1, 2)
        tl.addWidget(self.keep_cooler_check, 3, 0, 1, 2)
        tl.addWidget(self.apply_temp_button, 4, 0, 1, 2)
        tl.addWidget(self.temp_label, 5, 0, 1, 2)
        layout.addWidget(temp)

        # ── Shutter / Trigger ───────────────────────────────────────────────
        shutter = QtWidgets.QGroupBox("Shutter")
        sl = self._grid(shutter)
        sl.setColumnStretch(2, 1)  # third column also stretches
        self.auto_shutter_check = QtWidgets.QCheckBox("Auto shutter")
        self.auto_shutter_check.setChecked(True)
        self.auto_shutter_check.setToolTip("Default: open shutter when detection starts, close it when detection stops.")
        self.internal_shutter_combo = self._enum_combo(ShutterMode, ShutterMode.FULLY_AUTO)
        self.external_shutter_combo = self._enum_combo(ShutterMode, ShutterMode.FULLY_AUTO)
        self.ttl_high_check = QtWidgets.QCheckBox("TTL high opens external shutter")
        self.opening_ms_spin = QtWidgets.QSpinBox()
        self.opening_ms_spin.setRange(0, 10000)
        self.opening_ms_spin.setSuffix(" ms")
        self.closing_ms_spin = QtWidgets.QSpinBox()
        self.closing_ms_spin.setRange(0, 10000)
        self.closing_ms_spin.setSuffix(" ms")
        self.trigger_combo = self._enum_combo(TriggerMode, TriggerMode.INTERNAL)
        self.apply_shutter_button = QtWidgets.QPushButton("Apply")
        self.apply_shutter_button.setMinimumHeight(30)
        self.force_open_button = QtWidgets.QPushButton("Open")
        self.force_closed_button = QtWidgets.QPushButton("Close")
        self.shutter_status_label = QtWidgets.QLabel("Shutter: AUTO — commanded CLOSED")
        self.shutter_status_label.setWordWrap(True)
        self.shutter_status_label.setStyleSheet("color: #8fa3c0; font-size: 9pt; padding: 4px 6px;")
        sl.addWidget(self.auto_shutter_check, 0, 0, 1, 3)
        sl.addWidget(QtWidgets.QLabel("Internal:"), 1, 0)
        sl.addWidget(self.internal_shutter_combo, 1, 1, 1, 2)
        sl.addWidget(QtWidgets.QLabel("External:"), 2, 0)
        sl.addWidget(self.external_shutter_combo, 2, 1, 1, 2)
        sl.addWidget(self.ttl_high_check, 3, 0, 1, 3)
        sl.addWidget(QtWidgets.QLabel("Open / close delay:"), 4, 0)
        sl.addWidget(self.opening_ms_spin, 4, 1)
        sl.addWidget(self.closing_ms_spin, 4, 2)
        sl.addWidget(QtWidgets.QLabel("Trigger:"), 5, 0)
        sl.addWidget(self.trigger_combo, 5, 1, 1, 2)
        sl.addWidget(self.apply_shutter_button, 6, 0, 1, 3)
        force_row = QtWidgets.QHBoxLayout()
        force_row.setSpacing(6)
        force_row.addWidget(self.force_open_button, 1)
        force_row.addWidget(self.force_closed_button, 1)
        sl.addLayout(force_row, 7, 0, 1, 3)
        sl.addWidget(self.shutter_status_label, 8, 0, 1, 3)
        layout.addWidget(shutter)

        # ── Exposure / Readout ──────────────────────────────────────────────
        acq = QtWidgets.QGroupBox("Exposure / Gain / Readout")
        al = self._grid(acq)
        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setRange(1e-6, 3600.0)
        self.exposure_spin.setDecimals(6)
        self.exposure_spin.setValue(0.03)
        self.exposure_spin.setSingleStep(0.01)
        self.exposure_spin.setSuffix(" s")
        self.frame_transfer_check = QtWidgets.QCheckBox("Frame transfer mode")
        self.em_gain_mode_combo = self._enum_combo(EMGainMode, EMGainMode.REAL_GAIN)
        self.em_gain_spin = QtWidgets.QSpinBox()
        self.em_gain_spin.setRange(1, 1000)
        self.em_gain_spin.setValue(1)
        self.em_advanced_check = QtWidgets.QCheckBox("EM advanced (>300×)")
        self.allow_gain_check = QtWidgets.QCheckBox("Allow EM gain before temperature stable")
        self.preamp_combo = QtWidgets.QComboBox()
        self.hs_speed_combo = QtWidgets.QComboBox()
        self.vs_speed_combo = QtWidgets.QComboBox()
        self.baseline_clamp_check = QtWidgets.QCheckBox("Baseline clamp")
        self.baseline_clamp_check.setChecked(True)
        self.apply_settings_button = QtWidgets.QPushButton("Apply")
        self.apply_settings_button.setMinimumHeight(30)
        self.gain_status_label = QtWidgets.QLabel("Gain request: 1 | SDK applied: —")
        self.gain_status_label.setWordWrap(True)
        self.gain_status_label.setStyleSheet("color: #8fa3c0; font-size: 9pt; padding: 4px 6px;")
        al.addWidget(QtWidgets.QLabel("Exposure:"), 0, 0)
        al.addWidget(self.exposure_spin, 0, 1)
        al.addWidget(self.frame_transfer_check, 1, 0, 1, 2)
        al.addWidget(QtWidgets.QLabel("Gain mode:"), 2, 0)
        al.addWidget(self.em_gain_mode_combo, 2, 1)
        al.addWidget(QtWidgets.QLabel("EM gain:"), 3, 0)
        al.addWidget(self.em_gain_spin, 3, 1)
        al.addWidget(self.em_advanced_check, 4, 0, 1, 2)
        al.addWidget(self.allow_gain_check, 5, 0, 1, 2)
        al.addWidget(QtWidgets.QLabel("Preamp:"), 6, 0)
        al.addWidget(self.preamp_combo, 6, 1)
        al.addWidget(QtWidgets.QLabel("HS speed:"), 7, 0)
        al.addWidget(self.hs_speed_combo, 7, 1)
        al.addWidget(QtWidgets.QLabel("VS speed:"), 8, 0)
        al.addWidget(self.vs_speed_combo, 8, 1)
        al.addWidget(self.baseline_clamp_check, 9, 0, 1, 2)
        al.addWidget(self.apply_settings_button, 10, 0, 1, 2)
        al.addWidget(self.gain_status_label, 11, 0, 1, 2)
        layout.addWidget(acq)
        self._register_advanced(acq)

        # ── Camera ROI / Binning ────────────────────────────────────────────
        roi = QtWidgets.QGroupBox("Camera ROI")
        rl = self._grid(roi)
        rl.setColumnStretch(1, 1)
        rl.setColumnStretch(2, 1)
        self.hstart_spin = QtWidgets.QSpinBox(); self.hstart_spin.setRange(1, 512); self.hstart_spin.setValue(1)
        self.hend_spin = QtWidgets.QSpinBox(); self.hend_spin.setRange(1, 512); self.hend_spin.setValue(512)
        self.vstart_spin = QtWidgets.QSpinBox(); self.vstart_spin.setRange(1, 512); self.vstart_spin.setValue(1)
        self.vend_spin = QtWidgets.QSpinBox(); self.vend_spin.setRange(1, 512); self.vend_spin.setValue(512)
        self.hbin_spin = QtWidgets.QSpinBox(); self.hbin_spin.setRange(1, 16); self.hbin_spin.setValue(1)
        self.vbin_spin = QtWidgets.QSpinBox(); self.vbin_spin.setRange(1, 16); self.vbin_spin.setValue(1)
        self.full_roi_button = QtWidgets.QPushButton("Full")
        self.center_roi_button = QtWidgets.QPushButton("128×128")
        rl.addWidget(QtWidgets.QLabel("H start / end:"), 0, 0)
        rl.addWidget(self.hstart_spin, 0, 1); rl.addWidget(self.hend_spin, 0, 2)
        rl.addWidget(QtWidgets.QLabel("V start / end:"), 1, 0)
        rl.addWidget(self.vstart_spin, 1, 1); rl.addWidget(self.vend_spin, 1, 2)
        rl.addWidget(QtWidgets.QLabel("H / V bin:"), 2, 0)
        rl.addWidget(self.hbin_spin, 2, 1); rl.addWidget(self.vbin_spin, 2, 2)
        roi_btn_row = QtWidgets.QHBoxLayout()
        roi_btn_row.setSpacing(6)
        roi_btn_row.addWidget(self.full_roi_button, 1)
        roi_btn_row.addWidget(self.center_roi_button, 1)
        rl.addLayout(roi_btn_row, 3, 0, 1, 3)
        layout.addWidget(roi)
        self._register_advanced(roi)

        # ── SDK status ──────────────────────────────────────────────────────
        self.camera_status_label = QtWidgets.QLabel("SDK status: —")
        self.camera_status_label.setWordWrap(True)
        self.camera_status_label.setStyleSheet(
            "color: #607590; font-size: 9pt; padding: 4px 6px;")
        layout.addWidget(self.camera_status_label)

        self.tabs.addTab(self._scrollable_tab(content), "Camera")

    def _build_acquisition_tab(self) -> None:
        content, layout = self._make_content_widget()

        # ── Single / Live ───────────────────────────────────────────────────
        single = QtWidgets.QGroupBox("Single / Live")
        sl = QtWidgets.QVBoxLayout(single)
        sl.setContentsMargins(10, 8, 10, 10)
        sl.setSpacing(6)

        acq_btn_row = QtWidgets.QHBoxLayout()
        acq_btn_row.setSpacing(6)
        self.capture_button = QtWidgets.QPushButton("Single")
        self.start_live_button = QtWidgets.QPushButton("Live")
        self.stop_live_button = QtWidgets.QPushButton("Stop")
        for btn in [self.capture_button, self.start_live_button, self.stop_live_button]:
            btn.setMinimumHeight(32)
            acq_btn_row.addWidget(btn, 1)
        sl.addLayout(acq_btn_row)

        self.save_last_button = QtWidgets.QPushButton("Save Frame")
        self.save_last_button.setMinimumHeight(30)
        sl.addWidget(self.save_last_button)

        self.auto_levels_check = QtWidgets.QCheckBox("Auto display scale")
        self.auto_levels_check.setChecked(True)
        self.display_min_spin = QtWidgets.QDoubleSpinBox()
        self.display_min_spin.setRange(-1e12, 1e12)
        self.display_min_spin.setDecimals(3)
        self.display_min_spin.setValue(0.0)
        self.display_max_spin = QtWidgets.QDoubleSpinBox()
        self.display_max_spin.setRange(-1e12, 1e12)
        self.display_max_spin.setDecimals(3)
        self.display_max_spin.setValue(2000.0)
        self.lock_current_scale_button = QtWidgets.QPushButton("Lock")
        self.reset_auto_scale_button = QtWidgets.QPushButton("Auto")
        scale_grid = QtWidgets.QGridLayout()
        scale_grid.setContentsMargins(0, 0, 0, 0)
        scale_grid.setSpacing(6)
        scale_grid.addWidget(self.auto_levels_check, 0, 0, 1, 2)
        scale_grid.addWidget(QtWidgets.QLabel("Manual min / max:"), 1, 0)
        scale_grid.addWidget(self.display_min_spin, 1, 1)
        scale_grid.addWidget(self.display_max_spin, 1, 2)
        scale_grid.addWidget(self.lock_current_scale_button, 2, 1)
        scale_grid.addWidget(self.reset_auto_scale_button, 2, 2)
        sl.addLayout(scale_grid)
        self.show_processed_check = QtWidgets.QCheckBox("Display corrected frame when available")
        self.show_pc_check = QtWidgets.QCheckBox("Display photon-counted frame when available")
        sl.addWidget(self.show_processed_check)
        sl.addWidget(self.show_pc_check)
        layout.addWidget(single)

        # ── Sequence ────────────────────────────────────────────────────────
        seq = QtWidgets.QGroupBox("Sequence")
        ql = self._grid(seq)
        self.sequence_n_spin = QtWidgets.QSpinBox()
        self.sequence_n_spin.setRange(1, 1_000_000)
        self.sequence_n_spin.setValue(100)
        self.sequence_delay_spin = QtWidgets.QDoubleSpinBox()
        self.sequence_delay_spin.setRange(0.0, 3600.0)
        self.sequence_delay_spin.setDecimals(4)
        self.sequence_delay_spin.setValue(0.0)
        self.sequence_delay_spin.setSuffix(" s")

        fmt_row = QtWidgets.QHBoxLayout()
        fmt_row.setSpacing(8)
        self.save_hdf5_radio = QtWidgets.QRadioButton("HDF5")
        self.save_npz_radio = QtWidgets.QRadioButton("NPZ")
        self.save_hdf5_radio.setChecked(True)
        fmt_row.addWidget(self.save_hdf5_radio)
        fmt_row.addWidget(self.save_npz_radio)
        fmt_row.addStretch(1)

        self.acquire_sequence_button = QtWidgets.QPushButton("Acquire")
        self.abort_sequence_button = QtWidgets.QPushButton("■  Abort")
        self.acquire_sequence_button.setMinimumHeight(32)
        self.abort_sequence_button.setMinimumHeight(32)
        seq_btn_row = QtWidgets.QHBoxLayout()
        seq_btn_row.setSpacing(6)
        seq_btn_row.addWidget(self.acquire_sequence_button, 3)
        seq_btn_row.addWidget(self.abort_sequence_button, 1)

        self.sequence_progress = QtWidgets.QProgressBar()

        ql.addWidget(QtWidgets.QLabel("Frames:"), 0, 0)
        ql.addWidget(self.sequence_n_spin, 0, 1)
        ql.addWidget(QtWidgets.QLabel("Delay:"), 1, 0)
        ql.addWidget(self.sequence_delay_spin, 1, 1)
        ql.addWidget(QtWidgets.QLabel("Format:"), 2, 0)
        ql.addLayout(fmt_row, 2, 1)
        ql.addLayout(seq_btn_row, 3, 0, 1, 2)
        ql.addWidget(self.sequence_progress, 4, 0, 1, 2)
        layout.addWidget(seq)
        self._register_advanced(seq)

        # ── Analysis ROI ────────────────────────────────────────────────────
        viewroi = QtWidgets.QGroupBox("Analysis ROI")
        vl = self._grid(viewroi)
        vl.setColumnStretch(1, 1)
        vl.setColumnStretch(2, 1)
        self.analysis_x0_spin = QtWidgets.QSpinBox(); self.analysis_x0_spin.setRange(0, 100000)
        self.analysis_x1_spin = QtWidgets.QSpinBox(); self.analysis_x1_spin.setRange(1, 100000); self.analysis_x1_spin.setValue(512)
        self.analysis_y0_spin = QtWidgets.QSpinBox(); self.analysis_y0_spin.setRange(0, 100000)
        self.analysis_y1_spin = QtWidgets.QSpinBox(); self.analysis_y1_spin.setRange(1, 100000); self.analysis_y1_spin.setValue(512)
        vl.addWidget(QtWidgets.QLabel("x0 / x1:"), 0, 0)
        vl.addWidget(self.analysis_x0_spin, 0, 1)
        vl.addWidget(self.analysis_x1_spin, 0, 2)
        vl.addWidget(QtWidgets.QLabel("y0 / y1:"), 1, 0)
        vl.addWidget(self.analysis_y0_spin, 1, 1)
        vl.addWidget(self.analysis_y1_spin, 1, 2)
        layout.addWidget(viewroi)
        self._register_advanced(viewroi)

        specroi = QtWidgets.QGroupBox("Spectroscopy ROI")
        spl = self._grid(specroi)
        spl.setColumnStretch(1, 1)
        spl.setColumnStretch(2, 1)
        self.spectroscopy_enable_check = QtWidgets.QCheckBox("Enable")
        self.spectroscopy_enable_check.setChecked(False)
        self.spec_x0_spin = QtWidgets.QSpinBox(); self.spec_x0_spin.setRange(0, 100000)
        self.spec_x1_spin = QtWidgets.QSpinBox(); self.spec_x1_spin.setRange(1, 100000); self.spec_x1_spin.setValue(512)
        self.spec_y0_spin = QtWidgets.QSpinBox(); self.spec_y0_spin.setRange(0, 100000)
        self.spec_y1_spin = QtWidgets.QSpinBox(); self.spec_y1_spin.setRange(1, 100000); self.spec_y1_spin.setValue(512)
        self.spec_use_displayed_check = QtWidgets.QCheckBox("Use displayed unit")
        self.spec_use_displayed_check.setChecked(True)
        spl.addWidget(self.spectroscopy_enable_check, 0, 0, 1, 3)
        spl.addWidget(QtWidgets.QLabel("x0 / x1:"), 1, 0)
        spl.addWidget(self.spec_x0_spin, 1, 1); spl.addWidget(self.spec_x1_spin, 1, 2)
        spl.addWidget(QtWidgets.QLabel("y0 / y1:"), 2, 0)
        spl.addWidget(self.spec_y0_spin, 2, 1); spl.addWidget(self.spec_y1_spin, 2, 2)
        spl.addWidget(self.spec_use_displayed_check, 3, 0, 1, 3)
        layout.addWidget(specroi)
        self._register_advanced(specroi)

        layout.addStretch(1)
        self.tabs.addTab(self._scrollable_tab(content), "Acquisition")

    def _build_calibration_tab(self) -> None:
        content, layout = self._make_content_widget()

        # ── Manual calibration frames ───────────────────────────────────────
        cal = QtWidgets.QGroupBox("Calibration Frames")
        cl = self._grid(cal)

        self.calibration_n_spin = QtWidgets.QSpinBox()
        self.calibration_n_spin.setRange(1, 10000)
        self.calibration_n_spin.setValue(20)
        cl.addWidget(QtWidgets.QLabel("Frames:"), 0, 0)
        cl.addWidget(self.calibration_n_spin, 0, 1)

        self.acquire_bias_button = QtWidgets.QPushButton("Bias")
        self.acquire_dark_button = QtWidgets.QPushButton("Dark")
        self.acquire_background_button = QtWidgets.QPushButton("Background")
        for btn in [self.acquire_bias_button, self.acquire_dark_button, self.acquire_background_button]:
            btn.setMinimumHeight(30)
        acq_row = QtWidgets.QHBoxLayout()
        acq_row.setSpacing(6)
        acq_row.addWidget(self.acquire_bias_button, 1)
        acq_row.addWidget(self.acquire_dark_button, 1)
        acq_row.addWidget(self.acquire_background_button, 1)
        cl.addLayout(acq_row, 1, 0, 1, 2)

        self.subtract_bias_check = QtWidgets.QCheckBox("Subtract bias")
        self.subtract_dark_check = QtWidgets.QCheckBox("Subtract dark")
        self.subtract_background_check = QtWidgets.QCheckBox("Subtract background")
        self.clip_negative_check = QtWidgets.QCheckBox("Clip corrected frame at zero")
        corr_row = QtWidgets.QHBoxLayout()
        corr_row.setSpacing(12)
        corr_row.addWidget(self.subtract_bias_check)
        corr_row.addWidget(self.subtract_dark_check)
        corr_row.addWidget(self.subtract_background_check)
        corr_row.addStretch(1)
        cl.addLayout(corr_row, 2, 0, 1, 2)
        cl.addWidget(self.clip_negative_check, 3, 0, 1, 2)

        self.save_calibration_button = QtWidgets.QPushButton("Save Calibration")
        self.save_calibration_button.setMinimumHeight(30)
        cl.addWidget(self.save_calibration_button, 4, 0, 1, 2)

        self.calibration_label = QtWidgets.QLabel("No calibration loaded / acquired")
        self.calibration_label.setWordWrap(True)
        self.calibration_label.setStyleSheet("color: #607590; font-size: 9pt;")
        cl.addWidget(self.calibration_label, 5, 0, 1, 2)
        layout.addWidget(cal)

        # ── Quick auto-calibration ──────────────────────────────────────────
        auto = QtWidgets.QGroupBox("Quick Auto-Calibration")
        al = self._grid(auto)
        al.setColumnStretch(1, 1)
        al.setColumnStretch(2, 1)

        self.auto_cal_bias_n_spin = QtWidgets.QSpinBox()
        self.auto_cal_bias_n_spin.setRange(1, 10000)
        self.auto_cal_bias_n_spin.setValue(20)
        self.auto_cal_dark_n_spin = QtWidgets.QSpinBox()
        self.auto_cal_dark_n_spin.setRange(1, 10000)
        self.auto_cal_dark_n_spin.setValue(20)
        self.auto_threshold_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.auto_threshold_sigma_spin.setRange(1.0, 20.0)
        self.auto_threshold_sigma_spin.setDecimals(2)
        self.auto_threshold_sigma_spin.setValue(5.0)
        self.auto_threshold_sigma_spin.setSuffix(" σ")

        al.addWidget(QtWidgets.QLabel("Bias frames:"), 0, 0)
        al.addWidget(self.auto_cal_bias_n_spin, 0, 1)
        al.addWidget(QtWidgets.QLabel("Dark frames:"), 1, 0)
        al.addWidget(self.auto_cal_dark_n_spin, 1, 1)
        al.addWidget(QtWidgets.QLabel("Threshold sigma:"), 2, 0)
        al.addWidget(self.auto_threshold_sigma_spin, 2, 1)

        self.quick_auto_cal_button = QtWidgets.QPushButton("Quick Auto-Cal")
        self.quick_auto_cal_button.setMinimumHeight(32)
        al.addWidget(self.quick_auto_cal_button, 3, 0, 1, 2)

        self.auto_cal_label = QtWidgets.QLabel("Auto-calibration: not run")
        self.auto_cal_label.setWordWrap(True)
        self.auto_cal_label.setStyleSheet("color: #607590; font-size: 9pt;")
        al.addWidget(self.auto_cal_label, 4, 0, 1, 2)
        layout.addWidget(auto)

        layout.addStretch(1)
        self.tabs.addTab(self._scrollable_tab(content), "Calibration")

    def _build_units_tab(self) -> None:
        content, layout = self._make_content_widget()

        # ── Display unit ────────────────────────────────────────────────────
        disp = QtWidgets.QGroupBox("Units")
        dl = self._grid(disp)
        self.display_unit_combo = QtWidgets.QComboBox()
        for label in [
            "Raw ADU",
            "Corrected ADU",
            "Photon-counted events",
            "SDK estimated electrons",
            "SDK estimated photons",
            "Manual estimated electrons",
            "Manual estimated photons",
        ]:
            self.display_unit_combo.addItem(label)
        self.wavelength_spin = QtWidgets.QDoubleSpinBox()
        self.wavelength_spin.setRange(200.0, 1200.0)
        self.wavelength_spin.setDecimals(2)
        self.wavelength_spin.setValue(650.0)
        self.wavelength_spin.setSuffix(" nm")
        dl.addWidget(QtWidgets.QLabel("Display unit:"), 0, 0)
        dl.addWidget(self.display_unit_combo, 0, 1)
        dl.addWidget(QtWidgets.QLabel("Wavelength:"), 1, 0)
        dl.addWidget(self.wavelength_spin, 1, 1)
        layout.addWidget(disp)

        # ── Manual calibration factors ──────────────────────────────────────
        man = QtWidgets.QGroupBox("Manual Factors")
        ml = self._grid(man)
        self.manual_e_per_adu_spin = QtWidgets.QDoubleSpinBox()
        self.manual_e_per_adu_spin.setRange(1e-9, 1e9)
        self.manual_e_per_adu_spin.setDecimals(6)
        self.manual_e_per_adu_spin.setValue(1.0)
        self.manual_e_per_adu_spin.setSuffix(" e⁻/ADU")
        self.manual_qe_spin = QtWidgets.QDoubleSpinBox()
        self.manual_qe_spin.setRange(0.001, 1.0)
        self.manual_qe_spin.setDecimals(4)
        self.manual_qe_spin.setValue(0.90)
        self.manual_use_em_gain_check = QtWidgets.QCheckBox(
            "Divide by EM gain")
        self.manual_use_em_gain_check.setChecked(True)
        ml.addWidget(QtWidgets.QLabel("Electrons / ADU:"), 0, 0)
        ml.addWidget(self.manual_e_per_adu_spin, 0, 1)
        ml.addWidget(QtWidgets.QLabel("QE fraction:"), 1, 0)
        ml.addWidget(self.manual_qe_spin, 1, 1)
        ml.addWidget(self.manual_use_em_gain_check, 2, 0, 1, 2)
        layout.addWidget(man)
        self._register_advanced(man)

        # ── SDK Count Convert ───────────────────────────────────────────────
        sdk = QtWidgets.QGroupBox("SDK Count Convert")
        sk = self._grid(sdk)
        self.check_countconvert_button = QtWidgets.QPushButton(
            "Check Availability")
        self.check_countconvert_button.setMinimumHeight(30)
        self.sdk_countconvert_label = QtWidgets.QLabel("Not checked")
        self.sdk_countconvert_label.setWordWrap(True)
        self.sdk_countconvert_label.setStyleSheet("color: #607590; font-size: 9pt;")
        sk.addWidget(self.check_countconvert_button, 0, 0, 1, 2)
        sk.addWidget(self.sdk_countconvert_label, 1, 0, 1, 2)
        layout.addWidget(sdk)
        self._register_advanced(sdk)

        # ── Help note ───────────────────────────────────────────────────────
        self.units_help_label = QtWidgets.QLabel(
            "Raw / corrected ADU are direct camera units. SDK electrons/photons are "
            "Andor SDK estimates. Manual factors require a measured e⁻/ADU calibration "
            "and QE — use only as calibrated estimates.")
        self.units_help_label.setWordWrap(True)
        self.units_help_label.setStyleSheet(
            "color: #4e6a8c; font-size: 8.5pt; padding: 4px 2px;")
        layout.addWidget(self.units_help_label)

        layout.addStretch(1)
        self.tabs.addTab(self._scrollable_tab(content), "Units")

    def _build_photon_counting_tab(self) -> None:
        content, layout = self._make_content_widget()

        # ── Options ─────────────────────────────────────────────────────────
        opts = QtWidgets.QGroupBox("Photon-Counting Options")
        ol = QtWidgets.QVBoxLayout(opts)
        ol.setContentsMargins(10, 8, 10, 10)
        ol.setSpacing(8)
        self.pc_enable_check = QtWidgets.QCheckBox("Enable photon counting")
        self.pc_use_corrected_check = QtWidgets.QCheckBox(
            "Use corrected frame")
        self.pc_use_corrected_check.setChecked(True)
        self.pc_use_sdk_check = QtWidgets.QCheckBox(
            "Use SDK postprocess")
        ol.addWidget(self.pc_enable_check)
        ol.addWidget(self.pc_use_corrected_check)
        ol.addWidget(self.pc_use_sdk_check)
        layout.addWidget(opts)

        # ── Thresholds ──────────────────────────────────────────────────────
        thr = QtWidgets.QGroupBox("Thresholds")
        tl = self._grid(thr)
        self.pc_t1_spin = QtWidgets.QDoubleSpinBox()
        self.pc_t1_spin.setRange(0, 1e9)
        self.pc_t1_spin.setDecimals(3)
        self.pc_t1_spin.setValue(5.0)
        self.pc_t1_spin.setSuffix(" ADU")
        self.pc_t2_spin = QtWidgets.QDoubleSpinBox()
        self.pc_t2_spin.setRange(0, 1e9)
        self.pc_t2_spin.setDecimals(3)
        self.pc_t2_spin.setValue(0.0)
        self.pc_t2_spin.setSuffix(" ADU")
        self.pc_t3_spin = QtWidgets.QDoubleSpinBox()
        self.pc_t3_spin.setRange(0, 1e9)
        self.pc_t3_spin.setDecimals(3)
        self.pc_t3_spin.setValue(0.0)
        self.pc_t3_spin.setSuffix(" ADU")
        tl.addWidget(QtWidgets.QLabel("Threshold 1:"), 0, 0)
        tl.addWidget(self.pc_t1_spin, 0, 1)
        tl.addWidget(QtWidgets.QLabel("Threshold 2:"), 1, 0)
        tl.addWidget(self.pc_t2_spin, 1, 1)
        tl.addWidget(QtWidgets.QLabel("Threshold 3:"), 2, 0)
        tl.addWidget(self.pc_t3_spin, 2, 1)
        layout.addWidget(thr)

        # ── Actions ─────────────────────────────────────────────────────────
        act = QtWidgets.QGroupBox("Actions")
        av = QtWidgets.QVBoxLayout(act)
        av.setContentsMargins(10, 8, 10, 10)
        av.setSpacing(8)
        self.process_pc_button = QtWidgets.QPushButton("Process Frame")
        self.save_pc_button = QtWidgets.QPushButton("Save PC Frame")
        for btn in [self.process_pc_button, self.save_pc_button]:
            btn.setMinimumHeight(32)
        av.addWidget(self.process_pc_button)
        av.addWidget(self.save_pc_button)

        self.pc_label = QtWidgets.QLabel("Photon-counting result: —")
        self.pc_label.setWordWrap(True)
        self.pc_label.setStyleSheet(
            "font-family: 'Cascadia Mono', Consolas, monospace; "
            "font-size: 8.5pt; color: #90b8d8;")
        av.addWidget(self.pc_label)
        layout.addWidget(act)

        layout.addStretch(1)
        self.tabs.addTab(self._scrollable_tab(content), "Photon Counting")

    # --------------------------------------------------------------- Signals
    def _connect_signals(self) -> None:
        self.control_mode_combo.currentTextChanged.connect(lambda _: self._apply_control_mode())
        self.quick_connect_button.clicked.connect(self.connect_camera)
        self.quick_single_button.clicked.connect(self.capture_single_frame)
        self.quick_live_button.clicked.connect(self.start_live)
        self.quick_stop_button.clicked.connect(self.stop_live)
        self.quick_save_button.clicked.connect(self.save_current_frame)
        self.browse_base_button.clicked.connect(self._browse_base_dir)
        self.create_session_button.clicked.connect(self.create_session)
        self.browse_dll_button.clicked.connect(self._browse_dll)
        self.connect_button.clicked.connect(self.connect_camera)
        self.disconnect_button.clicked.connect(self.disconnect_camera)
        self.apply_temp_button.clicked.connect(self.apply_temperature)
        self.apply_shutter_button.clicked.connect(self.apply_shutter)
        self.force_open_button.clicked.connect(self.force_shutter_open)
        self.force_closed_button.clicked.connect(self.force_shutter_closed)
        self.auto_shutter_check.toggled.connect(self._on_auto_shutter_toggled)
        self.apply_settings_button.clicked.connect(lambda: self.apply_camera_settings(AcquisitionMode.SINGLE_SCAN))
        self.full_roi_button.clicked.connect(self._set_full_roi)
        self.center_roi_button.clicked.connect(self._set_center_roi)
        self.capture_button.clicked.connect(self.capture_single_frame)
        self.start_live_button.clicked.connect(self.start_live)
        self.stop_live_button.clicked.connect(self.stop_live)
        self.save_last_button.clicked.connect(self.save_current_frame)
        self.acquire_sequence_button.clicked.connect(self.acquire_sequence)
        self.abort_sequence_button.clicked.connect(self.abort_sequence)
        self.acquire_bias_button.clicked.connect(lambda: self.acquire_calibration("bias"))
        self.acquire_dark_button.clicked.connect(lambda: self.acquire_calibration("dark"))
        self.acquire_background_button.clicked.connect(lambda: self.acquire_calibration("background"))
        self.save_calibration_button.clicked.connect(self.save_calibration_frames)
        self.process_pc_button.clicked.connect(self.process_current_photon_counting)
        self.save_pc_button.clicked.connect(self.save_current_photon_counted)
        self.toggle_controls_button.clicked.connect(self.toggle_controls_panel)
        self.clean_view_button.clicked.connect(self.toggle_clean_detection_view)
        self.display_unit_combo.currentTextChanged.connect(lambda _: self._display_current_frame())
        self.wavelength_spin.valueChanged.connect(lambda _: self._display_current_frame())
        self.manual_e_per_adu_spin.valueChanged.connect(lambda _: self._display_current_frame())
        self.manual_qe_spin.valueChanged.connect(lambda _: self._display_current_frame())
        self.manual_use_em_gain_check.toggled.connect(lambda _: self._display_current_frame())
        self.auto_levels_check.toggled.connect(lambda _: self._display_current_frame())
        self.display_min_spin.valueChanged.connect(lambda _: self._display_current_frame())
        self.display_max_spin.valueChanged.connect(lambda _: self._display_current_frame())
        self.lock_current_scale_button.clicked.connect(self.lock_current_display_scale)
        self.reset_auto_scale_button.clicked.connect(self.reset_auto_display_scale)
        self.fit_view_button.clicked.connect(self.fit_image_view)
        self.reset_zoom_button.clicked.connect(self.fit_image_view)
        self.one_to_one_button.clicked.connect(self.one_to_one_image_view)
        for widget in [self.exposure_spin, self.em_gain_spin, self.frame_transfer_check, self.hstart_spin, self.hend_spin, self.vstart_spin, self.vend_spin, self.hbin_spin, self.vbin_spin, self.trigger_combo, self.preamp_combo, self.hs_speed_combo, self.vs_speed_combo]:
            try:
                widget.valueChanged.connect(lambda *_: self._schedule_live_reconfigure())
            except Exception:
                try:
                    widget.currentIndexChanged.connect(lambda *_: self._schedule_live_reconfigure())
                except Exception:
                    try:
                        widget.toggled.connect(lambda *_: self._schedule_live_reconfigure())
                    except Exception:
                        pass
        for widget in [self.subtract_bias_check, self.subtract_dark_check, self.subtract_background_check, self.clip_negative_check, self.pc_enable_check, self.pc_use_corrected_check, self.pc_t1_spin, self.pc_t2_spin, self.pc_t3_spin, self.spectroscopy_enable_check, self.spec_x0_spin, self.spec_x1_spin, self.spec_y0_spin, self.spec_y1_spin, self.spec_use_displayed_check]:
            try:
                widget.valueChanged.connect(lambda *_: self._reprocess_current_frame())
            except Exception:
                try:
                    widget.toggled.connect(lambda *_: self._reprocess_current_frame())
                except Exception:
                    pass
        self.check_countconvert_button.clicked.connect(self.check_sdk_count_convert)
        self.quick_auto_cal_button.clicked.connect(self.run_quick_auto_calibration)

    # --------------------------------------------------------------- Helpers
    @staticmethod
    def _enum_combo(enum_cls: type[E], default: E, allowed: list[E] | None = None) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        values = allowed if allowed is not None else list(enum_cls)
        for item in values:
            combo.addItem(getattr(item, "name", str(item)), int(item))
        idx = combo.findData(int(default))
        if idx >= 0:
            combo.setCurrentIndex(idx)
        return combo

    @staticmethod
    def _combo_int(combo: QtWidgets.QComboBox) -> int:
        return int(combo.currentData())

    def _browse_base_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output base directory", self.base_dir_edit.text())
        if path:
            self.base_dir_edit.setText(path)

    def _browse_dll(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose atmcd64d.dll", "", "DLL files (*.dll);;All files (*)")
        if path:
            self.dll_edit.setText(path)

    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{stamp}] {message}")
        if self.session is not None:
            try:
                self.session.log(message)
            except Exception:
                pass
        try:
            self.statusBar().showMessage(message, 5000)
        except Exception:
            pass

    def _show_error(self, title: str, exc: BaseException | str) -> None:
        msg = str(exc)
        self._log(f"ERROR - {title}: {msg}")
        QtWidgets.QMessageBox.critical(self, title, msg)

    def _current_output_dir(self) -> Path:
        if self.session is not None:
            return self.session.root
        path = Path(self.base_dir_edit.text().strip() or self.output_base)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create_session(self) -> None:
        try:
            self.session = ExperimentSession.create(
                self.base_dir_edit.text().strip() or self.output_base,
                experiment_name=self.experiment_edit.text().strip() or "Andor_iXon897_session",
                sample_name=self.sample_edit.text().strip(),
                operator=self.operator_edit.text().strip(),
                notes=self.notes_edit.toPlainText().strip(),
            )
            self.session_label.setText(f"Active session:\n{self.session.root}")
            self._log(f"Active session: {self.session.root}")
            self._update_header_status()
        except BaseException as exc:
            self._show_error("Session creation failed", exc)

    def _set_connected_ui(self, connected: bool) -> None:
        for widget in [
            self.disconnect_button, self.apply_temp_button, self.apply_shutter_button, self.force_open_button, self.force_closed_button,
            self.apply_settings_button, self.capture_button, self.start_live_button, self.stop_live_button, self.save_last_button,
            self.acquire_sequence_button, self.abort_sequence_button, self.acquire_bias_button, self.acquire_dark_button,
            self.acquire_background_button, self.save_calibration_button, self.process_pc_button, self.save_pc_button,
        ]:
            widget.setEnabled(connected)
        self.connect_button.setEnabled(not connected)
        self.browse_dll_button.setEnabled(not connected and not self.mock)
        self.dll_edit.setEnabled(not connected and not self.mock)
        self.stop_live_button.setEnabled(False)
        self.abort_sequence_button.setEnabled(False)

    def _set_live_ui(self, running: bool) -> None:
        self.start_live_button.setEnabled(not running)
        self.stop_live_button.setEnabled(running)
        self.capture_button.setEnabled(not running)
        self.acquire_sequence_button.setEnabled(not running)
        self.apply_settings_button.setEnabled(True)

    def _set_sequence_ui(self, running: bool) -> None:
        self.acquire_sequence_button.setEnabled(not running)
        self.abort_sequence_button.setEnabled(running)
        self.capture_button.setEnabled(not running)
        self.start_live_button.setEnabled(not running)
        self.apply_settings_button.setEnabled(not running)

    def toggle_controls_panel(self) -> None:
        self.controls_hidden = not self.controls_hidden
        self.left_panel.setVisible(not self.controls_hidden)
        self.toggle_controls_button.setText("⊞  Show Controls" if self.controls_hidden else "⊟  Hide Controls")
        if self.controls_hidden:
            self.main_splitter.setSizes([0, self.width()])
            self.statusBar().showMessage("Controls hidden — detector view expanded.")
        else:
            self._reset_splitter_sizes()
            self.statusBar().showMessage("Controls restored.")

    def toggle_clean_detection_view(self) -> None:
        self.clean_detection_mode = not self.clean_detection_mode
        if self.clean_detection_mode:
            if not self.controls_hidden:
                self.toggle_controls_panel()
            self.inspector_tabs.setVisible(False)
            self.clean_view_button.setText("⊞  Exit Full Screen")
            self.center_splitter.setSizes([self.height(), 0])
            self.statusBar().showMessage("Full screen detector view active.")
        else:
            if self.controls_hidden:
                self.toggle_controls_panel()
            self.inspector_tabs.setVisible(True)
            self.clean_view_button.setText("⬛  Full Screen View")
            self._reset_splitter_sizes()
            self.statusBar().showMessage("Full screen view closed.")

    def _unit_settings(self) -> UnitConversionSettings:
        return UnitConversionSettings(
            display_unit=str(self.display_unit_combo.currentText()) if hasattr(self, "display_unit_combo") else "Raw ADU",
            wavelength_nm=float(self.wavelength_spin.value()) if hasattr(self, "wavelength_spin") else 650.0,
            manual_electrons_per_adu=float(self.manual_e_per_adu_spin.value()) if hasattr(self, "manual_e_per_adu_spin") else 1.0,
            manual_qe_fraction=float(self.manual_qe_spin.value()) if hasattr(self, "manual_qe_spin") else 0.90,
            use_em_gain_for_manual=bool(self.manual_use_em_gain_check.isChecked()) if hasattr(self, "manual_use_em_gain_check") else True,
        )

    def check_sdk_count_convert(self) -> None:
        if self.camera is None:
            self.sdk_countconvert_label.setText("SDK Count Convert: camera not connected")
            return
        try:
            lines = []
            for mode, label in [(1, "electrons"), (2, "photons")]:
                available = False
                if hasattr(self.camera.sdk, "count_convert_mode_available"):
                    available = bool(self.camera.sdk.count_convert_mode_available(mode))
                lines.append(f"mode {mode} ({label}): {'available' if available else 'not available or not valid now'}")
            if hasattr(self.camera.sdk, "get_count_convert_wavelength_range_nm"):
                lo, hi = self.camera.sdk.get_count_convert_wavelength_range_nm()
                lines.append(f"wavelength range: {lo:.1f}–{hi:.1f} nm")
            if hasattr(self.camera.sdk, "get_qe_percent"):
                qe = self.camera.sdk.get_qe_percent(float(self.wavelength_spin.value()))
                lines.append(f"QE({self.wavelength_spin.value():.1f} nm): {qe:.2f} %")
            self.sdk_countconvert_label.setText("SDK Count Convert:\n" + "\n".join(lines))
            self._log("SDK Count Convert checked")
        except BaseException as exc:
            self.sdk_countconvert_label.setText(f"SDK Count Convert check failed: {exc}")
            self._log(f"SDK Count Convert check failed: {exc}")

    # --------------------------------------------------------------- Settings
    def _temperature_settings(self) -> TemperatureSettings:
        return TemperatureSettings(
            setpoint_c=int(self.temp_setpoint_spin.value()),
            cooler_on=bool(self.cooler_check.isChecked()),
            keep_cooler_on_at_shutdown=bool(self.keep_cooler_check.isChecked()),
            fan_mode=FanMode(self._combo_int(self.fan_combo)),
        )

    def _shutter_settings(self) -> ShutterSettings:
        return ShutterSettings(
            ttl_high_opens=bool(self.ttl_high_check.isChecked()),
            internal_mode=ShutterMode(self._combo_int(self.internal_shutter_combo)),
            external_mode=ShutterMode(self._combo_int(self.external_shutter_combo)),
            opening_ms=int(self.opening_ms_spin.value()),
            closing_ms=int(self.closing_ms_spin.value()),
        )

    def _roi_settings(self) -> ROI:
        return ROI(
            hstart=int(self.hstart_spin.value()),
            hend=int(self.hend_spin.value()),
            vstart=int(self.vstart_spin.value()),
            vend=int(self.vend_spin.value()),
            hbin=int(self.hbin_spin.value()),
            vbin=int(self.vbin_spin.value()),
        )

    def _acquisition_settings(self, mode: AcquisitionMode) -> AcquisitionSettings:
        return AcquisitionSettings(
            acquisition_mode=mode,
            read_mode=ReadMode.IMAGE,
            trigger_mode=TriggerMode(self._combo_int(self.trigger_combo)),
            exposure_s=float(self.exposure_spin.value()),
            frame_transfer=bool(self.frame_transfer_check.isChecked()),
            roi=self._roi_settings(),
        )

    def _emccd_settings(self) -> EMCCDSettings:
        return EMCCDSettings(
            output_amplifier=0,
            ad_channel=0,
            hs_speed_index=max(0, self.hs_speed_combo.currentIndex()),
            vs_speed_index=max(0, self.vs_speed_combo.currentIndex()),
            preamp_gain_index=max(0, self.preamp_combo.currentIndex()),
            em_gain_mode=EMGainMode(self._combo_int(self.em_gain_mode_combo)),
            em_gain=int(self.em_gain_spin.value()),
            em_advanced=bool(self.em_advanced_check.isChecked()),
            baseline_clamp=bool(self.baseline_clamp_check.isChecked()),
        )

    def _correction_flags(self) -> dict[str, bool]:
        return {
            "bias": bool(self.subtract_bias_check.isChecked()),
            "dark": bool(self.subtract_dark_check.isChecked()),
            "background": bool(self.subtract_background_check.isChecked()),
            "clip": bool(self.clip_negative_check.isChecked()),
        }

    def _photon_settings(self, *, force_enabled: bool | None = None) -> PhotonCountingSettings:
        thresholds = [float(self.pc_t1_spin.value())]
        for spin in [self.pc_t2_spin, self.pc_t3_spin]:
            if float(spin.value()) > 0:
                thresholds.append(float(spin.value()))
        enabled = bool(self.pc_enable_check.isChecked()) if force_enabled is None else bool(force_enabled)
        return PhotonCountingSettings(
            enabled=enabled,
            thresholds_adu=tuple(thresholds),
            use_corrected_frame=bool(self.pc_use_corrected_check.isChecked()),
            use_sdk_postprocess=bool(self.pc_use_sdk_check.isChecked()),
        )

    def _reprocess_current_frame(self) -> None:
        """Recompute correction, photon-counting and display without acquiring a new frame."""
        if self.last_raw is None:
            return
        try:
            self.last_processed = apply_calibration(
                self.last_raw,
                self.calibration,
                subtract_bias=bool(self.subtract_bias_check.isChecked()),
                subtract_dark=bool(self.subtract_dark_check.isChecked()),
                subtract_background=bool(self.subtract_background_check.isChecked()),
                clip_negative=bool(self.clip_negative_check.isChecked()),
            )
            if self.pc_enable_check.isChecked() or str(self.display_unit_combo.currentText()) == "Photon-counted events":
                self._run_photon_counting(show_errors=False)
            self._display_current_frame()
            self._update_stats(source="reprocess")
        except Exception as exc:
            self._log(f"Reprocess warning: {exc}")

    def _schedule_live_reconfigure(self) -> None:
        if not getattr(self, "live_running", False) or self.camera is None:
            return
        if getattr(self, "_live_reconfigure_pending", False):
            return
        self._live_reconfigure_pending = True
        QtCore.QTimer.singleShot(350, self._apply_live_reconfigure)

    def _apply_live_reconfigure(self) -> None:
        self._live_reconfigure_pending = False
        if not self.live_running or self.camera is None:
            return
        try:
            self.live_timer.stop()
            self.camera.stop_continuous()
            self._log("Live parameter change: acquisition paused for safe reconfiguration")
            if not self.apply_camera_settings(AcquisitionMode.RUN_TILL_ABORT, show_errors=False, allow_while_live=True):
                self._log("Live parameter change failed; live was not restarted")
                self.live_running = False
                self._set_live_ui(False)
                return
            self.camera.start_continuous()
            self.live_timer.start()
            self._log("Live parameter change applied and live acquisition restarted")
        except Exception as exc:
            self.live_running = False
            self._set_live_ui(False)
            self._log(f"Live reconfiguration warning: {exc}")
            self._show_error("Live reconfiguration failed", exc)

    # ------------------------------------------------------------- Camera ops
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
            if self.auto_shutter_check.isChecked():
                self._command_detection_shutter(False, reason="initial idle")
            else:
                self.apply_shutter(show_errors=False)
            self.apply_camera_settings(AcquisitionMode.SINGLE_SCAN, show_errors=False)
            self.status_timer.start()
            self._refresh_status()
            self._log("Camera connected")
            self._update_header_status()
        except BaseException as exc:
            self.camera = None
            self._set_connected_ui(False)
            self._show_error("Connection failed", exc)

    def disconnect_camera(self) -> None:
        self.stop_live()
        self.abort_sequence()
        if self.camera is not None:
            try:
                self.camera.shutdown(turn_cooler_off=False)
            except BaseException as exc:
                self._log(f"Shutdown warning: {exc}")
        self.camera = None
        self.shutter_commanded_state = "UNKNOWN"
        self._update_shutter_status()
        self.status_timer.stop()
        self.identity_label.setText("Not connected")
        self.temp_label.setText("Temperature: --")
        self.camera_status_label.setText("SDK status: --")
        self._set_connected_ui(False)
        self._log("Camera disconnected. Cooler was not forcibly turned off.")
        self._update_header_status()

    def apply_temperature(self, *, show_errors: bool = True) -> bool:
        if self.camera is None:
            return False
        try:
            state = self.camera.configure_temperature(self._temperature_settings())
            self._update_temperature_label(state.to_dict())
            self._log(f"Temperature applied: setpoint {self.temp_setpoint_spin.value()} °C")
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
            settings = self.camera.configure_shutter(self._shutter_settings())
            self.shutter_commanded_mode = "AUTO" if self.auto_shutter_check.isChecked() else "MANUAL"
            self.shutter_commanded_state = self._infer_shutter_state(settings)
            self._update_shutter_status()
            self._log(f"Shutter applied: internal={settings.internal_mode.name}, external={settings.external_mode.name}, commanded={self.shutter_commanded_state}")
            return True
        except BaseException as exc:
            if show_errors:
                self._show_error("Shutter error", exc)
            else:
                self._log(f"Shutter setup warning: {exc}")
            return False

    def _infer_shutter_state(self, settings: ShutterSettings) -> str:
        if settings.internal_mode == ShutterMode.PERMANENTLY_CLOSED or settings.external_mode == ShutterMode.PERMANENTLY_CLOSED:
            return "CLOSED"
        if settings.internal_mode == ShutterMode.PERMANENTLY_OPEN or settings.external_mode == ShutterMode.PERMANENTLY_OPEN:
            return "OPEN"
        return "AUTO"

    def _automatic_shutter_settings(self, open_state: bool) -> ShutterSettings:
        mode = ShutterMode.PERMANENTLY_OPEN if open_state else ShutterMode.PERMANENTLY_CLOSED
        return ShutterSettings(
            ttl_high_opens=bool(self.ttl_high_check.isChecked()),
            internal_mode=mode,
            external_mode=mode,
            opening_ms=int(self.opening_ms_spin.value()),
            closing_ms=int(self.closing_ms_spin.value()),
        )

    def _command_detection_shutter(self, open_state: bool, *, reason: str = "detection") -> bool:
        if self.camera is None:
            return False
        if not self.auto_shutter_check.isChecked():
            self.shutter_commanded_mode = "MANUAL"
            self._update_shutter_status()
            return self.apply_shutter(show_errors=False)
        try:
            settings = self.camera.configure_shutter(self._automatic_shutter_settings(open_state))
            self.shutter_commanded_mode = "AUTO"
            self.shutter_commanded_state = "OPEN" if open_state else "CLOSED"
            self._update_shutter_status()
            self._log(f"Automatic shutter {self.shutter_commanded_state.lower()} for {reason}: internal={settings.internal_mode.name}, external={settings.external_mode.name}")
            return True
        except BaseException as exc:
            self._log(f"Automatic shutter warning: {exc}")
            return False

    def _update_shutter_status(self) -> None:
        text = f"Shutter: {self.shutter_commanded_mode} — commanded {self.shutter_commanded_state}"
        if hasattr(self, "shutter_status_label"):
            self.shutter_status_label.setText(text)
        if hasattr(self, "shutter_tile"):
            self.shutter_tile.value_label.setText(f"{self.shutter_commanded_mode} · {self.shutter_commanded_state}")  # type: ignore[attr-defined]
            self._set_tile_state(self.shutter_tile, "Good" if self.shutter_commanded_state == "OPEN" else "Neutral")

    def _on_auto_shutter_toggled(self, checked: bool) -> None:
        self.shutter_commanded_mode = "AUTO" if checked else "MANUAL"
        if checked and not self.live_running:
            self._command_detection_shutter(False, reason="auto mode idle")
        else:
            self.apply_shutter(show_errors=False)
        self._update_shutter_status()

    def force_shutter_open(self) -> None:
        self.auto_shutter_check.setChecked(False)
        self._set_combo_data(self.internal_shutter_combo, int(ShutterMode.PERMANENTLY_OPEN))
        self._set_combo_data(self.external_shutter_combo, int(ShutterMode.PERMANENTLY_OPEN))
        self.apply_shutter()

    def force_shutter_closed(self) -> None:
        self.auto_shutter_check.setChecked(False)
        self._set_combo_data(self.internal_shutter_combo, int(ShutterMode.PERMANENTLY_CLOSED))
        self._set_combo_data(self.external_shutter_combo, int(ShutterMode.PERMANENTLY_CLOSED))
        self.apply_shutter()

    @staticmethod
    def _set_combo_data(combo: QtWidgets.QComboBox, value: int) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def apply_camera_settings(self, mode: AcquisitionMode, *, show_errors: bool = True, allow_while_live: bool = False) -> bool:
        if self.camera is None:
            return False
        try:
            if self.live_running and not allow_while_live:
                raise RuntimeError("Live is running. Use live reconfiguration or stop live before changing acquisition/readout settings.")
            if not self.auto_shutter_check.isChecked():
                self.apply_shutter(show_errors=False)
            acq = self._acquisition_settings(mode)
            self.last_acq_info = self.camera.configure_acquisition(acq)
            requested_gain = int(self.em_gain_spin.value())
            em = self.camera.configure_emccd(
                self._emccd_settings(),
                allow_high_gain_without_stable_temp=bool(self.allow_gain_check.isChecked()),
                clamp_gain_to_sdk_range=True,
            )
            self.requested_em_gain = requested_gain
            self.applied_em_gain = int(em.em_gain) if em.em_gain is not None else None
            gain_note = ""
            if self.applied_em_gain is not None and self.applied_em_gain != requested_gain:
                gain_note = f" (requested {requested_gain}; SDK minimum/applied {self.applied_em_gain})"
            if hasattr(self, "gain_status_label"):
                self.gain_status_label.setText(f"Gain request: {requested_gain} | SDK applied: {self.applied_em_gain}{gain_note}")
            self._log(f"Camera settings applied: mode={mode.name}, exposure={self.last_acq_info['actual_timings_s']['exposure']:.6f} s, EM gain={self.applied_em_gain}{gain_note}")
            return True
        except BaseException as exc:
            if show_errors:
                self._show_error("Camera settings error", exc)
            else:
                self._log(f"Camera settings warning: {exc}")
            return False

    def _populate_capability_controls(self) -> None:
        if self.camera is None:
            return
        self.capability_summary = self.camera.get_capability_summary()
        self.preamp_combo.clear(); self.hs_speed_combo.clear(); self.vs_speed_combo.clear()
        for i, gain in enumerate(self.capability_summary.get("preamp_gains", [])):
            self.preamp_combo.addItem(f"{i}: {gain:.3g}×", i)
        for i, speed in enumerate(self.capability_summary.get("hs_speeds_mhz_output_amp_0", [])):
            self.hs_speed_combo.addItem(f"{i}: {speed:.3g} MHz", i)
        for i, speed in enumerate(self.capability_summary.get("vs_speeds_us", [])):
            self.vs_speed_combo.addItem(f"{i}: {speed:.3g} µs", i)
        low, high = self.capability_summary.get("em_gain_range", (1, 300))
        self.em_gain_spin.setRange(1, int(max(high, low, 1)))
        self.em_gain_spin.setValue(1)
        ident = self.camera.get_identity()
        w, h = ident.detector_shape
        for spin in [self.hstart_spin, self.hend_spin]:
            spin.setRange(1, w)
        for spin in [self.vstart_spin, self.vend_spin]:
            spin.setRange(1, h)
        self.hend_spin.setValue(w); self.vend_spin.setValue(h)
        self.analysis_x1_spin.setValue(w); self.analysis_y1_spin.setValue(h)
        if hasattr(self, "spec_x1_spin"):
            self.spec_x1_spin.setValue(w); self.spec_y1_spin.setValue(h)

    def _set_full_roi(self) -> None:
        width, height = (512, 512)
        if self.camera is not None:
            width, height = self.camera.get_identity().detector_shape
        self.hstart_spin.setValue(1); self.hend_spin.setValue(width); self.vstart_spin.setValue(1); self.vend_spin.setValue(height)
        self.analysis_x0_spin.setValue(0); self.analysis_x1_spin.setValue(width); self.analysis_y0_spin.setValue(0); self.analysis_y1_spin.setValue(height)
        if hasattr(self, "spec_x0_spin"):
            self.spec_x0_spin.setValue(0); self.spec_x1_spin.setValue(width); self.spec_y0_spin.setValue(0); self.spec_y1_spin.setValue(height)

    def _set_center_roi(self) -> None:
        width, height = (512, 512)
        if self.camera is not None:
            width, height = self.camera.get_identity().detector_shape
        roi_w = min(128, width); roi_h = min(128, height)
        h0 = width // 2 - roi_w // 2 + 1
        v0 = height // 2 - roi_h // 2 + 1
        self.hstart_spin.setValue(h0); self.hend_spin.setValue(h0 + roi_w - 1)
        self.vstart_spin.setValue(v0); self.vend_spin.setValue(v0 + roi_h - 1)
        self.analysis_x0_spin.setValue(0); self.analysis_x1_spin.setValue(roi_w); self.analysis_y0_spin.setValue(0); self.analysis_y1_spin.setValue(roi_h)

    # ------------------------------------------------------------- Acquisition
    def capture_single_frame(self) -> None:
        if self.camera is None:
            return
        try:
            if self.live_running:
                self.stop_live()
            self._command_detection_shutter(True, reason="single frame")
            if not self.apply_camera_settings(AcquisitionMode.SINGLE_SCAN):
                return
            timeout_ms = max(30000, int(self.exposure_spin.value() * 3000 + 10000))
            frame = self.camera.acquire_single_frame(timeout_ms=timeout_ms)
            self._ingest_raw_frame(frame, source="single")
            self._log("Single frame acquired")
            self._command_detection_shutter(False, reason="single frame complete")
        except BaseException as exc:
            self._show_error("Single-frame acquisition failed", exc)

    def start_live(self) -> None:
        if self.camera is None:
            return
        try:
            self._command_detection_shutter(True, reason="live start")
            if not self.apply_camera_settings(AcquisitionMode.RUN_TILL_ABORT):
                return
            self.camera.start_continuous()
            self.live_running = True
            self.frame_counter = 0
            self._last_live_time = time.monotonic()
            self.live_timer.start()
            self._set_live_ui(True)
            if hasattr(self, "_live_indicator"):
                self._live_indicator.setVisible(True)
            self._log("Live acquisition started")
            self._update_header_status()
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
                self._log(f"Stop live warning: {exc}")
        was = self.live_running
        self.live_running = False
        if hasattr(self, "_live_indicator"):
            self._live_indicator.setVisible(False)
        if self.camera is not None:
            self._set_live_ui(False)
        if was:
            self._command_detection_shutter(False, reason="live stop")
            self._log("Live acquisition stopped")
            self._update_header_status()

    def _update_live_frame(self) -> None:
        if self.camera is None or not self.live_running:
            return
        try:
            frame = self.camera.get_latest_frame()
            self._ingest_raw_frame(frame, source="live")
            self.frame_counter += 1
            now = time.monotonic()
            dt = now - self._last_live_time
            if dt >= 1.0:
                self.live_fps = self.frame_counter / dt
                self.frame_counter = 0
                self._last_live_time = now
                self._update_header_status()
        except BaseException as exc:
            self._log(f"Live frame warning: {exc}")

    def acquire_sequence(self) -> None:
        if self.camera is None:
            return
        try:
            if self.live_running:
                self.stop_live()
            self._command_detection_shutter(True, reason="sequence start")
            if not self.apply_camera_settings(AcquisitionMode.SINGLE_SCAN):
                return
            self.sequence_progress.setValue(0)
            self.sequence_progress.setMaximum(int(self.sequence_n_spin.value()))
            timeout_ms = max(30000, int(self.exposure_spin.value() * 3000 + 10000))
            self._sequence_thread = QtCore.QThread(self)
            self._sequence_worker = SequenceWorker(
                camera=self.camera,
                n_frames=int(self.sequence_n_spin.value()),
                delay_s=float(self.sequence_delay_spin.value()),
                timeout_ms=timeout_ms,
                calibration=self.calibration,
                correction_flags=self._correction_flags(),
                photon_settings=self._photon_settings(),
                use_sdk_pc=bool(self.pc_use_sdk_check.isChecked()),
            )
            self._sequence_worker.moveToThread(self._sequence_thread)
            self._sequence_thread.started.connect(self._sequence_worker.run)
            self._sequence_worker.frame_ready.connect(lambda frame, idx: self._ingest_raw_frame(frame, source=f"sequence {idx}"))
            self._sequence_worker.progress.connect(self._on_sequence_progress)
            self._sequence_worker.finished.connect(self._on_sequence_finished)
            self._sequence_worker.error.connect(self._on_sequence_error)
            self._sequence_worker.finished.connect(self._sequence_thread.quit)
            self._sequence_worker.error.connect(self._sequence_thread.quit)
            self._sequence_thread.finished.connect(self._sequence_worker.deleteLater)
            self._sequence_thread.finished.connect(self._sequence_thread.deleteLater)
            self._sequence_thread.finished.connect(lambda: self._set_sequence_ui(False))
            self._set_sequence_ui(True)
            self.status_timer.stop()
            self._sequence_thread.start()
            self._log(f"Sequence acquisition started: {self.sequence_n_spin.value()} frames")
        except BaseException as exc:
            self._set_sequence_ui(False)
            self.status_timer.start()
            self._show_error("Sequence acquisition failed", exc)

    def abort_sequence(self) -> None:
        if self._sequence_worker is not None:
            self._sequence_worker.abort()
            self._command_detection_shutter(False, reason="sequence abort")
            self._log("Sequence abort requested")

    def _on_sequence_progress(self, idx: int, total: int) -> None:
        self.sequence_progress.setMaximum(total)
        self.sequence_progress.setValue(idx)

    def _on_sequence_error(self, message: str) -> None:
        self.status_timer.start()
        self._command_detection_shutter(False, reason="sequence error")
        self._show_error("Sequence error", message)

    def _on_sequence_finished(self, raw_obj: object, processed_obj: object, counted_obj: object) -> None:
        self.status_timer.start()
        try:
            raw = np.asarray(raw_obj)
            processed = None if processed_obj is None else np.asarray(processed_obj)
            counted = None if counted_obj is None else np.asarray(counted_obj)
            metadata = self._metadata(extra={"sequence_frames": int(raw.shape[0]), "calibration": self.calibration.to_dict(), "photon_counting": self._photon_settings().to_dict()})
            root = self._current_output_dir()
            if self.session is not None:
                path = self.session.next_path("raw", "sequence", ".h5" if self.save_hdf5_radio.isChecked() else ".npz")
            else:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = root / f"{stamp}_sequence" / ("sequence.h5" if self.save_hdf5_radio.isChecked() else "sequence.npz")
            if self.save_hdf5_radio.isChecked():
                saved = write_hdf5_stack(path, raw, metadata=metadata, processed_frames=processed, photon_counted_frames=counted)
            else:
                saved = write_npz_stack(path, raw, metadata=metadata, processed_frames=processed, photon_counted_frames=counted)
            self._command_detection_shutter(False, reason="sequence complete")
            self._log(f"Sequence saved: {saved}")
        except BaseException as exc:
            self._command_detection_shutter(False, reason="sequence save error")
            self._show_error("Sequence save failed", exc)

    # ------------------------------------------------------------- Calibration
    def acquire_calibration(self, kind: str) -> None:
        if self.camera is None:
            return
        try:
            if self.live_running:
                self.stop_live()
            if kind.lower() in {"bias", "dark"}:
                self._command_detection_shutter(False, reason=f"{kind} calibration")
            else:
                self._command_detection_shutter(True, reason=f"{kind} calibration")
            if not self.apply_camera_settings(AcquisitionMode.SINGLE_SCAN):
                return
            n = int(self.calibration_n_spin.value())
            frames = []
            timeout_ms = max(30000, int(self.exposure_spin.value() * 3000 + 10000))
            self._log(f"Acquiring {kind}: {n} frames")
            for idx in range(n):
                frame = self.camera.acquire_single_frame(timeout_ms=timeout_ms)
                frames.append(frame)
                if idx == n - 1:
                    self._ingest_raw_frame(frame, source=f"{kind} last")
                QtWidgets.QApplication.processEvents()
            stack = np.stack(frames, axis=0)
            cal = make_calibration_frame(kind, stack, metadata=self._metadata(extra={"calibration_kind": kind, "n_frames": n}))
            setattr(self.calibration, kind, cal)
            self._update_calibration_label()
            self._log(f"Calibration acquired: {kind}, mean={float(cal.frame.mean()):.3f}, std={float(cal.frame.std()):.3f}")
            if self.auto_shutter_check.isChecked():
                self._command_detection_shutter(False, reason=f"{kind} calibration complete")
        except BaseException as exc:
            if self.auto_shutter_check.isChecked():
                self._command_detection_shutter(False, reason=f"{kind} calibration error")
            self._show_error(f"Acquire {kind} failed", exc)

    def run_quick_auto_calibration(self) -> None:
        """Acquire shutter-closed bias/dark frames and suggest a photon-counting threshold."""
        if self.camera is None:
            return
        try:
            if self.live_running:
                self.stop_live()
            old_internal = self._combo_int(self.internal_shutter_combo)
            old_external = self._combo_int(self.external_shutter_combo)
            old_exposure = float(self.exposure_spin.value())
            self._log("Quick auto-calibration started: closing shutters")
            self._set_combo_data(self.internal_shutter_combo, int(ShutterMode.PERMANENTLY_CLOSED))
            self._set_combo_data(self.external_shutter_combo, int(ShutterMode.PERMANENTLY_CLOSED))
            self.apply_shutter(show_errors=False)

            # Bias: shortest supported exposure requested by the GUI.
            n_bias = int(self.auto_cal_bias_n_spin.value())
            n_dark = int(self.auto_cal_dark_n_spin.value())
            self.exposure_spin.setValue(1e-6)
            if not self.apply_camera_settings(AcquisitionMode.SINGLE_SCAN):
                return
            bias_frames = []
            timeout_ms = 30000
            self._log(f"Quick auto-calibration: acquiring {n_bias} bias frames")
            for idx in range(n_bias):
                frame = self.camera.acquire_single_frame(timeout_ms=timeout_ms)
                bias_frames.append(frame)
                if idx == n_bias - 1:
                    self._ingest_raw_frame(frame, source="quick auto-cal bias last")
                QtWidgets.QApplication.processEvents()

            # Dark: restore experiment exposure with shutter still closed.
            self.exposure_spin.setValue(old_exposure)
            if not self.apply_camera_settings(AcquisitionMode.SINGLE_SCAN):
                return
            dark_frames = []
            timeout_ms = max(30000, int(old_exposure * 3000 + 10000))
            self._log(f"Quick auto-calibration: acquiring {n_dark} dark frames at {old_exposure:.6g} s")
            for idx in range(n_dark):
                frame = self.camera.acquire_single_frame(timeout_ms=timeout_ms)
                dark_frames.append(frame)
                if idx == n_dark - 1:
                    self._ingest_raw_frame(frame, source="quick auto-cal dark last")
                QtWidgets.QApplication.processEvents()

            self.calibration.bias = make_calibration_frame("bias", np.stack(bias_frames, axis=0), metadata=self._metadata(extra={"calibration_kind": "bias", "quick_auto": True, "n_frames": n_bias}))
            self.calibration.dark = make_calibration_frame("dark", np.stack(dark_frames, axis=0), metadata=self._metadata(extra={"calibration_kind": "dark", "quick_auto": True, "n_frames": n_dark, "exposure_s": old_exposure}))
            self.subtract_dark_check.setChecked(True)
            self.clip_negative_check.setChecked(False)

            ident = self.camera.get_identity()
            profile = build_auto_calibration_profile(
                bias=self.calibration.bias,
                dark=self.calibration.dark,
                camera_model=ident.model,
                serial_number=ident.serial_number,
                threshold_sigma=float(self.auto_threshold_sigma_spin.value()),
                metadata=self._metadata(extra={"quick_auto_calibration": True}),
                timestamp_unix_s=time.time(),
            )
            self.auto_calibration_profile = profile
            if profile.suggested_threshold_adu is not None:
                self.pc_t1_spin.setValue(float(max(profile.suggested_threshold_adu, 0.0)))
                self.pc_enable_check.setChecked(True)
                self.pc_use_corrected_check.setChecked(True)

            root = self._current_output_dir()
            cal_dir = root / "calibration" if self.session is not None else root / "calibration"
            cal_dir.mkdir(parents=True, exist_ok=True)
            save_auto_calibration_profile(cal_dir / f"{ident.model}_serial_{ident.serial_number}_quick_auto_calibration.json", profile)
            self.save_calibration_frames()
            self._update_calibration_label()
            self.auto_cal_label.setText(
                "Quick auto-calibration complete.\n"
                f"bias mean={profile.bias_mean_adu:.3f} ADU | dark mean={profile.dark_mean_adu:.3f} ADU\n"
                f"dark sigma={profile.corrected_dark_sigma_adu:.3f} ADU | suggested threshold={profile.suggested_threshold_adu:.3f} ADU\n"
                f"false dark fraction={profile.false_event_fraction_dark:.3e} | hot pixels={profile.hot_pixel_count}"
            )
            self._log("Quick auto-calibration complete")
        except BaseException as exc:
            self._show_error("Quick auto-calibration failed", exc)
        finally:
            # Restore user shutter choice and exposure.
            try:
                self.exposure_spin.setValue(old_exposure)
                self._set_combo_data(self.internal_shutter_combo, old_internal)
                self._set_combo_data(self.external_shutter_combo, old_external)
                self.apply_shutter(show_errors=False)
                self.apply_camera_settings(AcquisitionMode.SINGLE_SCAN, show_errors=False)
            except Exception as exc:
                self._log(f"Quick auto-calibration restore warning: {exc}")

    def save_calibration_frames(self) -> None:
        try:
            from andor_control.analysis.calibration import save_calibration_frame
            root = self._current_output_dir()
            cal_dir = root / "calibration" if self.session is not None else root
            if self.calibration.bias is not None:
                save_calibration_frame(cal_dir / "bias_calibration.npz", self.calibration.bias)
            if self.calibration.dark is not None:
                save_calibration_frame(cal_dir / "dark_calibration.npz", self.calibration.dark)
            if self.calibration.background is not None:
                save_calibration_frame(cal_dir / "background_calibration.npz", self.calibration.background)
            save_metadata_json(cal_dir / "calibration_metadata.json", self.calibration.to_dict())
            self._log(f"Calibration saved in {cal_dir}")
        except BaseException as exc:
            self._show_error("Save calibration failed", exc)

    def _update_calibration_label(self) -> None:
        available = self.calibration.available()
        self.calibration_label.setText("Calibration available: " + (", ".join(available) if available else "none"))

    # ---------------------------------------------------------- Processing/save
    def _ingest_raw_frame(self, frame: np.ndarray, *, source: str) -> None:
        self.last_raw = np.asarray(frame, dtype=np.int32)
        self.last_processed = apply_calibration(
            self.last_raw,
            self.calibration,
            subtract_bias=bool(self.subtract_bias_check.isChecked()),
            subtract_dark=bool(self.subtract_dark_check.isChecked()),
            subtract_background=bool(self.subtract_background_check.isChecked()),
            clip_negative=bool(self.clip_negative_check.isChecked()),
        )
        if self.pc_enable_check.isChecked():
            self._run_photon_counting(show_errors=False)
        self._display_current_frame()
        self._update_stats(source=source)
        self.save_last_button.setEnabled(True)

    def _display_current_frame(self) -> None:
        if self.last_raw is None:
            return
        # Backward-compatible checkboxes from milestone 6: if the user did not
        # explicitly choose another display unit, these checkboxes can still
        # select corrected or photon-counted views.
        if hasattr(self, "display_unit_combo"):
            current = str(self.display_unit_combo.currentText())
            if current == "Raw ADU":
                if self.show_pc_check.isChecked():
                    self.display_unit_combo.setCurrentText("Photon-counted events")
                    current = "Photon-counted events"
                elif self.show_processed_check.isChecked():
                    self.display_unit_combo.setCurrentText("Corrected ADU")
                    current = "Corrected ADU"
            if current == "Photon-counted events" and self.last_photon_counted is None:
                self._run_photon_counting(show_errors=False)
        try:
            result = convert_display_frame(
                raw=self.last_raw,
                corrected=self.last_processed,
                photon_counted=self.last_photon_counted,
                settings=self._unit_settings(),
                sdk=self.camera.sdk if self.camera is not None else None,
                metadata=self._metadata(extra={}),
            )
            frame = result.frame
            display_name = result.unit_label
            self.last_unit_metadata = result.metadata | {"method": result.method, "unit_label": result.unit_label}
        except Exception as exc:
            frame = self.last_raw
            display_name = "ADU"
            self.last_unit_metadata = {"method": "raw_fallback", "error": str(exc)}
            self._log(f"Display-unit warning: {exc}")
        arr = np.asarray(frame)
        self.last_display_frame = arr.astype(np.float32, copy=False) if not np.issubdtype(arr.dtype, np.integer) else arr
        auto_levels = bool(self.auto_levels_check.isChecked())
        self.image_view.setImage(arr.T, autoLevels=auto_levels, autoRange=not bool(self.live_running))
        if not auto_levels:
            lo = float(self.display_min_spin.value())
            hi = float(self.display_max_spin.value())
            if hi <= lo:
                hi = lo + 1.0
            try:
                self.image_view.setLevels(lo, hi)
            except TypeError:
                self.image_view.setLevels((lo, hi))
            self.display_scale_label.setText(f"Scale: {lo:.3g}–{hi:.3g}")
        else:
            self.display_scale_label.setText("Scale: Auto")
        self._update_histogram(arr)
        self._update_spectroscopy(arr)
        if hasattr(self, "display_mode_label"):
            mode_text = str(self.display_unit_combo.currentText()) if hasattr(self, "display_unit_combo") else display_name
            self.display_mode_label.setText(f"Display: {mode_text} [{display_name}]")
            self.frame_shape_label.setText(f"Frame: {arr.shape[1]} × {arr.shape[0]} | {arr.dtype}")
            sat = int(np.count_nonzero(arr >= 16383)) if np.issubdtype(arr.dtype, np.integer) else 0
            self.saturation_label.setText(f"Saturation: {sat} px")

    def _update_histogram(self, frame: np.ndarray) -> None:
        try:
            self.hist_plot.clear()
            arr = np.asarray(frame).ravel()
            if arr.size == 0:
                return
            lo, hi = np.percentile(arr, [0.5, 99.5])
            if hi <= lo:
                lo, hi = float(arr.min()), float(arr.max() + 1)
            hist, edges = np.histogram(arr, bins=120, range=(lo, hi))
            centers = 0.5 * (edges[:-1] + edges[1:])
            pen = pg.mkPen(color="#1a6fff", width=1.5)
            brush = pg.mkBrush(color=(26, 111, 255, 60))
            self.hist_plot.plot(centers, hist, stepMode=False, pen=pen, fillLevel=0, brush=brush)
            for threshold in self._photon_settings(force_enabled=True).thresholds_adu:
                self.hist_plot.addLine(x=threshold, pen=pg.mkPen(color="#f0a500", width=1, style=QtCore.Qt.PenStyle.DashLine))
        except Exception as exc:
            self._log(f"Histogram warning: {exc}")

    def lock_current_display_scale(self) -> None:
        if self.last_display_frame is None:
            return
        arr = np.asarray(self.last_display_frame, dtype=np.float32)
        if arr.size == 0:
            return
        lo, hi = np.percentile(arr, [0.5, 99.5])
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max() + 1.0)
        self.display_min_spin.setValue(float(lo))
        self.display_max_spin.setValue(float(hi))
        self.auto_levels_check.setChecked(False)
        self._display_current_frame()

    def reset_auto_display_scale(self) -> None:
        self.auto_levels_check.setChecked(True)
        self._display_current_frame()

    def fit_image_view(self) -> None:
        try:
            self.image_view.view.autoRange()
        except Exception:
            pass

    def one_to_one_image_view(self) -> None:
        try:
            self.image_view.view.setAspectLocked(True)
            self.image_view.view.autoRange()
        except Exception:
            pass

    def _update_spectroscopy(self, displayed_frame: np.ndarray) -> None:
        try:
            self.spectroscopy_plot.clear()
            if not self.spectroscopy_enable_check.isChecked():
                self.spectroscopy_status_label.setText("Spectroscopy: off")
                return
            source = np.asarray(displayed_frame if self.spec_use_displayed_check.isChecked() else (self.last_processed if self.last_processed is not None else self.last_raw), dtype=np.float32)
            if source is None or source.size == 0:
                return
            h, w = source.shape
            x0 = max(0, min(int(self.spec_x0_spin.value()), w - 1))
            x1 = max(x0 + 1, min(int(self.spec_x1_spin.value()), w))
            y0 = max(0, min(int(self.spec_y0_spin.value()), h - 1))
            y1 = max(y0 + 1, min(int(self.spec_y1_spin.value()), h))
            roi = source[y0:y1, x0:x1]
            profile = roi.sum(axis=0)
            x = np.arange(x0, x1)
            self.spectroscopy_plot.plot(x, profile, pen=pg.mkPen(color="#00c8d4", width=1.5))
            self.spectroscopy_status_label.setText(f"Spectroscopy: x={x0}:{x1}, y={y0}:{y1}, sum={float(profile.sum()):.4g}")
        except Exception as exc:
            self._log(f"Spectroscopy warning: {exc}")

    def _update_stats(self, *, source: str) -> None:
        if self.last_raw is None:
            return
        raw_stats = frame_statistics(self.last_raw)
        proc_stats = frame_statistics(self.last_processed) if self.last_processed is not None else None
        roi_raw = roi_frame(
            self.last_raw,
            self.analysis_x0_spin.value(),
            self.analysis_x1_spin.value(),
            self.analysis_y0_spin.value(),
            self.analysis_y1_spin.value(),
        )
        roi_stats = frame_statistics(roi_raw)

        def fmt_block(title: str, data: dict[str, Any]) -> list[str]:
            preferred = ["shape", "dtype", "min", "max", "mean", "median", "std", "sum", "saturated_pixels"]
            lines = [title]
            for key in preferred:
                if key in data:
                    lines.append(f"  {key:<18}: {data[key]}")
            for key, value in data.items():
                if key not in preferred:
                    lines.append(f"  {key:<18}: {value}")
            return lines

        lines = [
            f"SOURCE                 : {source}",
            f"LIVE FPS               : {self.live_fps:.2f}",
            f"DISPLAY MODE           : {self.display_mode_label.text().replace('Display: ', '') if hasattr(self, 'display_mode_label') else 'raw ADU'}",
            "",
        ]
        lines += fmt_block("RAW FRAME", raw_stats.to_dict())
        lines += [""]
        lines += fmt_block("ANALYSIS ROI - RAW", roi_stats.to_dict())
        if proc_stats is not None:
            lines += [""]
            lines += fmt_block("CORRECTED / PROCESSED FRAME", proc_stats.to_dict())
        if self.last_pc_metadata is not None:
            lines += ["", "PHOTON COUNTING"]
            for key, value in self.last_pc_metadata.items():
                lines.append(f"  {key:<28}: {value}")
        self.stats_text.setPlainText("\n".join(lines))

    def process_current_photon_counting(self) -> None:
        self._run_photon_counting(show_errors=True)
        self._display_current_frame()
        self._update_stats(source="photon_counting")

    def _run_photon_counting(self, *, show_errors: bool) -> bool:
        if self.last_raw is None:
            return False
        try:
            settings = self._photon_settings(force_enabled=True)
            source = self.last_processed if settings.use_corrected_frame and self.last_processed is not None else self.last_raw
            result = photon_count_frame(source, settings, sdk=self.camera.sdk if self.camera is not None else None)
            self.last_photon_counted = result.counted_frame
            self.last_pc_metadata = result.to_metadata()
            self.pc_label.setText(
                f"Photon-counting result:\n"
                f"total_events={result.total_events}\n"
                f"active_pixels={result.active_pixels}\n"
                f"active_fraction={result.active_fraction:.6g}\n"
                f"mean_events_per_pixel={result.mean_events_per_pixel:.6g}\n"
                f"max_events_per_pixel={result.max_events_per_pixel}"
            )
            return True
        except BaseException as exc:
            if show_errors:
                self._show_error("Photon-counting failed", exc)
            else:
                self._log(f"Photon-counting warning: {exc}")
            return False

    def save_current_frame(self) -> None:
        if self.last_raw is None:
            return
        try:
            root = self._current_output_dir()
            if self.session is not None:
                raw_path = self.session.next_path("raw", "single_raw", ".npy")
                meta_path = raw_path.with_suffix(".json")
            else:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                raw_path = root / f"{stamp}_single_raw.npy"
                meta_path = root / f"{stamp}_single_raw.json"
            save_frame_npy(raw_path, self.last_raw)
            if self.last_processed is not None:
                if self.session is not None:
                    processed_path = self.session.root / "processed" / raw_path.name.replace("raw", "processed")
                else:
                    processed_path = raw_path.with_name(raw_path.stem.replace("raw", "processed") + ".npy")
                save_frame_npy(processed_path, self.last_processed)
            if self.last_photon_counted is not None:
                pc_path = (self.session.root / "photon_counted" / raw_path.name.replace("raw", "photon_counted")) if self.session else raw_path.with_name(raw_path.stem.replace("raw", "photon_counted") + ".npy")
                save_frame_npy(pc_path, self.last_photon_counted)
            save_metadata_json(meta_path, self._metadata(extra={"frame_statistics": frame_statistics(self.last_raw).to_dict(), "photon_counting_result": self.last_pc_metadata}))
            self._log(f"Saved current frame: {raw_path}")
        except BaseException as exc:
            self._show_error("Save current frame failed", exc)

    def save_current_photon_counted(self) -> None:
        if self.last_photon_counted is None:
            if not self._run_photon_counting(show_errors=True):
                return
        try:
            assert self.last_photon_counted is not None
            root = self._current_output_dir()
            if self.session is not None:
                path = self.session.next_path("photon_counted", "current_photon_counted", ".npy")
            else:
                path = root / f"{time.strftime('%Y%m%d_%H%M%S')}_current_photon_counted.npy"
            save_frame_npy(path, self.last_photon_counted)
            save_metadata_json(path.with_suffix(".json"), self._metadata(extra={"photon_counting_result": self.last_pc_metadata}))
            self._log(f"Saved photon-counted frame: {path}")
        except BaseException as exc:
            self._show_error("Save photon-counted failed", exc)

    def _metadata(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "timestamp_unix_s": time.time(),
            "session": self.session.to_metadata() if self.session is not None else None,
            "capability_summary": self.capability_summary,
            "last_acquisition_info": self.last_acq_info,
            "calibration": self.calibration.to_dict(),
            "correction_flags": self._correction_flags(),
            "photon_counting_settings": self._photon_settings().to_dict(),
            "unit_conversion_settings": self._unit_settings().to_dict() if hasattr(self, "display_unit_combo") else None,
            "last_display_unit_metadata": self.last_unit_metadata,
            "auto_calibration_profile": self.auto_calibration_profile.to_dict() if self.auto_calibration_profile is not None else None,
            "shutter_commanded": {
                "mode": self.shutter_commanded_mode,
                "state": self.shutter_commanded_state,
                "automatic_during_detection": bool(self.auto_shutter_check.isChecked()) if hasattr(self, "auto_shutter_check") else None,
            },
            "em_gain_request": {
                "requested_gain": self.requested_em_gain,
                "applied_sdk_gain": self.applied_em_gain,
                "note": "Requested gain may be clamped to the SDK minimum in REAL_GAIN mode.",
            },
            "display_scale": {
                "auto": bool(self.auto_levels_check.isChecked()) if hasattr(self, "auto_levels_check") else True,
                "manual_min": float(self.display_min_spin.value()) if hasattr(self, "display_min_spin") else None,
                "manual_max": float(self.display_max_spin.value()) if hasattr(self, "display_max_spin") else None,
            },
            "spectroscopy_roi": {
                "enabled": bool(self.spectroscopy_enable_check.isChecked()) if hasattr(self, "spectroscopy_enable_check") else False,
                "x0": int(self.spec_x0_spin.value()) if hasattr(self, "spec_x0_spin") else None,
                "x1": int(self.spec_x1_spin.value()) if hasattr(self, "spec_x1_spin") else None,
                "y0": int(self.spec_y0_spin.value()) if hasattr(self, "spec_y0_spin") else None,
                "y1": int(self.spec_y1_spin.value()) if hasattr(self, "spec_y1_spin") else None,
                "vertical_sum": "S(x)=sum_y I(x,y) inside Spectroscopy ROI",
            },
        }
        if self.camera is not None:
            try:
                meta["camera_state"] = self.camera.build_metadata()
            except Exception as exc:
                meta["camera_state_error"] = str(exc)
        if extra:
            meta.update(extra)
        return meta

    # -------------------------------------------------------------- Status
    def _refresh_status(self) -> None:
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
            self.camera_status_label.setText(f"SDK status: {status}\nTotal images: {total} | Circular buffer: {buf}\nLive: {self.live_running}")
            self._update_header_status()
        except BaseException as exc:
            self._log(f"Status warning: {exc}")

    def _update_temperature_label(self, state: dict[str, Any]) -> None:
        self.temp_label.setText(f"Temperature: {state['temperature_c']:.2f} °C\nStatus: {state['status']} | Cooler: {'ON' if state['cooler_on'] else 'OFF'}")
        if hasattr(self, "temperature_tile"):
            status = str(state.get("status", "--"))
            temp_text = f"{state['temperature_c']:.1f} °C · {status}"
            self.temperature_tile.value_label.setText(temp_text)  # type: ignore[attr-defined]
            if status in {"stable", "stabilized", "reached"}:
                tile_state = "Good"
            elif status in {"not_reached", "drift", "not stabilized"}:
                tile_state = "Warning"
            else:
                tile_state = "Info" if state.get("cooler_on") else "Neutral"
            self._set_tile_state(self.temperature_tile, tile_state)

    def _reset_splitter_sizes(self) -> None:
        """Set splitter proportions relative to current window dimensions."""
        total_w = self.main_splitter.width() or self.width()
        ctrl_w = max(320, min(int(total_w * 0.28), 560))
        self.main_splitter.setSizes([ctrl_w, total_w - ctrl_w])

        total_h = self.center_splitter.height() or self.height()
        img_h = int(total_h * 0.74)
        self.center_splitter.setSizes([img_h, total_h - img_h])

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # Defer sizing until the window has been fully laid out by Qt
        QtCore.QTimer.singleShot(0, self._reset_splitter_sizes)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self.controls_hidden and not self.clean_detection_mode:
            self._reset_splitter_sizes()

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self.disconnect_camera()
        finally:
            event.accept()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scientific acquisition GUI for Andor iXon 897")
    parser.add_argument("--dll", default=None, help="Path to atmcd64d.dll. If omitted, normal SDK DLL lookup is used.")
    parser.add_argument("--mock", action="store_true", help="Use simulated camera backend instead of real hardware")
    parser.add_argument("--output", default="andor_data", help="Base directory where sessions/data are written")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    window = ScientificAcquisitionWindow(dll_path=args.dll, mock=args.mock, output_dir=args.output)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
