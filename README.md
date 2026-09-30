# Andor iXon 897 Control — Professional Scientific Acquisition GUI

This package provides a clean Python control layer for an **Andor iXon 897 EMCCD** using the **Andor SDK2 / ATMCD DLL** and includes two GUI launchers:

1. `scripts/minimal_gui.py` — the validated diagnostic GUI from the previous milestone.
2. `scripts/scientific_gui.py` — the new scientific acquisition GUI with sessions, calibration, sequence saving and photon-counting post-processing.

The project does **not** bundle the vendor DLL. Keep `atmcd64d.dll` installed through the Andor Driver Pack / Solis / SDK2 installation, or pass the DLL path explicitly.

## Main additions in milestone 6

- Professional session manager with experiment name, sample, operator, notes and automatic folders.
- Organized output structure: `raw`, `processed`, `photon_counted`, `calibration`, `metadata`, `logs`, `exports`.
- Single-frame acquisition, live view and sequence acquisition.
- HDF5 stack saving for sequences, with NPZ fallback.
- Complete metadata saved with camera state, temperature, shutter, trigger, ROI, EMCCD/readout settings and photon-counting thresholds.
- Bias/dark/background acquisition by frame averaging.
- Optional bias/dark/background subtraction before display and photon counting.
- Photon-counting post-processing by threshold.
- Optional use of the Andor SDK2 `PostProcessPhotonCounting` helper when available.
- Raw data is preserved even when corrected or photon-counted frames are generated.
- ROI statistics, frame histogram and photon-counting summary.


## Professional layout update in milestone 6

The scientific GUI was reorganized to look and behave more like a laboratory acquisition console:

- Professional dark theme with consistent cards, groups, controls and typography.
- Top status header for camera connection, detector temperature, acquisition state and active session.
- Left control console for session, camera, acquisition, calibration and photon-counting panels.
- Large central detector image area with display mode, frame shape and saturation summary.
- Bottom inspector tabs for histogram, frame statistics and session log.
- Cleaner, human-readable statistics panel instead of raw JSON-only output.
- Status-bar feedback for recent actions and warnings.

The underlying acquisition, calibration, photon-counting and saving logic is kept compatible with milestone 5.

## Installation

From the package directory:

```bat
py -3.11 -m pip install -e .
```

For the graphical interfaces and HDF5 support:

```bat
py -3.11 -m pip install -e .[gui]
```

Alternative:

```bat
py -3.11 -m pip install -r requirements-gui.txt
```

For tests:

```bat
py -3.11 -m pip install -r requirements.txt
py -3.11 -m pytest
```

## Test without camera

```bat
py -3.11 scripts\diagnose_ixon897.py --mock --output out_mock
py -3.11 scripts\minimal_gui.py --mock
py -3.11 scripts\scientific_gui.py --mock
```

## Real camera diagnostic

Close Andor Solis, Micro-Manager, or any other program that may be using the camera.

```bat
py -3.11 scripts\diagnose_ixon897.py --dll "C:\Path\To\atmcd64d.dll" --output out_real --exposure 0.03 --temperature -70
```

## Diagnostic GUI

```bat
py -3.11 scripts\minimal_gui.py --dll "C:\Users\Usuario\Documents\New_andor_test\andor_ixon897_control\atmcd64d.dll" --output gui_output
```

Or:

```bat
run_minimal_gui.bat --dll "C:\Users\Usuario\Documents\New_andor_test\andor_ixon897_control\atmcd64d.dll" --output gui_output
```

## Scientific GUI

Mock mode:

```bat
py -3.11 scripts\scientific_gui.py --mock --output andor_data
```

Real camera:

```bat
py -3.11 scripts\scientific_gui.py --dll "C:\Users\Usuario\Documents\New_andor_test\andor_ixon897_control\atmcd64d.dll" --output andor_data
```

Or:

```bat
run_scientific_gui.bat --dll "C:\Users\Usuario\Documents\New_andor_test\andor_ixon897_control\atmcd64d.dll" --output andor_data
```

Recommended first use:

1. Create a session in the **Session** tab.
2. Connect the camera in **Camera Control**.
3. Keep EM gain at the SDK minimum until the camera is cold/stable.
4. Use **Force open both** only when it is optically safe.
5. Capture a single frame and verify signal/baseline.
6. Acquire bias/dark/background in the **Calibration** tab if needed.
7. Enable correction flags only after calibration frames exist.
8. Use **Photon Counting** with a threshold chosen from the histogram.
9. Acquire and save a sequence in the **Acquisition** tab.

## Photon counting interpretation

The photon-counting panel implements threshold-based event counting:

- Single threshold: pixels below the threshold are 0, pixels above it are 1.
- Multiple thresholds: thresholds `[T1, T2, T3]` produce counts 0, 1, 2 or 3 depending on the signal range.

This is **photon-counting post-processing**, not the same as SDK Count Convert. Always save raw frames together with photon-counted frames so thresholds can be changed later without repeating the experiment.

## Safety notes

- The default EM gain is the minimum valid SDK gain for the selected gain mode.
- The high-level API refuses to set EM gain above the SDK minimum unless the temperature is stabilized, unless explicitly overridden.
- The cooler is **not** forcibly turned off during shutdown by default.
- Acquisition settings cannot be changed while the SDK reports that the camera is acquiring.
- Use high EM gain only in controlled low-light conditions.


## Milestone 7 — Clean Detection View + Calibration Assistant + Scientific Units

This version deliberately starts from the stable milestone 6 professional layout. It adds:

- collapsible left control panel;
- Clean Detection View for a larger detector image during alignment/detection;
- display-unit selector: Raw ADU, Corrected ADU, Photon-counted events, SDK estimated electrons, SDK estimated photons, Manual estimated electrons and Manual estimated photons;
- detection wavelength field used by SDK/manual photon estimates;
- Quick Auto Calibration that closes the shutter, acquires bias/dark stacks, builds master calibration frames, estimates dark noise, detects hot-pixel statistics and suggests an initial photon-counting threshold;
- saved quick auto-calibration profile tied to camera model/serial;
- improved metadata recording for unit conversion and calibration state.

Important: SDK/manual electrons and photons are estimates. Raw ADU and corrected ADU remain the primary camera units. Photon-counted events are threshold-based events per pixel per frame, not an absolute measurement of photons emitted by the sample.

## Milestone 8 — Live Controls, Auto Shutter, Corrected Units and Spectroscopy ROI

This release is based on the stable `milestone7_clean_units_autocal_v2` code line.

Main changes:

- Automatic shutter during detection is enabled by default: the software commands the shutter open when live/single/sequence detection starts and commands it closed when detection stops.
- A shutter status panel reports the commanded state: `AUTO/MANUAL` and `OPEN/CLOSED/AUTO/UNKNOWN`.
- The EM gain control accepts user request `1`. If the Andor SDK minimum in the selected gain mode is higher, the software clamps to the SDK-safe value and records both the requested and applied gain in the GUI and metadata.
- Display scale can be automatic or manually fixed with user-selected min/max values.
- Image view supports zoom/pan through pyqtgraph plus Fit/Reset/1:1 buttons.
- Corrected ADU processing now uses float data, validates calibration-frame shape, allows negative corrected values, and avoids silent display blanking.
- Background subtraction now treats background as an experimental light-background frame and avoids double-subtracting bias/dark when those corrections are enabled.
- Added a spectroscopy ROI with vertical sum profile: `S(x)=sum_y I(x,y)`.
- Live parameter changes are auto-applied through a safe pause/reconfigure/restart cycle while live acquisition is running.

Recommended workflow:

1. Connect the camera.
2. Keep shutter mode automatic unless diagnosing the shutter manually.
3. Start live detection; the shutter should open automatically.
4. Stop live detection; the shutter should close automatically.
5. Acquire dark/background calibrations with the same ROI/binning used for measurements.
6. Use manual display scale when comparing signal levels between frames.
7. Enable spectroscopy ROI to monitor the horizontal spectral profile.

## Milestone 9 — Instrument Control Ergonomics

This update keeps the Milestone 8 camera/acquisition logic and improves the left **Instrument Control** panel:

- Horizontal scrolling is disabled in all control tabs.
- The side panel is width-limited and uses compact controls.
- Main actions are available in a compact quick-action strip: Connect, Single, Live, Stop and Save.
- Control tabs are now vertical to avoid top-tab overflow.
- A Basic/Advanced selector hides rarely used controls during normal operation.
- Long labels and buttons were shortened while preserving tooltips/log behaviour.
- Advanced sections such as readout, camera ROI, sequence, calibration and photon-counting can be hidden in Basic mode.

The update is intended to improve ergonomics without changing the validated camera-control pipeline.
