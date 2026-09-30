from andor_control.camera.ixon897 import AndorIXon897
from andor_control.camera.settings import AcquisitionSettings, EMCCDSettings, ROI, TemperatureSettings
from andor_control.sdk2.enums import AcquisitionMode
from andor_control.sdk2.mock import MockAndorSDK2


def test_mock_single_frame():
    cam = AndorIXon897(MockAndorSDK2())
    try:
        identity = cam.initialize()
        assert identity.detector_shape == (512, 512)
        cam.configure_temperature(TemperatureSettings(cooler_on=False))
        cam.configure_emccd(EMCCDSettings(em_gain=1), allow_high_gain_without_stable_temp=True)
        cam.configure_acquisition(AcquisitionSettings(acquisition_mode=AcquisitionMode.SINGLE_SCAN, roi=ROI(1, 128, 1, 64)))
        frame = cam.acquire_single_frame()
        assert frame.shape == (64, 128)
        assert frame.dtype.kind in {"i", "u"}
    finally:
        cam.shutdown()
