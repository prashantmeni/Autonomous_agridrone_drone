"""Hardware gates: skipped unless --hardware."""
import pytest
pytestmark = pytest.mark.hardware
def test_pixhawk_present():
    import os
    assert os.path.exists(os.environ.get("MAVLINK_CONNECTION","/dev/ttyACM0"))
