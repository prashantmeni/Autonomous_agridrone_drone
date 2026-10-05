"""Marker for SITL tests (skipped without SITL)."""
import pytest
pytestmark = pytest.mark.simulation
def test_sitl_placeholder():
    pytest.skip("requires PX4 SITL on udp://127.0.0.1:14540")
