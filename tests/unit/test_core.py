from drone.autonomy.geofence import point_in_polygon, Geofence
from drone.navigation.coordinate_transform import haversine_m, polygon_area_m2
from drone.autonomy.waypoint_manager import validate_mission
from drone.safety.flight_safety import battery_status
from drone.perception.obstacle_detection import ObstacleDetector
from drone.core.state import FlightStateMachine, FlightState
import pytest

def test_geofence():
    poly = [(0,0),(0,1),(1,1),(1,0)]
    assert point_in_polygon(0.5,0.5,poly) and not point_in_polygon(2,2,poly)
    g = Geofence(poly)
    ok,_ = g.validate_mission([{"lat":0.5,"lon":0.5},{"lat":5,"lon":5}])
    assert not ok

def test_haversine():
    d = haversine_m(0,0,0,1)
    assert 110000 < d < 112000

def test_area():
    assert polygon_area_m2([(0,0),(0,0.001),(0.001,0.001),(0.001,0)]) > 10000

def test_mission_validation():
    ok,_ = validate_mission({"waypoints":[{"lat":1,"lon":1,"alt":10}]},30,8)
    assert ok
    ok,errs = validate_mission({"waypoints":[{"lat":1,"lon":1,"alt":99}]},30,8)
    assert not ok

def test_battery():
    assert battery_status(10,25,15)=="CRITICAL"
    assert battery_status(20,25,15)=="LOW"
    assert battery_status(80,25,15)=="OK"

def test_obstacle():
    d = ObstacleDetector(warning=5,critical=2,emergency=1)
    assert d.evaluate(0.5)["state"]=="EMERGENCY_STOP"
    assert d.evaluate(10)["state"]=="CLEAR"
    assert d.evaluate(None)["status"]=="OBSTACLE_SENSOR_UNAVAILABLE"

def test_fsm():
    m = FlightStateMachine()
    m.transition(FlightState.PRE_FLIGHT_CHECK,"t")
    with pytest.raises(ValueError):
        m.transition(FlightState.LANDING,"illegal")

def test_disease_no_model():
    from drone.perception import crop_ai
    s = crop_ai.model_status("", False, 0.6)
    assert s["status"]=="MODEL_NOT_AVAILABLE"
