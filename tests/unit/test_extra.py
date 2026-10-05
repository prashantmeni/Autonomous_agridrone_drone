from drone.mapping.kml import kml_to_geojson
from drone.mapping.survey_planner import SurveyPlanner
from drone.autonomy.geofence import Geofence
def test_kml():
    gj = kml_to_geojson("<kml><coordinates>77,12 77.01,12 77.01,12.01 77,12</coordinates></kml>")
    assert gj["type"] == "Polygon"
def test_geofence_clip():
    gj = {"type": "Polygon", "coordinates": [[[77, 12], [77.01, 12], [77.01, 12.01], [77, 12.01], [77, 12]]]}
    out = SurveyPlanner(20, 5, 0.7).generate(gj)
    poly = [(p[1], p[0]) for p in gj["coordinates"][0]]
    g = Geofence(poly)
    inside = [w for w in out["waypoints"] if g.contains(w["lat"], w["lon"])]
    assert len(inside) > 0
def test_db_mission():
    from drone.db.store import Database
    db = Database(":memory:")
    db.save_mission("m1", "test", {"name": "test"})
    assert db.get_mission("m1")["name"] == "test"
