from drone.mapping.survey_planner import SurveyPlanner, validate_geojson
def test_survey():
    gj = {"type":"Polygon","coordinates":[[[77.0,12.0],[77.01,12.0],[77.01,12.01],[77.0,12.01],[77.0,12.0]]]}
    ok,_ = validate_geojson(gj)
    assert ok
    out = SurveyPlanner(20,5,0.7).generate(gj)
    assert len(out["waypoints"]) >= 2 and out["distance_m"] > 0
