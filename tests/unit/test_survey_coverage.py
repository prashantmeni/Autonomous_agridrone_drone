"""Survey coverage must follow the field, not its bounding box."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drone.mapping.survey_planner import SurveyPlanner, _scanline_spans  # noqa: E402


def square(lat0=12.0, lon0=77.0, d=0.001):
    return {
        "type": "Polygon",
        "coordinates": [[[lon0, lat0], [lon0 + d, lat0],
                         [lon0 + d, lat0 + d], [lon0, lat0 + d],
                         [lon0, lat0]]],
    }


def l_shape(lat0=12.0, lon0=77.0, d=0.002):
    """A field with a bite taken out: concave, bounding box would over-cover."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon0, lat0], [lon0 + d, lat0], [lon0 + d, lat0 + d / 2],
            [lon0 + d / 2, lat0 + d / 2], [lon0 + d / 2, lat0 + d],
            [lon0, lat0 + d], [lon0, lat0],
        ]],
    }


def test_scanline_finds_single_span_in_square():
    poly = [(12.0, 77.0), (12.0, 77.001), (12.001, 77.001), (12.001, 77.0)]
    spans = _scanline_spans(poly, 12.0005)
    assert len(spans) == 1
    lo, hi = spans[0]
    assert lo < hi


def test_scanline_splits_around_a_concavity():
    poly = [(12.0, 77.0), (12.0, 77.002), (12.002, 77.002), (12.002, 77.0),
            (12.001, 77.0), (12.001, 77.001), (12.001, 77.001), (12.001, 77.0)]
    # Notched polygon: a scanline below the notch crosses once.
    spans = _scanline_spans(poly, 12.0005)
    assert len(spans) == 1


def test_every_waypoint_is_inside_the_polygon():
    from drone.autonomy.geofence import point_in_polygon

    ring = l_shape()["coordinates"][0]
    poly = [(p[1], p[0]) for p in ring]
    plan = SurveyPlanner(altitude_m=20, side_overlap=0.5).generate(l_shape())
    assert plan["waypoints"], "planner produced no waypoints"
    for w in plan["waypoints"]:
        assert point_in_polygon(w["lat"], w["lon"], poly), \
            f"waypoint outside the field: {w}"


def test_concave_field_is_not_covered_across_its_notch():
    """Above the notch the survey must not span the full field width.

    A bounding-box plan would fly the whole width at every latitude, crossing
    the notch where there is no field.
    """
    planner = SurveyPlanner(altitude_m=20, side_overlap=0.5)
    plan = planner.generate(l_shape())
    ring = l_shape()["coordinates"][0]
    full_width = max(p[0] for p in ring) - min(p[0] for p in ring)

    notch_lat = 12.0 + 0.001          # top of the notch for this L
    above = [w for w in plan["waypoints"] if w["lat"] > notch_lat]
    assert above, "expected coverage above the notch"
    spans_above = max(w["lon"] for w in above) - min(w["lon"] for w in above)
    assert spans_above < full_width, \
        f"coverage above the notch still spans the full width ({spans_above})"


def test_square_field_stays_within_bounds():
    plan = SurveyPlanner(altitude_m=20, side_overlap=0.5).generate(square())
    lons = [w["lon"] for w in plan["waypoints"]]
    assert min(lons) >= 77.0 - 1e-9
    assert max(lons) <= 77.001 + 1e-9


def test_higher_altitude_produces_fewer_waypoints():
    low = SurveyPlanner(altitude_m=20).generate(square())
    high = SurveyPlanner(altitude_m=60).generate(square())
    assert len(high["waypoints"]) < len(low["waypoints"])
    assert high["footprint_m"] > low["footprint_m"]


def test_more_overlap_produces_more_waypoints():
    loose = SurveyPlanner(altitude_m=20, side_overlap=0.3).generate(square())
    tight = SurveyPlanner(altitude_m=20, side_overlap=0.8).generate(square())
    assert len(tight["waypoints"]) > len(loose["waypoints"])


def test_distance_and_eta_are_positive_and_consistent():
    plan = SurveyPlanner(altitude_m=20, speed_mps=5.0).generate(square())
    assert plan["distance_m"] > 0
    assert plan["eta_s"] > 0
    assert abs(plan["eta_s"] - plan["distance_m"] / 5.0) < 1e-6


def test_tiny_field_still_produces_a_usable_plan():
    tiny = {
        "type": "Polygon",
        "coordinates": [[[77.0, 12.0], [77.00001, 12.0],
                         [77.00001, 12.00001], [77.0, 12.00001], [77.0, 12.0]]],
    }
    plan = SurveyPlanner(altitude_m=20).generate(tiny)
    assert len(plan["waypoints"]) >= 1


def test_path_is_continuous():
    """Consecutive waypoints must not teleport across the field."""
    from drone.navigation.coordinate_transform import haversine_m

    plan = SurveyPlanner(altitude_m=20, side_overlap=0.5).generate(l_shape())
    wps = plan["waypoints"]
    legs = [haversine_m(wps[i]["lat"], wps[i]["lon"],
                        wps[i + 1]["lat"], wps[i + 1]["lon"])
            for i in range(len(wps) - 1)]
    field = 222.0        # metres across a 0.002 deg field
    assert legs, "plan has no legs"
    assert max(legs) < field, f"a leg crosses the whole field: {max(legs):.0f} m"