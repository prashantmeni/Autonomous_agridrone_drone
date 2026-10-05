# SITL scenario: survey
# 1. Start PX4 SITL (Gazebo): `make px4_sitl gz_x500`
# 2. python -m drone.main --config config/simulation.yaml
# 3. POST /api/survey/generate with data/boundaries/example.geojson
# 4. POST /api/drone/takeoff, /api/missions/{id}/start, observe /ws/telemetry
