"""CLI: status/health/mavlink-test/camera-test/gps-test/mission/logs."""
from __future__ import annotations
import typer
app = typer.Typer()

@app.command()
def status(config: str = "config/development.yaml"):
    from drone.core.config import load_config
    c = load_config(config)
    typer.echo(f"{c.drone_name} env={c.environment} mavlink={c.mavlink.connection}")

@app.command()
def health(config: str = "config/development.yaml"):
    from drone.core.config import load_config
    from drone.telemetry.metrics import pi_stats
    c = load_config(config); print(pi_stats(), c.environment)

@app.command()
def mavlink_test(config: str = "config/development.yaml"):
    from drone.core.config import load_config
    from drone.mavlink.connection import MavlinkConnection
    c = load_config(config)
    m = MavlinkConnection(c.mavlink.connection, c.mavlink.baud)
    m.connect()
    typer.echo(f"heartbeat: {m.wait_heartbeat(8.0)} connected={m.connected}")

@app.command()
def camera_test():
    from drone.perception.camera import CameraManager
    from types import SimpleNamespace
    m = CameraManager(SimpleNamespace(enabled=True, width=640, height=480, fps=15))
    typer.echo(m.start())

@app.command()
def gps_test(config: str = "config/development.yaml"):
    from drone.core.config import load_config
    from drone.mavlink.connection import MavlinkConnection
    from drone.mavlink.telemetry import TelemetryStore
    import time
    c = load_config(config)
    m = MavlinkConnection(c.mavlink.connection, c.mavlink.baud)
    m.connect(); m.wait_heartbeat(5.0)
    st = TelemetryStore()
    t0 = time.time()
    while time.time() - t0 < 5 and m.master:
        msg = m.master.recv_match(blocking=False)
        if msg: st.update_from_msg(msg)
    typer.echo(f"fix={st.snap.gps_fix} sats={st.snap.satellites} lat={st.snap.lat} lon={st.snap.lon}")

@app.command()
def mission_validate(path: str):
    import yaml
    from drone.autonomy.waypoint_manager import validate_mission
    body = yaml.safe_load(open(path))
    ok, errs = validate_mission(body, 30.0, 8.0)
    typer.echo(f"valid={ok} errors={errs}")

@app.command()
def logs(n: int = 20):
    from pathlib import Path
    p = Path("data/logs/drone.log")
    if p.exists(): typer.echo("\n".join(p.read_text().splitlines()[-n:]))
    else: typer.echo("no logs yet")

if __name__ == "__main__": app()
