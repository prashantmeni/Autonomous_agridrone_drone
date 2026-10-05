"""Entry point: python -m drone.main --config config/simulation.yaml"""
from __future__ import annotations
import argparse, asyncio, logging, os
import uvicorn

log = logging.getLogger("drone.main")

def build_app(config_path: str):
    from .core.config import load_config
    from .core.lifecycle import DroneApp
    from .telemetry.logger import setup_logging
    cfg = load_config(config_path)
    setup_logging(log_dir=cfg.telemetry.log_dir)
    dapp = DroneApp(cfg)
    from .api.server import create_app
    api = create_app(dapp)
    api.state.drone_app = dapp
    return cfg, dapp, api

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/development.yaml")
    p.add_argument("--no-api", action="store_true")
    a = p.parse_args()
    cfg, dapp, api = build_app(a.config)
    os.environ.setdefault("DRONE_CONFIG_PATH", os.path.abspath(a.config))
    async def runner():
        await dapp.start()
        if not a.no_api:
            # Warm the camera up on a background task: V4L2 negotiation does
            # blocking reads that would otherwise starve the MAVLink reader
            # and trip heartbeat timeouts. Never inline, never blocking.
            async def warm_camera():
                try:
                    from .runtime import get_services
                    await asyncio.sleep(2.0)
                    svc = get_services(dapp)
                    svc.camera.start()
                    log.info("camera warm-up started (status=%s)", svc.camera.status()["status"])
                except Exception as e:
                    log.warning("camera warm-up skipped: %s", e)

            asyncio.create_task(warm_camera())
        if a.no_api: 
            while True: await asyncio.sleep(3600)
        srv = uvicorn.Server(uvicorn.Config(api, host=cfg.api.host, port=cfg.api.port, log_level="info"))
        await srv.serve()
    asyncio.run(runner())

if __name__ == "__main__":
    main()
