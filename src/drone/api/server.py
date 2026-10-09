"""FastAPI app: health, drone, missions, boundaries, survey, detections. Dangerous endpoints gated."""
from __future__ import annotations
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from .schemas import TakeoffReq, MissionCreate, BoundaryCreate, SurveyReq

import logging

log = logging.getLogger("drone.api.server")

# Label files may ride along with a model as `<model>.onnx.txt` and are stripped
# to the model name when checking that a model already exists.
LABEL_SUFFIXES = (".txt", ".json", ".labels")

# Upload ceiling, matching the standalone AI module's limit.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def _strip_label_suffix(name: str) -> str:
    """`model.onnx.txt` -> `model.onnx` (labels attach to the model by name)."""
    for suf in LABEL_SUFFIXES:
        if name.lower().endswith(suf):
            return name[: -len(suf)]
    return name


def _crop_ai_undecodable(image_path):
    """Standard response when an uploaded file is not a decodable image."""
    return {
        "success": False,
        "status": "IMAGE_INVALID",
        "error_code": "IMAGE_INVALID",
        "message": "Uploaded file could not be decoded as an image.",
        "suggestion": "Upload a JPEG or PNG captured from the camera.",
        "crop": None, "disease": None, "confidence": 0.0,
        "confidence_level": "NONE", "reliable": False,
        "image_quality": {"valid": False, "metrics": {}},
        "model": {"name": None, "version": None},
        "timing": {"preprocess_ms": 0, "inference_ms": 0,
                   "postprocess_ms": 0, "total_ms": 0},
        "image_path": image_path,
    }


def _persist_disease_config(drone_app, model_path: str) -> None:
    """Write the adopted model path back to the active config file."""
    import os
    import re

    path = os.environ.get("DRONE_CONFIG_PATH", "").strip()
    if not path:
        return
    try:
        p = Path(path)
        if not p.is_file():
            return
        text = p.read_text(encoding="utf-8")
        block = re.search(r"(?m)^disease_detection:\s*\n(?:\s+.*\n)*", text)
        if not block:
            return
        body = block.group(0)
        new_body = body
        if re.search(r"(?m)^\s*model_path:", body):
            new_body = re.sub(r"(?m)^(\s*model_path:).*$", rf'\1 "{model_path}"', body)
        else:
            new_body = body.rstrip("\n") + f'\n  model_path: "{model_path}"\n'
        if re.search(r"(?m)^\s*enabled:", new_body):
            new_body = re.sub(r"(?m)^(\s*enabled:).*$", r"\1 true", new_body)
        else:
            new_body = new_body.rstrip("\n") + "\n  enabled: true\n"
        p.write_text(text.replace(body, new_body), encoding="utf-8")
        log.info("persisted disease model path to %s", p)
    except Exception as e:
        log.warning("could not persist disease config: %s", e)

def create_app(drone_app) -> FastAPI:
    app = FastAPI(title="AgriDrone API", version="0.1.0")

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    async def guard(x_token: str | None = Header(default=None)):
        import os
        secret = os.environ.get("API_SECRET_KEY", "")
        if secret and x_token != secret:
            # allow open in dev when no secret configured? require if set
            if secret and secret != "change-me-generate-strong-secret":
                raise HTTPException(403, "forbidden")
        return True

    @app.get("/api/health")
    async def health():
        from ..telemetry.metrics import pi_stats
        from ..mavlink.health import evaluate
        snap = drone_app.tstore.snap
        return {"environment": drone_app.cfg.environment, "fsm": drone_app.fsm.state.value,
                **evaluate(snap, drone_app.cfg.safety, pi_stats())}

    @app.get("/api/drone/status")
    async def status():
        s = drone_app.tstore.snap.to_dict(); s["fsm_state"] = drone_app.fsm.state.value
        return s
    @app.get("/api/drone/telemetry")
    async def telem(): return drone_app.tstore.snap.to_dict()

    @app.get("/api/drone/preflight")
    async def preflight():
        from ..safety.flight_safety import preflight_check
        return preflight_check(drone_app)

    @app.get("/api/drone/param")
    async def get_param(name: str):
        from ..mavlink import parameters as P
        import asyncio
        val = await asyncio.to_thread(P.read_param, drone_app.conn, name)
        if val is None: raise HTTPException(404, f"parameter {name} not found")
        return {"name": name, "value": val}

    @app.post("/api/drone/arm")
    async def arm(_=Depends(guard)):
        from ..mavlink import commands as C
        from ..safety.flight_safety import preflight_check
        import asyncio, logging, time
        log = logging.getLogger("drone.api.arm")
        snap = drone_app.tstore.snap
        pf = preflight_check(drone_app)
        fails = [f"{k}: {v}" for k, v in pf["checks"].items() if v in ("FAIL", "CRITICAL")]
        ok, why = await asyncio.to_thread(C.arm, drone_app.conn)
        if not ok:
            parts = [f"arm failed: {why}", f"mode={snap.mode}", f"fsm={drone_app.fsm.state.value}"]
            if fails:
                parts.append("preflight fail: " + ", ".join(fails))
            if snap.sensor_issues:
                parts.append("unhealthy sensors: " + ", ".join(snap.sensor_issues))
            px4 = C._recent_px4_text(drone_app.conn, time.time() - 10)
            if px4:
                parts.append(f"px4 says: {px4}")
            detail = " | ".join(parts)
            log.error(f"ARM DENIED — {detail}")
            raise HTTPException(400, detail)
        log.info(f"ARM ACCEPTED — mode={snap.mode}")
        return {"armed": True}

    @app.post("/api/drone/disarm")
    async def disarm(_=Depends(guard)):
        from ..mavlink import commands as C
        from ..core.state import FlightState
        import asyncio, logging
        log = logging.getLogger("drone.api.arm")
        ok, why = await asyncio.to_thread(C.disarm, drone_app.conn)
        if not ok:
            log.error(f"DISARM DENIED — {why}")
            raise HTTPException(400, f"disarm failed: {why}")
        drone_app.fsm.force(FlightState.DISARMED, "disarmed via API")
        return {"armed": False}

    @app.post("/api/drone/takeoff")
    async def takeoff(r: TakeoffReq, _=Depends(guard)):
        from ..autonomy.flight_manager import FlightManager
        try: return await FlightManager(drone_app).takeoff(r.altitude_m)
        except Exception as e: raise HTTPException(400, str(e))

    @app.post("/api/drone/rtl")
    async def rtl(_=Depends(guard)):
        from ..autonomy.flight_manager import FlightManager
        return await FlightManager(drone_app).rtl()
    @app.post("/api/drone/land")
    async def land(_=Depends(guard)):
        from ..autonomy.flight_manager import FlightManager
        return await FlightManager(drone_app).land()

    # ------------------------------------------------------------- precision landing
    @app.post("/api/drone/precision-landing")
    async def precision_land(_=Depends(guard)):
        from ..runtime import get_services
        svc = get_services(drone_app)
        svc.start()
        try:
            return await svc.precision.start(drone_app, svc.camera)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.get("/api/drone/precision-landing")
    async def precision_land_status():
        from ..runtime import get_services
        return get_services(drone_app).precision.get()

    # ------------------------------------------------------------------- gimbal
    @app.get("/api/drone/gimbal")
    async def get_gimbal():
        from ..runtime import get_services
        return get_services(drone_app).gimbal.status()

    @app.post("/api/drone/gimbal")
    async def set_gimbal(payload: dict, _=Depends(guard)):
        from ..runtime import get_services
        svc = get_services(drone_app)
        pan = payload.get("pan_deg", payload.get("pan"))
        tilt = payload.get("tilt_deg", payload.get("tilt"))
        if pan is None and tilt is None:
            raise HTTPException(422, "pan_deg or tilt_deg required")
        return svc.gimbal.set(pan, tilt, drone_app.conn)

    # ----------------------------------------------------------------- charging
    @app.get("/api/drone/charging")
    async def charging_status():
        from ..runtime import get_services
        return get_services(drone_app).charging.status(drone_app)

    # ---------------------------------------------------------------- obstacles
    @app.get("/api/drone/obstacles")
    async def obstacles_status():
        from ..runtime import get_services
        return get_services(drone_app).proximity.status(drone_app.tstore.snap)

    # ------------------------------------------------------------------- camera
    @app.get("/api/camera/status")
    async def camera_status():
        from ..runtime import get_services
        svc = get_services(drone_app)
        return svc.camera.status()

    @app.get("/api/camera/snapshot")
    async def camera_snapshot():
        from ..runtime import get_services
        svc = get_services(drone_app)
        svc.start()
        jpeg = svc.camera.snapshot()
        if not jpeg:
            raise HTTPException(503, "CAMERA_UNAVAILABLE")
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/api/camera/stream")
    async def camera_stream():
        from ..runtime import get_services
        svc = get_services(drone_app)
        svc.start()

        def gen():
            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            for jpeg in svc.camera.mjpeg():
                yield boundary + jpeg + b"\r\n"

        return StreamingResponse(
            gen(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
        )

    @app.post("/api/camera/record")
    async def camera_record(payload: dict, _=Depends(guard)):
        from ..runtime import get_services
        svc = get_services(drone_app)
        svc.start()
        action = (payload or {}).get("action", "toggle")
        st = svc.camera.status()
        if action == "start" or (action == "toggle" and not st["recording"]):
            return svc.camera.start_recording()
        if action == "stop" or action == "toggle":
            return svc.camera.stop_recording()
        raise HTTPException(422, "action must be start|stop|toggle")

    # ------------------------------------------------------------- flight mode
    @app.post("/api/drone/mode")
    async def set_mode(payload: dict, _=Depends(guard)):
        import asyncio
        from ..mavlink import commands as C
        mode = (payload or {}).get("mode", "").upper()
        if not mode:
            raise HTTPException(422, "mode required")
        ok = await asyncio.to_thread(C.set_mode, drone_app.conn, mode)
        if not ok:
            raise HTTPException(400, f"could not set mode {mode} (link down or unknown mode)")
        drone_app.db.log_event("INFO", "api", "MODE_SET", {"mode": mode})
        return {"mode": mode}

    # ---------------------------------------------------------------- behaviour
    @app.get("/api/behaviour")
    async def get_behaviour():
        from ..core.behaviour import load_behaviour
        return load_behaviour().model_dump()

    @app.post("/api/behaviour")
    async def set_behaviour(payload: dict, _=Depends(guard)):
        from ..core.behaviour import BehaviourRules, save_behaviour
        try:
            rules = BehaviourRules(**(payload or {}))
        except Exception as e:
            raise HTTPException(422, str(e))
        save_behaviour(rules)
        drone_app.db.log_event("INFO", "api", "BEHAVIOUR_UPDATED", rules.model_dump())
        return rules.model_dump()

    # ------------------------------------------------------- disease analysis
    # Inference lives in the standalone ai_disease_detection module. This
    # adapter keeps the endpoint contract the dashboard already renders while
    # all detection logic stays in one place.
    @app.get("/api/disease/model")
    async def disease_model_status():
        from ..perception import crop_ai
        cfg = drone_app.cfg.disease_detection
        return crop_ai.model_status(cfg.model_path, cfg.enabled,
                                    cfg.confidence_threshold)

    @app.get("/api/crop-ai/model")
    async def crop_ai_model_status():
        """Model metadata plus the AI module's quality thresholds."""
        from ..perception import crop_ai
        cfg = drone_app.cfg.disease_detection
        status = crop_ai.model_status(cfg.model_path, cfg.enabled,
                                      cfg.confidence_threshold)
        status["confidence_thresholds"] = {
            "high": cfg.confidence_high,
            "medium": cfg.confidence_medium,
            "reliable_min": cfg.reliable_min_confidence,
            "store_min": cfg.confidence_threshold,
        }
        thresholds = crop_ai.quality_thresholds()
        if thresholds:
            status["image_quality_thresholds"] = thresholds
        status["module_available"] = crop_ai.available()
        if not crop_ai.available():
            status["module_error"] = crop_ai.unavailable_reason()
        return status

    @app.post("/api/disease/model")
    async def disease_model_upload(file: UploadFile = File(...), _=Depends(guard)):
        """Accept a model file; validation is the AI module's job."""
        from ..perception import crop_ai

        name = Path((file.filename or "").strip()).name
        if not name:
            raise HTTPException(400, "no filename provided")
        ext = Path(name).suffix.lower()

        if ext in (".pt", ".pth"):
            raise HTTPException(400, "PyTorch models need torch (~200MB+). "
                                      "Convert to ONNX instead.")
        if ext in (".h5", ".keras"):
            raise HTTPException(400, "Keras models need tensorflow (~600MB+). "
                                      "Convert to ONNX instead.")
        is_label = ext in (".txt", ".json", ".labels")
        if not is_label and ext != ".onnx":
            raise HTTPException(400, f"{ext or 'no extension'} is not supported. "
                                      "Use .onnx (or a label file: .txt, .json).")

        target_dir = crop_ai.model_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / name

        # Labels must accompany a model that already exists.
        if is_label:
            expected = Path(crop_ai._strip_label_suffix(name)).name
            if not (target_dir / expected).is_file():
                raise HTTPException(400, f"No model named {expected} in "
                                          f"{target_dir.name}/ yet. Upload the "
                                          "model file first, then its labels.")

        written = 0
        try:
            with dest.open("wb") as fh:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > crop_ai.MAX_UPLOAD_BYTES:
                        fh.close()
                        dest.unlink(missing_ok=True)
                        raise HTTPException(413, "model exceeds the size limit")
                    fh.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            dest.unlink(missing_ok=True)
            raise HTTPException(500, f"upload failed: {e}")
        finally:
            await file.close()

        if written == 0:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "uploaded file was empty")

        cfg = drone_app.cfg.disease_detection

        if is_label:
            status = crop_ai.model_status(cfg.model_path, cfg.enabled,
                                          cfg.confidence_threshold)
            status["uploaded"] = {"name": name, "bytes": written,
                                  "kind": "labels"}
            return status

        # Adopt only if it loads; otherwise keep whatever is already active.
        previous = str(cfg.model_path)
        candidate = crop_ai.model_status(dest, True, cfg.confidence_threshold)
        if candidate.get("status") != "OK":
            dest.unlink(missing_ok=True)
            raise HTTPException(400, {
                "error_code": "MODEL_INVALID",
                "message": "Model rejected; the previously active model is unchanged.",
                "errors": [candidate.get("detail", "could not load")],
                "active_model": Path(previous).name if previous else None,
            })

        cfg.model_path = str(dest)
        cfg.enabled = True
        _persist_disease_config(drone_app, str(dest))

        status = crop_ai.model_status(cfg.model_path, cfg.enabled,
                                      cfg.confidence_threshold)
        status["uploaded"] = {"name": name, "bytes": written, "kind": "model"}
        status["validation"] = {"valid": True}
        return status

    @app.post("/api/disease/analyze")
    async def disease_analyze(payload: dict | None = None,
                              file: UploadFile | None = File(default=None),
                              mark: str | None = Form(default=None),
                              _=Depends(guard)):
        """Classify a frame from the camera, or an uploaded still image."""
        from ..runtime import get_services
        from ..perception import crop_ai

        cfg = drone_app.cfg.disease_detection
        want_mark = False
        if mark is not None:
            want_mark = str(mark).strip().lower() in ("1", "true", "yes", "on")
        elif payload:
            want_mark = bool(payload.get("mark", False))

        source = "camera"
        image_path = None
        if file is not None and file.filename:
            source = "upload"
            image_path = str(file.filename)

        try:
            if source == "upload":
                from ai_disease_detection.ai_disease import preprocessing as ap

                data = await file.read()
                await file.close()
                frame = ap.decode_image(data)
                if frame is None:
                    return _crop_ai_undecodable(image_path)
            else:
                svc = get_services(drone_app)
                svc.start()
                frame = svc.camera.frame()
                if frame is None:
                    raise HTTPException(503, "CAMERA_UNAVAILABLE")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"could not read frame: {e}")

        try:
            result = crop_ai.analyze(
                frame, cfg.model_path, cfg.confidence_threshold,
                source=source, image_path=image_path,
                high_threshold=cfg.confidence_high,
                medium_threshold=cfg.confidence_medium,
            )
        except Exception as e:
            raise HTTPException(503, str(e))

        result["source"] = source
        if result.get("reliable") and want_mark and \
                result.get("confidence", 0.0) >= cfg.confidence_threshold:
            snap = drone_app.tstore.snap
            drone_app.db.add_detection(
                result.get("crop") or "unknown",
                result.get("disease") or "unclassified",
                float(result.get("confidence", 0.0)),
                snap.lat or 0.0, snap.lon or 0.0,
            )
            result["marked"] = True
        return result

    @app.get("/api/missions")
    async def mlist(): return drone_app.db.list_missions()
    # Must be registered before /api/missions/{mid}: FastAPI matches in
    # declaration order, so the wildcard route would otherwise swallow
    # "active" as a mission id and answer 404.
    @app.get("/api/missions/active")
    async def mactive(): return drone_app.missions.status()
    @app.post("/api/missions")
    async def mcreate(m: MissionCreate):
        mid = str(uuid.uuid4())
        drone_app.db.save_mission(mid, m.name, m.model_dump())
        return {"id": mid}
    @app.get("/api/missions/{mid}")
    async def mget(mid: str):
        b = drone_app.db.get_mission(mid)
        if not b: raise HTTPException(404, "not found")
        return b
    @app.post("/api/missions/{mid}/start")
    async def mstart(mid: str, _=Depends(guard)):
        b = drone_app.db.get_mission(mid)
        if not b: raise HTTPException(404, "not found")
        try:
            result = await drone_app.missions.start(mid, b)
        except ValueError as e:
            raise HTTPException(409, str(e))
        except Exception as e:
            raise HTTPException(400, str(e))
        if result.get("status") == "failed":
            raise HTTPException(409, result.get("error", "mission failed to start"))
        return result
    @app.post("/api/missions/{mid}/pause")
    async def mpause(mid: str, _=Depends(guard)):
        return await drone_app.missions.pause()
    @app.post("/api/missions/{mid}/resume")
    async def mresume(mid: str, _=Depends(guard)):
        return await drone_app.missions.resume()
    @app.post("/api/missions/{mid}/abort")
    async def mabort(mid: str, _=Depends(guard)):
        return await drone_app.missions.abort(f"abort {mid}")

    @app.post("/api/boundaries")
    async def bcreate(b: BoundaryCreate):
        from ..mapping.survey_planner import validate_geojson, BoundaryMapper
        ok, errs = validate_geojson(b.geojson)
        if not ok: raise HTTPException(400, str(errs))
        stats = BoundaryMapper.stats(b.geojson)
        bid = str(uuid.uuid4())
        drone_app.db.save_boundary(bid, b.name, b.geojson, stats["area_m2"])
        return {"id": bid, **stats}
    @app.post("/api/boundaries/import-kml")
    async def bimport(payload: dict):
        from ..mapping.kml import kml_to_geojson
        from ..mapping.survey_planner import BoundaryMapper
        try: gj = kml_to_geojson(payload.get("kml", ""))
        except Exception as e: raise HTTPException(400, str(e))
        stats = BoundaryMapper.stats(gj)
        bid = str(uuid.uuid4())
        drone_app.db.save_boundary(bid, payload.get("name", "kml"), gj, stats["area_m2"])
        drone_app.db.log_event("INFO", "api", "BOUNDARY_IMPORT_KML", {"id": bid})
        return {"id": bid, "geojson": gj, **stats}
    @app.post("/api/survey/generate")
    async def survey(r: SurveyReq):
        from ..mapping.survey_planner import SurveyPlanner
        from ..autonomy.geofence import Geofence
        p = SurveyPlanner(r.altitude_m, r.speed_mps, r.overlap)
        out = p.generate(r.boundary_geojson)
        try:
            poly = [(pt[1], pt[0]) for pt in r.boundary_geojson["coordinates"][0]]
            g = Geofence(poly)
            out["waypoints"] = [w for w in out["waypoints"] if g.contains(w["lat"], w["lon"])]
        except Exception: pass
        return out

    @app.get("/api/detections")
    async def dets():
        rows = drone_app.db.con.execute("SELECT crop,disease,conf,lat,lon,ts FROM detections ORDER BY ts DESC LIMIT 100").fetchall()
        return [{"crop": r[0], "disease": r[1], "confidence": r[2], "latitude": r[3], "longitude": r[4], "timestamp": r[5]} for r in rows]
    @app.post("/api/detections/mark")
    async def dmark(payload: dict, _=Depends(guard)):
        """Record a manually confirmed detection (dashboard 'plot disease zone')."""
        crop = str(payload.get("crop") or "unknown")
        disease = str(payload.get("disease") or "unclassified")
        try:
            conf = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            raise HTTPException(400, "confidence must be a number")
        lat = float(payload.get("latitude") or 0.0)
        lon = float(payload.get("longitude") or 0.0)
        drone_app.db.add_detection(crop, disease, conf, lat, lon)
        return {"status": "marked", "crop": crop, "disease": disease, "confidence": conf}

    @app.get("/api/logs")
    async def logs(level: str = "INFO", limit: int = 100):
        rows = drone_app.db.con.execute("SELECT ts,level,component,event,data FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": r[0], "level": r[1], "component": r[2], "event": r[3]} for r in rows]

    from .websocket import router as ws_router
    app.include_router(ws_router)

    # Serve the built dashboard when present, so the UI is reachable on the
    # same origin as the API (no dev server, no tunnel, no CORS).
    _mount_dashboard(app, drone_app)
    return app


def _dashboard_dir(drone_app) -> Path | None:
    """Locate the built dashboard: DASHBOARD_DIR env, else ./dashboard/dist."""
    import os
    candidates = []
    env = os.environ.get("DASHBOARD_DIR", "").strip()
    if env:
        candidates.append(Path(env))
    candidates.append(Path("dashboard") / "dist")
    candidates.append(Path(__file__).resolve().parents[3] / "dashboard" / "dist")
    for d in candidates:
        if (d / "index.html").is_file():
            return d
    return None


def _mount_dashboard(app: FastAPI, drone_app) -> None:
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    root = _dashboard_dir(drone_app)
    if root is None:
        log.warning("dashboard build not found; API-only mode (GET / returns 404)")
        return
    assets = root / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/", include_in_schema=False)
    async def dashboard_index():
        return FileResponse(str(root / "index.html"))

    @app.get("/{path:path}", include_in_schema=False)
    async def dashboard_spa(path: str):
        """Client-side routing: hand back index.html for unknown non-API paths."""
        if path.startswith(("api/", "ws/")):
            raise HTTPException(404, "not found")
        candidate = (root / path).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            raise HTTPException(404, "not found")
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(root / "index.html"))
