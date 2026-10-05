"""Operator-configurable drone behaviour rules, persisted on disk as JSON."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from pydantic import BaseModel, Field

log = logging.getLogger("drone.core.behaviour")

DEFAULT_PATH = Path("data/behaviour.json")


class BehaviourRules(BaseModel):
    survey_altitude_m: float = Field(default=20.0, ge=5, le=120)
    cruise_speed_mps: float = Field(default=5.5, ge=1, le=15)
    overlap_pct: int = Field(default=70, ge=30, le=90)
    rth_altitude_m: float = Field(default=25.0, ge=10, le=120)
    rth_battery_reserve_pct: int = Field(default=25, ge=10, le=60)
    obstacle_brake_distance_m: float = Field(default=3.0, ge=0.5, le=15.0)
    auto_charge_on_landing: bool = True
    auto_nadir_lock: bool = True
    geofence_enabled: bool = True
    landing_pad_id: str = "PAD_01"


_lock = threading.Lock()


def load_behaviour(path: str | Path = DEFAULT_PATH) -> BehaviourRules:
    p = Path(path)
    with _lock:
        try:
            if p.exists():
                return BehaviourRules(**(json.loads(p.read_text()) or {}))
        except Exception as e:
            log.warning("behaviour file unreadable, using defaults: %s", e)
    return BehaviourRules()


def save_behaviour(rules: BehaviourRules, path: str | Path = DEFAULT_PATH) -> BehaviourRules:
    p = Path(path)
    with _lock:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(rules.model_dump_json(indent=2))
        except Exception as e:
            log.error("could not persist behaviour rules: %s", e)
    return rules
