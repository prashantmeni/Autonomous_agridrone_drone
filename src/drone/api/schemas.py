from __future__ import annotations
from pydantic import BaseModel
from typing import Any
class TakeoffReq(BaseModel): altitude_m: float = 15.0
class MissionCreate(BaseModel):
    name: str
    waypoints: list[dict[str, Any]] = []
    takeoff: dict[str, Any] = {}
    survey: dict[str, Any] = {}
    return_to_home: bool = True
    landing: dict[str, Any] = {}
class BoundaryCreate(BaseModel):
    name: str
    geojson: dict[str, Any]
class SurveyReq(BaseModel):
    boundary_geojson: dict[str, Any]
    altitude_m: float = 20.0
    speed_mps: float = 5.0
    overlap: float = 0.7
