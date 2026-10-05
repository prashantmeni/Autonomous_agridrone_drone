"""Split route modules (imported by server.py for clarity; server wires them)."""
from fastapi import APIRouter
health_router = APIRouter()
drone_router = APIRouter()
mission_router = APIRouter()
