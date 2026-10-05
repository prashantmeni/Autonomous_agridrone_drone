"""Flight state machine: DISARMED -> ... -> LANDED / EMERGENCY / ABORT."""
from __future__ import annotations
from enum import Enum
import time
from dataclasses import dataclass, field

class FlightState(str, Enum):
    DISARMED = "DISARMED"
    PRE_FLIGHT_CHECK = "PRE_FLIGHT_CHECK"
    ARMING = "ARMING"
    TAKEOFF = "TAKEOFF"
    MISSION = "MISSION"
    SURVEY = "SURVEY"
    RETURN_HOME = "RETURN_HOME"
    PRECISION_LANDING = "PRECISION_LANDING"
    LANDING = "LANDING"
    LANDED = "LANDED"
    EMERGENCY = "EMERGENCY"
    ABORT = "ABORT"

_ALLOWED: dict[FlightState, set[FlightState]] = {
    FlightState.DISARMED: {FlightState.PRE_FLIGHT_CHECK, FlightState.EMERGENCY},
    FlightState.PRE_FLIGHT_CHECK: {FlightState.ARMING, FlightState.DISARMED, FlightState.ABORT},
    FlightState.ARMING: {FlightState.TAKEOFF, FlightState.DISARMED, FlightState.ABORT, FlightState.EMERGENCY},
    FlightState.TAKEOFF: {FlightState.MISSION, FlightState.SURVEY, FlightState.RETURN_HOME, FlightState.LANDING, FlightState.ABORT, FlightState.EMERGENCY},
    FlightState.MISSION: {FlightState.SURVEY, FlightState.RETURN_HOME, FlightState.LANDING, FlightState.ABORT, FlightState.EMERGENCY},
    FlightState.SURVEY: {FlightState.MISSION, FlightState.RETURN_HOME, FlightState.LANDING, FlightState.ABORT, FlightState.EMERGENCY},
    FlightState.RETURN_HOME: {FlightState.PRECISION_LANDING, FlightState.LANDING, FlightState.LANDED, FlightState.ABORT, FlightState.EMERGENCY},
    FlightState.PRECISION_LANDING: {FlightState.LANDING, FlightState.LANDED, FlightState.ABORT, FlightState.EMERGENCY},
    FlightState.LANDING: {FlightState.LANDED, FlightState.ABORT, FlightState.EMERGENCY},
    FlightState.LANDED: {FlightState.DISARMED},
    FlightState.ABORT: {FlightState.RETURN_HOME, FlightState.LANDING, FlightState.DISARMED, FlightState.EMERGENCY},
    FlightState.EMERGENCY: {FlightState.DISARMED, FlightState.LANDED},
}

@dataclass
class Transition:
    ts: float
    frm: str
    to: str
    reason: str

@dataclass
class FlightStateMachine:
    state: FlightState = FlightState.DISARMED
    history: list[Transition] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def can(self, nxt: FlightState) -> bool:
        return nxt in _ALLOWED.get(self.state, set())

    def transition(self, nxt: FlightState, reason: str = "") -> Transition:
        if not self.can(nxt):
            raise ValueError(f"Illegal transition {self.state} -> {nxt}: {reason}")
        t = Transition(time.time(), self.state.value, nxt.value, reason)
        self.state = nxt
        self.updated_at = t.ts
        self.history.append(t)
        return t

    def force(self, nxt: FlightState, reason: str = "") -> Transition:
        """Emergency override, always logged."""
        t = Transition(time.time(), self.state.value, nxt.value, f"FORCE:{reason}")
        self.state = nxt
        self.updated_at = t.ts
        self.history.append(t)
        return t
