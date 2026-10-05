"""Domain exceptions."""
class DroneError(Exception): pass
class ConfigError(DroneError): pass
class MavlinkError(DroneError): pass
class PreflightError(DroneError): pass
class MissionError(DroneError): pass
class SafetyError(DroneError): pass
class HardwareUnavailable(DroneError):
    def __init__(self, component: str):
        super().__init__(f"{component} unavailable")
        self.component = component
