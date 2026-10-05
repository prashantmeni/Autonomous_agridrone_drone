import pytest
from drone.main import build_app
def test_api_health():
    cfg, dapp, api = build_app("config/development.yaml")
    from fastapi.testclient import TestClient
    c = TestClient(api)
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/drone/status").status_code == 200
