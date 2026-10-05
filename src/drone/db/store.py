"""SQLite metadata store (missions, boundaries, detections, events). High-rate telemetry goes to JSONL."""
from __future__ import annotations
import sqlite3, json, time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS missions(id TEXT PRIMARY KEY, name TEXT, body TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS boundaries(id TEXT PRIMARY KEY, name TEXT, geojson TEXT, area_m2 REAL, created_at REAL);
CREATE TABLE IF NOT EXISTS detections(id INTEGER PRIMARY KEY AUTOINCREMENT, crop TEXT, disease TEXT, conf REAL, lat REAL, lon REAL, ts REAL, image TEXT);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, level TEXT, component TEXT, event TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS flights(id TEXT PRIMARY KEY, started_at REAL, ended_at REAL, summary TEXT);
"""

class Database:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.executescript(SCHEMA)

    def log_event(self, level, component, event, data=None):
        self.con.execute("INSERT INTO events(ts,level,component,event,data) VALUES(?,?,?,?,?)",
            (time.time(), level, component, event, json.dumps(data or {})))
        self.con.commit()

    def save_mission(self, mid, name, body: dict):
        self.con.execute("INSERT OR REPLACE INTO missions VALUES(?,?,?,?)", (mid, name, json.dumps(body), time.time()))
        self.con.commit()

    def get_mission(self, mid):
        r = self.con.execute("SELECT body FROM missions WHERE id=?", (mid,)).fetchone()
        return json.loads(r[0]) if r else None

    def list_missions(self):
        return [{"id": r[0], "name": r[1]} for r in self.con.execute("SELECT id,name FROM missions")]

    def save_boundary(self, bid, name, geojson: dict, area: float):
        self.con.execute("INSERT OR REPLACE INTO boundaries VALUES(?,?,?,?,?)",
            (bid, name, json.dumps(geojson), area, time.time()))
        self.con.commit()

    def add_detection(self, crop, disease, conf, lat, lon, image=""):
        self.con.execute("INSERT INTO detections(crop,disease,conf,lat,lon,ts,image) VALUES(?,?,?,?,?,?,?)",
            (crop, disease, conf, lat, lon, time.time(), image))
        self.con.commit()
