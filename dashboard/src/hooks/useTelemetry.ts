import { useState, useEffect } from "react";
import { TelemetryData } from "../types";

const EMPTY_TELEMETRY: TelemetryData = {
  connected: false,
  armed: false,
  mode: "—",
  lat: null,
  lon: null,
  alt_m: 0,
  relative_alt_m: 0,
  ground_speed_mps: 0,
  heading_deg: 0,
  battery_voltage_v: 0,
  battery_remaining_pct: -1,
  gps_fix: 0,
  satellites: 0,
  hdop: 99.9,
  roll_deg: 0,
  pitch_deg: 0,
  fsm_state: "DISARMED",
};

export function useTelemetry() {
  const [telemetry, setTelemetry] = useState<TelemetryData>(EMPTY_TELEMETRY);
  const [isConnected, setIsConnected] = useState<boolean>(false);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const host = window.location.host;
      ws = new WebSocket(`${protocol}//${host}/ws/telemetry`);

      ws.onopen = () => setIsConnected(true);

      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed && parsed.data) {
            setTelemetry(parsed.data);
            setIsConnected(true);
          }
        } catch {
          /* ignore malformed frames */
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        reconnectTimeout = setTimeout(connect, 2500);
      };

      ws.onerror = () => setIsConnected(false);
    };

    connect();

    return () => {
      if (ws) ws.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, []);

  return { telemetry, isConnected };
}
