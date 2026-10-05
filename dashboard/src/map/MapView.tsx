import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import { Crosshair, Layers, MapPin, ZoomIn, ZoomOut, Compass } from "lucide-react";
import { TelemetryData, Detection } from "../types";

export type FreeMapSource = "osm" | "satellite" | "dark" | "topo";

const FREE_MAP_LAYERS: Record<
  FreeMapSource,
  { name: string; url: string; attribution: string; maxZoom: number }
> = {
  osm: {
    name: "OpenStreetMap (Free / Open Source)",
    url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors',
    maxZoom: 19,
  },
  satellite: {
    name: "Satellite Imagery (Free / Public)",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP",
    maxZoom: 19,
  },
  dark: {
    name: "Dark Aerospace (Free Open Tiles)",
    url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap',
    maxZoom: 20,
  },
  topo: {
    name: "OpenTopoMap (Elevation Contours)",
    url: "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>, SRTM | Map style &copy; <a href="https://opentopomap.org">OpenTopoMap</a>',
    maxZoom: 17,
  },
};

interface MapViewProps {
  telemetry: TelemetryData;
  boundaryGeojson?: any;
  surveyWaypoints?: { lat: number; lon: number; alt: number }[];
  surveyLines?: [number, number][][];
  detections?: Detection[];
  isDrawingBoundary?: boolean;
  onAddPoint?: (lat: number, lon: number) => void;
  drawnPoints?: [number, number][];
}

export const MapView: React.FC<MapViewProps> = ({
  telemetry,
  boundaryGeojson,
  surveyWaypoints = [],
  surveyLines = [],
  detections = [],
  isDrawingBoundary = false,
  onAddPoint,
  drawnPoints = [],
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const droneMarkerRef = useRef<L.Marker | null>(null);
  const trailPolylineRef = useRef<L.Polyline | null>(null);
  const boundaryLayerRef = useRef<L.GeoJSON | null>(null);
  const surveyLineLayerRef = useRef<L.Polyline | null>(null);
  const waypointLayerRef = useRef<L.LayerGroup | null>(null);
  const detectionLayerRef = useRef<L.LayerGroup | null>(null);
  const drawingLayerRef = useRef<L.LayerGroup | null>(null);

  const trailCoordinatesRef = useRef<[number, number][]>([]);
  const [followDrone, setFollowDrone] = useState<boolean>(true);
  const [mapSource, setMapSource] = useState<FreeMapSource>("dark");
  const [showLayerSelector, setShowLayerSelector] = useState<boolean>(false);
  const tileLayerRef = useRef<L.TileLayer | null>(null);

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const initialLat = telemetry.lat || 12.9716;
    const initialLon = telemetry.lon || 77.5946;

    const map = L.map(mapContainerRef.current, {
      center: [initialLat, initialLon],
      zoom: 17,
      zoomControl: false,
      attributionControl: true,
    });

    const initialLayerConfig = FREE_MAP_LAYERS[mapSource];
    const initialTileLayer = L.tileLayer(initialLayerConfig.url, {
      maxZoom: initialLayerConfig.maxZoom,
      attribution: initialLayerConfig.attribution,
    }).addTo(map);
    tileLayerRef.current = initialTileLayer;

    // Layer groups
    waypointLayerRef.current = L.layerGroup().addTo(map);
    detectionLayerRef.current = L.layerGroup().addTo(map);
    drawingLayerRef.current = L.layerGroup().addTo(map);

    // Trail polyline
    const trail = L.polyline([], {
      color: "#06b6d4",
      weight: 3,
      opacity: 0.7,
      dashArray: "4, 6",
    }).addTo(map);
    trailPolylineRef.current = trail;

    // Survey line
    const surveyLine = L.polyline([], {
      color: "#10b981",
      weight: 2,
      opacity: 0.85,
    }).addTo(map);
    surveyLineLayerRef.current = surveyLine;

    // Drone Marker
    const droneHtml = `
      <div style="position: relative; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center;">
        <div style="position: absolute; inset: 0; border-radius: 50%; background: rgba(16, 185, 129, 0.25); animation: pulse-animation 2s infinite;"></div>
        <div id="drone-rotator" style="transform: rotate(0deg); transition: transform 0.2s linear; display: flex; align-items: center; justify-content: center;">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="12 2 19 21 12 17 5 21 12 2" fill="rgba(16, 185, 129, 0.4)"/>
          </svg>
        </div>
      </div>
    `;

    const droneIcon = L.divIcon({
      html: droneHtml,
      className: "drone-leaflet-icon",
      iconSize: [44, 44],
      iconAnchor: [22, 22],
    });

    const marker = L.marker([initialLat, initialLon], { icon: droneIcon, zIndexOffset: 1000 }).addTo(map);
    droneMarkerRef.current = marker;

    // Click handler for drawing boundaries
    map.on("click", (e: L.LeafletMouseEvent) => {
      if (isDrawingBoundary && onAddPoint) {
        onAddPoint(e.latlng.lat, e.latlng.lng);
      }
    });

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update Free Map Tile Source
  const setFreeMapSource = (source: FreeMapSource) => {
    if (!mapInstanceRef.current) return;
    if (tileLayerRef.current) {
      mapInstanceRef.current.removeLayer(tileLayerRef.current);
    }
    const layerConfig = FREE_MAP_LAYERS[source];
    const newTileLayer = L.tileLayer(layerConfig.url, {
      maxZoom: layerConfig.maxZoom,
      attribution: layerConfig.attribution,
    }).addTo(mapInstanceRef.current);

    tileLayerRef.current = newTileLayer;
    setMapSource(source);
    setShowLayerSelector(false);
  };

  // Update Drone Marker Position & Trail
  useEffect(() => {
    if (!droneMarkerRef.current || !mapInstanceRef.current) return;
    if (telemetry.lat == null || telemetry.lon == null) return;

    const latlng: [number, number] = [telemetry.lat, telemetry.lon];
    droneMarkerRef.current.setLatLng(latlng);

    // Rotate icon smoothly
    const rotator = document.getElementById("drone-rotator");
    if (rotator) {
      rotator.style.transform = `rotate(${telemetry.heading_deg}deg)`;
    }

    // Append to flight trail
    const coords = trailCoordinatesRef.current;
    if (coords.length === 0 || coords[coords.length - 1][0] !== latlng[0] || coords[coords.length - 1][1] !== latlng[1]) {
      coords.push(latlng);
      if (coords.length > 500) coords.shift();
      trailPolylineRef.current?.setLatLngs(coords);
    }

    if (followDrone) {
      mapInstanceRef.current.panTo(latlng, { animate: true, duration: 0.5 });
    }
  }, [telemetry.lat, telemetry.lon, telemetry.heading_deg, followDrone]);

  // Update Field Boundary GeoJSON
  useEffect(() => {
    if (!mapInstanceRef.current) return;

    if (boundaryLayerRef.current) {
      mapInstanceRef.current.removeLayer(boundaryLayerRef.current);
      boundaryLayerRef.current = null;
    }

    if (boundaryGeojson && boundaryGeojson.coordinates) {
      const geoLayer = L.geoJSON(boundaryGeojson, {
        style: {
          color: "#34d399",
          weight: 2.5,
          opacity: 0.9,
          fillColor: "#10b981",
          fillOpacity: 0.15,
          dashArray: "6, 4",
        },
      }).addTo(mapInstanceRef.current);

      boundaryLayerRef.current = geoLayer;
      try {
        const bounds = geoLayer.getBounds();
        if (bounds.isValid()) {
          mapInstanceRef.current.fitBounds(bounds, { padding: [40, 40] });
        }
      } catch {}
    }
  }, [boundaryGeojson]);

  // Update Survey Lines and Waypoints
  useEffect(() => {
    if (!mapInstanceRef.current) return;

    // Survey path
    if (surveyLines.length > 0 && surveyLineLayerRef.current) {
      const latlngs: [number, number][] = [];
      surveyLines.forEach((line) => {
        line.forEach((pt) => latlngs.push([pt[0], pt[1]]));
      });
      surveyLineLayerRef.current.setLatLngs(latlngs);
    } else {
      surveyLineLayerRef.current?.setLatLngs([]);
    }

    // Numbered Waypoints
    if (waypointLayerRef.current) {
      waypointLayerRef.current.clearLayers();
      surveyWaypoints.forEach((wp, idx) => {
        const wpIcon = L.divIcon({
          html: `<div style="background: #10b981; color: black; font-family: var(--font-mono); font-size: 10px; font-weight: 800; width: 18px; height: 18px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 1.5px solid white; box-shadow: 0 0 6px rgba(0,0,0,0.6);">${idx + 1}</div>`,
          className: "wp-icon",
          iconSize: [18, 18],
          iconAnchor: [9, 9],
        });
        const marker = L.marker([wp.lat, wp.lon], { icon: wpIcon });
        marker.bindPopup(`<b>Waypoint #${idx + 1}</b><br/>Lat: ${wp.lat.toFixed(6)}<br/>Lon: ${wp.lon.toFixed(6)}<br/>Alt: ${wp.alt}m`);
        waypointLayerRef.current?.addLayer(marker);
      });
    }
  }, [surveyWaypoints, surveyLines]);

  // Update Plant Disease Detections
  useEffect(() => {
    if (!detectionLayerRef.current || !mapInstanceRef.current) return;
    detectionLayerRef.current.clearLayers();

    detections.forEach((d) => {
      const detIcon = L.divIcon({
        html: `<div style="background: #ef4444; width: 14px; height: 14px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 10px #ef4444; animation: pulse-animation 1.5s infinite;"></div>`,
        className: "det-icon",
        iconSize: [14, 14],
        iconAnchor: [7, 7],
      });
      const marker = L.marker([d.latitude, d.longitude], { icon: detIcon });
      marker.bindPopup(`
        <div style="font-family: system-ui; font-size: 12px; color: #111;">
          <b style="color: #ef4444;">Crop Disease Detected</b><br/>
          <b>Crop:</b> ${d.crop}<br/>
          <b>Disease:</b> ${d.disease}<br/>
          <b>Confidence:</b> ${(d.confidence * 100).toFixed(0)}%<br/>
          <b>Coords:</b> ${d.latitude.toFixed(5)}, ${d.longitude.toFixed(5)}
        </div>
      `);
      detectionLayerRef.current?.addLayer(marker);
    });
  }, [detections]);

  // Update Active Drawing Layer
  useEffect(() => {
    if (!drawingLayerRef.current || !mapInstanceRef.current) return;
    drawingLayerRef.current.clearLayers();

    if (drawnPoints.length > 0) {
      drawnPoints.forEach(([lat, lon]) => {
        const marker = L.circleMarker([lat, lon], {
          radius: 5,
          color: "#f59e0b",
          fillColor: "#fbbf24",
          fillOpacity: 1,
        });
        drawingLayerRef.current?.addLayer(marker);
      });

      if (drawnPoints.length > 1) {
        const polyline = L.polyline(drawnPoints, {
          color: "#f59e0b",
          dashArray: "4, 4",
          weight: 2,
        });
        drawingLayerRef.current?.addLayer(polyline);
      }
    }
  }, [drawnPoints]);

  return (
    <div className="map-wrapper">
      <div ref={mapContainerRef} style={{ width: "100%", height: "100%" }} />

      {/* Floating telemetry HUD over map */}
      <div className="map-floating-panel">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ fontSize: "0.75rem", fontWeight: 700, letterSpacing: "0.05em", color: "var(--text-dim)", textTransform: "uppercase" }}>
            Real-time Position
          </span>
          <span className="badge badge-connected" style={{ padding: "2px 6px", fontSize: "0.68rem" }}>
            GPS FIXED
          </span>
        </div>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", color: "var(--text-main)" }}>
          Lat: {telemetry.lat != null ? telemetry.lat.toFixed(6) : "—"}<br />
          Lon: {telemetry.lon != null ? telemetry.lon.toFixed(6) : "—"}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-muted)", borderTop: "1px solid var(--border-color)", paddingTop: 6 }}>
          <span>ALT: <b>{telemetry.relative_alt_m.toFixed(1)}m</b></span>
          <span>SPD: <b>{telemetry.ground_speed_mps.toFixed(1)}m/s</b></span>
          <span>HDG: <b>{Math.round(telemetry.heading_deg)}°</b></span>
        </div>
      </div>

      {/* Map Control Tools (top right) */}
      <div className="map-floating-right">
        <button
          className={`btn btn-sm ${followDrone ? "btn-primary" : "btn-secondary"}`}
          onClick={() => setFollowDrone(!followDrone)}
          title="Toggle Auto-Follow Drone"
        >
          <Crosshair size={14} />
          {followDrone ? "Following" : "Free Pan"}
        </button>

        {/* Free Map Source Selector */}
        <div style={{ position: "relative" }}>
          <button
            className="btn btn-sm btn-secondary"
            onClick={() => setShowLayerSelector(!showLayerSelector)}
            title="Choose Free Open-Source Map Provider (0 API Key Required)"
          >
            <Layers size={14} />
            <span>Map: {mapSource.toUpperCase()}</span>
          </button>

          {showLayerSelector && (
            <div
              style={{
                position: "absolute",
                top: "100%",
                right: 0,
                marginTop: 6,
                background: "rgba(13, 18, 31, 0.95)",
                backdropFilter: "blur(12px)",
                border: "1px solid var(--border-color)",
                borderRadius: "var(--radius-md)",
                padding: "8px",
                display: "flex",
                flexDirection: "column",
                gap: 4,
                width: 230,
                zIndex: 2000,
                boxShadow: "0 10px 25px rgba(0,0,0,0.6)",
              }}
            >
              <div style={{ fontSize: "0.68rem", fontWeight: 700, color: "var(--text-dim)", textTransform: "uppercase", padding: "4px 8px" }}>
                100% Free Tile Providers (No Key)
              </div>
              {(Object.keys(FREE_MAP_LAYERS) as FreeMapSource[]).map((src) => (
                <button
                  key={src}
                  onClick={() => setFreeMapSource(src)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "6px 10px",
                    background: mapSource === src ? "rgba(16, 185, 129, 0.2)" : "transparent",
                    color: mapSource === src ? "#34d399" : "var(--text-main)",
                    border: "none",
                    borderRadius: "var(--radius-sm)",
                    cursor: "pointer",
                    fontSize: "0.78rem",
                    textAlign: "left",
                    transition: "var(--transition)",
                  }}
                >
                  <span>{FREE_MAP_LAYERS[src].name}</span>
                  {mapSource === src && <span style={{ fontSize: "0.85rem" }}>✓</span>}
                </button>
              ))}
            </div>
          )}
        </div>

        <div style={{ display: "flex", gap: 4 }}>
          <button
            className="btn btn-sm btn-secondary"
            onClick={() => mapInstanceRef.current?.zoomIn()}
            title="Zoom In"
          >
            <ZoomIn size={14} />
          </button>
          <button
            className="btn btn-sm btn-secondary"
            onClick={() => mapInstanceRef.current?.zoomOut()}
            title="Zoom Out"
          >
            <ZoomOut size={14} />
          </button>
        </div>
      </div>
    </div>
  );
};
