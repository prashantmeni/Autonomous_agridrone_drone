import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TelemetryData, ChargingStatus } from "../types";

interface Drone3DProps {
  telemetry: TelemetryData;
  charging: ChargingStatus | null;
  registerRecenter: (fn: () => void) => void;
}

const LANDING_Y = 0.112;
const D2R = Math.PI / 180;

export const Drone3D: React.FC<Drone3DProps> = ({ telemetry, charging, registerRecenter }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const telRef = useRef(telemetry);
  const chgRef = useRef(charging);
  telRef.current = telemetry;
  chgRef.current = charging;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    // ------------------------------------------------------------- scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x04060c);
    scene.fog = new THREE.Fog(0x04060c, 16, 48);

    const camera = new THREE.PerspectiveCamera(52, 1, 0.1, 120);
    camera.position.set(2.1, 1.35, 2.7);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.5, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 1.0;
    controls.maxDistance = 30;
    controls.maxPolarAngle = Math.PI * 0.495;

    scene.add(new THREE.HemisphereLight(0x9db8ff, 0x0b1018, 0.85));
    const sun = new THREE.DirectionalLight(0xffffff, 1.7);
    sun.position.set(5, 8, 3);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.left = -8;
    sun.shadow.camera.right = 8;
    sun.shadow.camera.top = 8;
    sun.shadow.camera.bottom = -8;
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 24;
    scene.add(sun);
    const rim = new THREE.DirectionalLight(0x34d399, 0.35);
    rim.position.set(-4, 2, -5);
    scene.add(rim);

    const grid = new THREE.GridHelper(40, 40, 0x1f6f56, 0x0e2231);
    grid.position.y = 0.001;
    scene.add(grid);
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(40, 40),
      new THREE.ShadowMaterial({ opacity: 0.35 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // -------------------------------------------------------- landing pad
    const padDisk = new THREE.Mesh(
      new THREE.CylinderGeometry(0.45, 0.45, 0.012, 48),
      new THREE.MeshStandardMaterial({ color: 0x111722, roughness: 0.9 })
    );
    padDisk.position.y = 0.006;
    padDisk.receiveShadow = true;
    scene.add(padDisk);
    const ringMat = new THREE.MeshStandardMaterial({
      color: 0x0b2c24,
      emissive: 0x10b981,
      emissiveIntensity: 0.6,
      roughness: 0.4,
    });
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.45, 0.014, 12, 64), ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.014;
    scene.add(ring);

    // ------------------------------------- F450 frame + A2212 KV1000 + 1045 props
    // Modelled after the physical build: red/white diagonal truss arms,
    // white curved landing gear, gold motor bells, black 1045 props,
    // GPS puck on mast, blue FC glow, orange battery, ESCs on arms.
    const drone = new THREE.Group();
    const plateMat = new THREE.MeshStandardMaterial({ color: 0xd8232a, metalness: 0.2, roughness: 0.55 });
    const pdbMat = new THREE.MeshStandardMaterial({ color: 0x0b0d12, roughness: 0.8 });
    const armRed = new THREE.MeshStandardMaterial({ color: 0xd8232a, roughness: 0.5 });
    const armWhite = new THREE.MeshStandardMaterial({ color: 0xeeeae2, roughness: 0.5 });
    const whiteMat = new THREE.MeshStandardMaterial({ color: 0xf5f5f0, roughness: 0.6 });
    const motorBaseMat = new THREE.MeshStandardMaterial({ color: 0xb8bcc4, metalness: 0.85, roughness: 0.3 });
    const bellMat = new THREE.MeshStandardMaterial({ color: 0xd08a2e, metalness: 0.75, roughness: 0.35 });
    const propMat = new THREE.MeshStandardMaterial({ color: 0x0d0d10, roughness: 0.65, metalness: 0.1 });
    const legMat = new THREE.MeshStandardMaterial({ color: 0xf0efe8, roughness: 0.55 });
    const footMat = new THREE.MeshStandardMaterial({ color: 0x33373d, roughness: 0.9 });
    const puckMat = new THREE.MeshStandardMaterial({ color: 0x101216, roughness: 0.5 });
    const fcMat = new THREE.MeshStandardMaterial({ color: 0x1b4fa5, roughness: 0.5 });
    const escMat = new THREE.MeshStandardMaterial({ color: 0x1f4a2a, roughness: 0.7 });
    const battMat = new THREE.MeshStandardMaterial({ color: 0xcc5a1e, roughness: 0.6 });
    const wireMat = new THREE.MeshStandardMaterial({ color: 0xc22f2f, roughness: 0.6 });
    const blueLedMat = new THREE.MeshStandardMaterial({ color: 0x2255ff, emissive: 0x3366ff, emissiveIntensity: 2.2 });
    const chevMat = new THREE.MeshStandardMaterial({ color: 0x22c55e, emissive: 0x16a34a, emissiveIntensity: 1.4 });

    const mkShadow = (m: THREE.Mesh) => {
      m.castShadow = true;
      return m;
    };

    // lightweight routed wire (thin tube along a smooth curve)
    const mkWire = (pts: [number, number, number][], r: number, mat: THREE.Material, seg = 16) => {
      const curve = new THREE.CatmullRomCurve3(pts.map((p) => new THREE.Vector3(p[0], p[1], p[2])));
      return new THREE.Mesh(new THREE.TubeGeometry(curve, seg, r, 5, false), mat);
    };

    // center plates (octagonal like the F450) + black PDB layer between them
    const bottomPlate = mkShadow(new THREE.Mesh(new THREE.CylinderGeometry(0.095, 0.095, 0.008, 8), plateMat));
    bottomPlate.position.y = -0.012;
    drone.add(bottomPlate);
    const topPlate = mkShadow(new THREE.Mesh(new THREE.CylinderGeometry(0.088, 0.088, 0.008, 8), plateMat));
    topPlate.position.y = 0.014;
    drone.add(topPlate);
    const pdb = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.016, 0.1), pdbMat);
    drone.add(pdb);

    // flight-controller stack: exposed board layers, headers, LEDs, antenna
    const fcBoard = new THREE.Mesh(new THREE.BoxGeometry(0.064, 0.005, 0.054), pdbMat);
    fcBoard.position.y = 0.0205;
    drone.add(fcBoard);
    const fc = mkShadow(new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.01, 0.05), fcMat));
    fc.position.y = 0.028;
    drone.add(fc);
    const pins = new THREE.Mesh(new THREE.BoxGeometry(0.034, 0.007, 0.007), whiteMat);
    pins.position.set(0, 0.036, -0.018);
    drone.add(pins);
    const pins2 = new THREE.Mesh(new THREE.BoxGeometry(0.026, 0.007, 0.007), whiteMat);
    pins2.position.set(-0.004, 0.036, 0.014);
    drone.add(pins2);
    const chip = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.004, 0.02), pdbMat);
    chip.position.set(0.016, 0.035, 0.004);
    drone.add(chip);
    const blueLed = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.005, 0.01), blueLedMat);
    blueLed.position.set(-0.022, 0.035, 0.016);
    drone.add(blueLed);
    const greenLed = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.006, 0.008), chevMat);
    greenLed.position.set(0.03, 0.033, -0.008);
    drone.add(greenLed);
    const blueGlow = new THREE.PointLight(0x3366ff, 0.9, 0.7, 2);
    blueGlow.position.set(0, 0.055, 0);
    drone.add(blueGlow);

    // short exposed jumper wires around the stack + telemetry antenna wire
    drone.add(mkWire([[0, 0.039, -0.016], [0.012, 0.043, -0.006], [0.016, 0.038, 0.002]], 0.0015, whiteMat, 10));
    drone.add(mkWire([[-0.02, 0.038, 0.014], [-0.03, 0.034, 0.022], [-0.036, 0.024, 0.03]], 0.0015, wireMat, 10));
    drone.add(mkWire([[0.028, 0.03, 0.026], [0.05, 0.07, 0.045], [0.058, 0.115, 0.052]], 0.0015, pdbMat, 14));

    // battery strapped under the frame + red XT60-style wire loop
    const batt = mkShadow(new THREE.Mesh(new THREE.BoxGeometry(0.075, 0.034, 0.05), battMat));
    batt.position.y = -0.034;
    drone.add(batt);
    const strap = new THREE.Mesh(new THREE.BoxGeometry(0.079, 0.037, 0.01), pdbMat);
    strap.position.y = -0.034;
    drone.add(strap);
    const wireLoop = new THREE.Mesh(new THREE.TorusGeometry(0.016, 0.003, 6, 14, Math.PI), wireMat);
    wireLoop.position.set(0, -0.05, -0.03);
    wireLoop.rotation.x = Math.PI / 2;
    drone.add(wireLoop);

    // battery -> FC power wires routed around the plate rim (red + black)
    drone.add(mkWire([[0.03, -0.017, 0.012], [0.07, -0.006, 0.05], [0.078, 0.01, 0.052], [0.03, 0.02, 0.036]], 0.002, wireMat));
    drone.add(mkWire([[0.024, -0.017, 0.004], [0.062, -0.008, 0.042], [0.07, 0.008, 0.046], [0.024, 0.018, 0.032]], 0.002, pdbMat));

    // XT60-style connector beside the battery + short cable (visual only)
    const xt60 = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.013, 0.014), wireMat);
    xt60.position.set(0.052, -0.014, -0.008);
    xt60.rotation.y = 0.4;
    drone.add(xt60);
    const xt60Face = new THREE.Mesh(new THREE.BoxGeometry(0.004, 0.011, 0.012), pdbMat);
    xt60Face.position.set(0.061, -0.014, -0.0045);
    xt60Face.rotation.y = 0.4;
    drone.add(xt60Face);
    drone.add(mkWire([[0.04, -0.05, -0.028], [0.05, -0.03, -0.02], [0.052, -0.016, -0.01]], 0.0022, wireMat, 10));

    // GPS mast + round M10-style puck, centered above the electronics
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.0025, 0.0025, 0.078, 8), puckMat);
    mast.position.set(0, 0.056, 0.022);
    drone.add(mast);
    const puck = mkShadow(new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.033, 0.015, 24), puckMat));
    puck.position.set(0, 0.1, 0.022);
    drone.add(puck);
    const puckBand = new THREE.Mesh(new THREE.CylinderGeometry(0.0335, 0.0335, 0.004, 24), wireMat);
    puckBand.position.set(0, 0.097, 0.022);
    drone.add(puckBand);
    const puckTop = new THREE.Mesh(new THREE.CylinderGeometry(0.026, 0.028, 0.004, 24), puckMat);
    puckTop.position.set(0, 0.108, 0.022);
    drone.add(puckTop);
    const puckLogo = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.002, 16), whiteMat);
    puckLogo.position.set(0, 0.1105, 0.022);
    drone.add(puckLogo);
    const puckLed = new THREE.Mesh(new THREE.SphereGeometry(0.003, 8, 8), new THREE.MeshBasicMaterial({ color: 0xff4444 }));
    puckLed.position.set(0.033, 0.1, 0.022);
    drone.add(puckLed);

    // green chevron marks the front (-Z)
    const chev = new THREE.Mesh(new THREE.BoxGeometry(0.022, 0.004, 0.016), chevMat);
    chev.position.set(0, 0.019, -0.072);
    drone.add(chev);

    const propMeshes: THREE.Group[] = [];
    const propDirs: number[] = [];
    const corners: [number, number][] = [
      [1, 1],
      [1, -1],
      [-1, 1],
      [-1, -1],
    ];

    // one 1045 blade: scimitar curve, tapered chord, rounded tip, pitched ~10 deg
    const mkBlade = () => {
      const sh = new THREE.Shape();
      sh.moveTo(0.014, -0.023);
      sh.quadraticCurveTo(0.05, -0.024, 0.086, -0.015);
      sh.quadraticCurveTo(0.112, -0.009, 0.125, -0.004);
      sh.quadraticCurveTo(0.131, 0.0, 0.125, 0.005);
      sh.quadraticCurveTo(0.1, 0.011, 0.068, 0.015);
      sh.quadraticCurveTo(0.04, 0.02, 0.015, 0.022);
      sh.quadraticCurveTo(0.006, 0.0, 0.014, -0.023);
      const geo = new THREE.ExtrudeGeometry(sh, { depth: 0.0035, bevelEnabled: false, curveSegments: 5 });
      const blade = new THREE.Group();
      const mesh = mkShadow(new THREE.Mesh(geo, propMat));
      mesh.rotation.x = -Math.PI / 2 + 0.18;
      blade.add(mesh);
      return blade;
    };

    corners.forEach(([sx, sz]) => {
      const armMat = sx * sz > 0 ? armRed : armWhite;
      const arm = new THREE.Group();
      arm.rotation.y = -Math.atan2(sz, sx);
      const len = 0.19;

      // truss lattice: thin rails + dense zigzag struts (open, never solid)
      const lower = mkShadow(new THREE.Mesh(new THREE.BoxGeometry(len, 0.006, 0.02), armMat));
      lower.position.set(0.035 + len / 2, 0.002, 0);
      arm.add(lower);
      const upper = mkShadow(new THREE.Mesh(new THREE.BoxGeometry(len * 0.86, 0.006, 0.015), armMat));
      upper.position.set(0.035 + (len * 0.86) / 2, 0.026, 0);
      arm.add(upper);
      for (let k = 0; k < 9; k++) {
        const strut = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.004, 0.013), armMat);
        strut.position.set(0.055 + k * 0.0195, 0.014, 0);
        strut.rotation.z = k % 2 === 0 ? 0.62 : -0.62;
        arm.add(strut);
      }

      // ESC zip-tied under the arm
      const esc = mkShadow(new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.009, 0.018), escMat));
      esc.position.set(0.11, -0.007, 0);
      arm.add(esc);
      const zip = new THREE.Mesh(new THREE.BoxGeometry(0.006, 0.014, 0.024), pdbMat);
      zip.position.set(0.11, -0.004, 0);
      arm.add(zip);

      // motor -> ESC -> arm-root wires routed along the arm (red + black)
      const wA: [number, number, number][] = [
        [0.22, 0.038, 0.005],
        [0.17, 0.016, 0.009],
        [0.115, -0.01, 0.008],
        [0.05, -0.007, 0.004],
      ];
      const wB = wA.map((p) => [p[0], p[1] - 0.003, p[2] - 0.006] as [number, number, number]);
      arm.add(mkWire(wA, 0.0016, wireMat));
      arm.add(mkWire(wB, 0.0016, pdbMat));

      // motor mount + A2212 stator (static half)
      const mount = mkShadow(new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.006, 0.05), armMat));
      mount.position.set(0.225, 0.032, 0);
      arm.add(mount);
      const stator = mkShadow(new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.015, 0.009, 20), motorBaseMat));
      stator.position.set(0.225, 0.04, 0);
      arm.add(stator);

      // spinning group: gold A2212 bell + adapter shaft + 1045 prop + nut
      const prop = new THREE.Group();
      prop.position.set(0.225, 0.044, 0);
      const bell = mkShadow(new THREE.Mesh(new THREE.CylinderGeometry(0.0135, 0.0135, 0.022, 20), bellMat));
      bell.position.y = 0.011;
      const shaft = mkShadow(new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, 0.018, 10), motorBaseMat));
      shaft.position.y = 0.03;
      const hub = mkShadow(new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.009, 0.01, 14), propMat));
      hub.position.y = 0.026;
      const nut = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.006, 8), motorBaseMat);
      nut.position.y = 0.036;
      const blade1 = mkBlade();
      blade1.position.y = 0.027;
      const blade2 = mkBlade();
      blade2.position.y = 0.027;
      blade2.rotation.y = Math.PI;
      prop.add(bell, shaft, hub, nut, blade1, blade2);
      arm.add(prop);
      propMeshes.push(prop);
      propDirs.push(sx * sz > 0 ? 1 : -1);

      drone.add(arm);
    });

    // four white curved landing legs splaying out from the bottom plate
    for (const [sx, sz] of corners) {
      const a = Math.atan2(sz, sx);
      const dir = new THREE.Vector3(Math.cos(a), 0, Math.sin(a));
      const p0 = dir.clone().multiplyScalar(0.072).setY(-0.014);
      const p1 = dir.clone().multiplyScalar(0.112).setY(-0.05);
      const p2 = dir.clone().multiplyScalar(0.134).setY(-0.096);
      const leg = mkShadow(
        new THREE.Mesh(new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(p0, p1, p2), 14, 0.0075, 8, false), legMat)
      );
      drone.add(leg);
      const foot = new THREE.Mesh(new THREE.CylinderGeometry(0.013, 0.015, 0.008, 14), footMat);
      foot.position.copy(p2).setY(-0.096);
      drone.add(foot);
    }

    // rear status LED (colored by flight state in the animation loop)
    const ledMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const led = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 12), ledMat);
    led.position.set(0, 0.022, 0.078);
    drone.add(led);
    scene.add(drone);

    // ------------------------------------------------------ animation state
    const pos = new THREE.Vector3(0, LANDING_Y, 0);
    const vel = new THREE.Vector3();
    const accF = { x: 0, y: 0 };
    let propSpeed = 0;
    let baroBase: number | null = null;
    const targetQ = new THREE.Quaternion();
    const eul = new THREE.Euler(0, 0, 0, "YXZ");

    registerRecenter(() => {
      pos.set(0, LANDING_Y, 0);
      vel.set(0, 0, 0);
      accF.x = 0;
      accF.y = 0;
      baroBase = null;
    });

    const clock = new THREE.Clock();
    let raf = 0;

    const animate = () => {
      raf = requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.1);
      const t = telRef.current;
      const chargingNow = chgRef.current?.charging ?? false;

      const heading =
        t.gps_fix > 0 && t.heading_deg ? t.heading_deg : (t.yaw_deg ?? t.heading_deg) || 0;
      const hdgRad = heading * D2R;

      // orientation: nose=-Z; yaw clockwise, pitch nose-up, roll right-down
      eul.set((t.pitch_deg || 0) * D2R, -hdgRad, -(t.roll_deg || 0) * D2R);
      targetQ.setFromEuler(eul);
      drone.quaternion.slerp(targetQ, Math.min(1, dt * 12));

      // altitude: baro relative to screen-open baseline; deadband applies at
      // all times so bench/baro noise never lifts the model off the pad
      const baro = t.alt_baro_m;
      if (baroBase == null && baro != null && baro !== 0) baroBase = baro;
      const relAlt =
        baroBase != null && baro != null ? baro - baroBase : t.relative_alt_m || 0;
      const relAltEff = Math.abs(relAlt) < 0.3 ? 0 : relAlt;
      const targetY = Math.max(relAltEff, LANDING_Y);
      pos.y += (targetY - pos.y) * Math.min(1, dt * 4);

      // horizontal: trust velocity only with a real GPS fix; otherwise
      // accel dead-reckon through a low-pass (~8 Hz) so motor vibration
      // averages out and only genuine sustained motion moves the model
      const gs = t.ground_speed_mps || 0;
      if (t.gps_fix >= 3 && Math.abs(gs) > 0.25) {
        vel.x = Math.sin(hdgRad) * gs;
        vel.z = -Math.cos(hdgRad) * gs;
      } else {
        const pitchR = (t.pitch_deg || 0) * D2R;
        const rollR = (t.roll_deg || 0) * D2R;
        const ax = (t.accel_x || 0) - 9.80665 * Math.sin(pitchR);
        const ay = (t.accel_y || 0) + 9.80665 * Math.sin(rollR);
        accF.x += (ax - accF.x) * Math.min(1, dt * 8);
        accF.y += (ay - accF.y) * Math.min(1, dt * 8);
        if (Math.abs(accF.x) > 0.6 || Math.abs(accF.y) > 0.6) {
          const north = accF.x * Math.cos(hdgRad) - accF.y * Math.sin(hdgRad);
          const east = accF.x * Math.sin(hdgRad) + accF.y * Math.cos(hdgRad);
          vel.x += east * dt * 1.2;
          vel.z += -north * dt * 1.2;
        }
        const decay = Math.exp(-1.1 * dt);
        vel.x *= decay;
        vel.z *= decay;
        const sp = Math.hypot(vel.x, vel.z);
        if (sp > 5) {
          vel.x *= 5 / sp;
          vel.z *= 5 / sp;
        }
      }
      pos.x += vel.x * dt;
      pos.z += vel.z * dt;
      drone.position.copy(pos);
      if (t.armed) {
        // motor idle vibration
        drone.position.x += (Math.random() - 0.5) * 0.005;
        drone.position.y += (Math.random() - 0.5) * 0.003;
        drone.position.z += (Math.random() - 0.5) * 0.005;
      }

      // propellers: spin when armed, stop when not
      const targetProp = t.armed ? 34 : 0;
      propSpeed += (targetProp - propSpeed) * Math.min(1, dt * (t.armed ? 3 : 1.6));
      for (let i = 0; i < propMeshes.length; i++) {
        propMeshes[i].rotation.y += propDirs[i] * propSpeed * dt;
      }

      // status LED: green armed, amber charging, red idle
      ledMat.color.setHex(t.armed ? 0x34d399 : chargingNow ? 0xf59e0b : 0xef4444);

      // pad glow: amber pulse only while charging
      if (chargingNow) {
        ringMat.emissive.setHex(0xf59e0b);
        ringMat.emissiveIntensity = 0.7 + 0.5 * Math.sin(performance.now() / 300);
      } else {
        ringMat.emissive.setHex(0x10b981);
        ringMat.emissiveIntensity = 0.55;
      }

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // ------------------------------------------------------------ resizing
    const resize = () => {
      const w = mount.clientWidth || 1;
      const h = mount.clientHeight || 1;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(mount);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      scene.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else if (mat) mat.dispose();
      });
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
    };
  }, [registerRecenter]);

  return <div ref={mountRef} className="drone3d-mount" />;
};
