/**
 * Neural Posture v1.0 — React Dashboard
 * ======================================
 * Translates the Google Stitch UI into a fully wired React component
 * with live webcam → WebSocket → FastAPI backend → annotated frame loop.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";

/* ── Constants ──────────────────────────────────────────────────── */
const WS_URL = "ws://localhost:8000/ws";
const CAPTURE_FPS = 15;
const CAPTURE_INTERVAL_MS = Math.round(1000 / CAPTURE_FPS);
const ALERT_COOLDOWN_MS = 2500; // min gap between audio warnings
const MAX_LOG_ENTRIES = 80;

/* ── Utility: timestamp string ──────────────────────────────────── */
function ts() {
  return new Date().toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/* ═══════════════════════════════════════════════════════════════════
   APP COMPONENT
   ═══════════════════════════════════════════════════════════════════ */

export default function App() {
  /* ── Refs ──────────────────────────────────────────────────────── */
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const annotatedRef = useRef(null);
  const lastAlertRef = useRef(0);
  const logEndRef = useRef(null);

  /* ── State ────────────────────────────────────────────────────── */
  const [connected, setConnected] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [fps, setFps] = useState(0);

  const [metrics, setMetrics] = useState({
    ear: null,
    is_drowsy: false,
    posture_score: null,
    is_slouching: false,
    status: "INITIALIZING",
  });

  const [earThreshold, setEarThreshold] = useState(0.25);
  const [postureThreshold, setPostureThreshold] = useState(15);

  const [logs, setLogs] = useState([
    { time: ts(), message: "System initialized.", type: "success" },
  ]);

  /* ── Logging helper ───────────────────────────────────────────── */
  const addLog = useCallback((message, type = "default") => {
    setLogs((prev) => [
      ...prev.slice(-(MAX_LOG_ENTRIES - 1)),
      { time: ts(), message, type },
    ]);
  }, []);

  // Auto-scroll log
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  /* ── 1. Initialize Webcam ─────────────────────────────────────── */
  useEffect(() => {
    let stream = null;

    async function init() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setCameraReady(true);
          addLog("Camera CAM_01 connected.", "success");
        }
      } catch (err) {
        addLog(`Camera error: ${err.message}`, "error");
      }
    }

    init();

    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, [addLog]);

  /* ── 2. WebSocket Connection (with auto-reconnect) ────────────── */
  useEffect(() => {
    let ws = null;
    let reconnectTimer = null;
    let alive = true;

    function connect() {
      if (!alive) return;
      ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        addLog("WebSocket connected to backend.", "success");
        addLog("Calibration loaded — monitoring active.", "primary");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setMetrics({
            ear: data.ear,
            is_drowsy: data.is_drowsy,
            posture_score: data.posture_score,
            is_slouching: data.is_slouching,
            status: data.status,
          });
          // Render annotated frame
          if (annotatedRef.current && data.annotated_frame) {
            annotatedRef.current.src = data.annotated_frame;
          }
        } catch {
          /* ignore malformed messages */
        }
      };

      ws.onclose = () => {
        setConnected(false);
        wsRef.current = null;
        if (alive) {
          addLog("WebSocket disconnected — reconnecting…", "warning");
          reconnectTimer = setTimeout(connect, 3000);
        }
      };

      ws.onerror = () => ws.close();
    }

    connect();

    return () => {
      alive = false;
      clearTimeout(reconnectTimer);
      if (ws) ws.close();
    };
  }, [addLog]);

  /* ── 3. Frame Capture Loop ────────────────────────────────────── */
  useEffect(() => {
    if (!cameraReady || !connected) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas?.getContext("2d");
    let frameCount = 0;
    let fpsStart = performance.now();

    const interval = setInterval(() => {
      if (
        !video ||
        !ctx ||
        !wsRef.current ||
        wsRef.current.readyState !== WebSocket.OPEN
      )
        return;

      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 480;
      canvas.width = vw;
      canvas.height = vh;
      ctx.drawImage(video, 0, 0, vw, vh);

      const b64 = canvas.toDataURL("image/jpeg", 0.70);

      wsRef.current.send(
        JSON.stringify({
          frame: b64,
          ear_threshold: earThreshold,
          posture_threshold: postureThreshold,
        })
      );

      frameCount++;
      const elapsed = performance.now() - fpsStart;
      if (elapsed >= 1000) {
        setFps(Math.round((frameCount * 1000) / elapsed));
        frameCount = 0;
        fpsStart = performance.now();
      }
    }, CAPTURE_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [cameraReady, connected, earThreshold, postureThreshold]);

  /* ── 4. Audible Alert ─────────────────────────────────────────── */
  useEffect(() => {
    const now = Date.now();
    if (now - lastAlertRef.current < ALERT_COOLDOWN_MS) return;
    if (!metrics.is_drowsy && !metrics.is_slouching) return;

    lastAlertRef.current = now;

    try {
      const audioCtx = new (window.AudioContext ||
        window.webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.type = "sine";
      osc.frequency.value = metrics.is_drowsy ? 880 : 660;
      gain.gain.setValueAtTime(0.10, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(
        0.001,
        audioCtx.currentTime + 0.25
      );
      osc.start();
      osc.stop(audioCtx.currentTime + 0.25);
    } catch {
      /* AudioContext may be blocked until user gesture */
    }

    if (metrics.is_drowsy) {
      addLog("DROWSY state detected!", "error");
    }
    if (metrics.is_slouching) {
      addLog("SLOUCHING posture detected.", "warning");
    }
  }, [metrics.is_drowsy, metrics.is_slouching, addLog]);

  /* ── 5. Log threshold changes ─────────────────────────────────── */
  const prevEar = useRef(earThreshold);
  const prevPosture = useRef(postureThreshold);

  useEffect(() => {
    if (prevEar.current !== earThreshold) {
      addLog(`EAR threshold → ${earThreshold.toFixed(2)}`, "default");
      prevEar.current = earThreshold;
    }
  }, [earThreshold, addLog]);

  useEffect(() => {
    if (prevPosture.current !== postureThreshold) {
      addLog(`Posture tolerance → ${postureThreshold}°`, "default");
      prevPosture.current = postureThreshold;
    }
  }, [postureThreshold, addLog]);

  /* ── Derived display values ───────────────────────────────────── */
  const postureLabel = metrics.is_slouching ? "SLOUCHING" : "OPTIMAL";
  const drowsinessLabel = metrics.is_drowsy ? "ALERT" : "MONITORING";

  const videoClasses = [
    "video-container",
    metrics.is_drowsy && "alert-border-red",
    !metrics.is_drowsy && metrics.is_slouching && "alert-border-amber",
  ]
    .filter(Boolean)
    .join(" ");

  /* ═══════════════════════════════════════════════════════════════
     RENDER
     ═══════════════════════════════════════════════════════════════ */
  return (
    <div className="app">
      {/* ── Top Navigation Bar ─────────────────────────────────── */}
      <header className="top-nav">
        <div className="nav-left">
          <h1 className="app-title">
            Real-Time Fatigue And Posture Tracker
          </h1>
        </div>

        <div className="nav-right">
          <div className="nav-icons">
            <button className="icon-btn" title="Sensors">
              <span className="material-symbols-outlined">sensors</span>
            </button>
            <button className="icon-btn" title="Camera">
              <span className="material-symbols-outlined">videocam</span>
            </button>
          </div>

          <div className="nav-status">
            <button
              className={`status-chip ${
                connected ? "connected" : "disconnected"
              }`}
            >
              <span className="status-dot" />
              {connected ? "CONNECTED" : "DISCONNECTED"}
            </button>
          </div>
        </div>
      </header>

      {/* ── Dashboard Grid ─────────────────────────────────────── */}
      <main className="dashboard">
        {/* ──── Left / Center Column ──────────────────────────── */}
        <div className="main-content">
          {/* Video Feed */}
          <div className={videoClasses}>
            {/* Hidden native elements */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ display: "none" }}
            />
            <canvas ref={canvasRef} style={{ display: "none" }} />

            {/* Annotated frame from backend */}
            <img
              ref={annotatedRef}
              className="annotated-feed"
              alt="Annotated feed"
            />

            {/* Placeholder before camera starts */}
            {!cameraReady && (
              <div className="feed-placeholder">
                <span className="material-symbols-outlined feed-icon">
                  videocam
                </span>
                <p>INITIALIZING CAMERA…</p>
              </div>
            )}

            {/* Overlays */}
            <div className="scanlines" />

            <div className="feed-badges">
              <div className="feed-badge">
                <span className="live-dot" />
                CAM_01
              </div>
              <div className="feed-badge">{fps} FPS</div>
            </div>

            <div className="tracking-frame">
              <div className="tracking-frame-inner">
                <div className="corner tl" />
                <div className="corner tr" />
                <div className="corner bl" />
                <div className="corner br" />
              </div>
            </div>
          </div>

          {/* Metric Cards */}
          <div className="metrics-row">
            {/* Posture Status */}
            <div
              className={`metric-card ${
                metrics.is_slouching ? "alert" : "good"
              }`}
            >
              <div
                className={`metric-accent ${
                  metrics.is_slouching ? "accent-red" : "accent-green"
                }`}
              />
              <h3 className="metric-label">Posture Status</h3>
              <span
                className={`metric-value ${
                  metrics.is_slouching
                    ? "text-alert glow-red"
                    : "text-good glow-green"
                }`}
              >
                {postureLabel}
              </span>
              <div className="metric-footer">
                <span>
                  Angle:{" "}
                  {metrics.posture_score !== null
                    ? `${metrics.posture_score.toFixed(1)}°`
                    : "—"}
                </span>
                <span>Threshold: {postureThreshold}°</span>
              </div>
            </div>

            {/* Drowsiness Alert */}
            <div
              className={`metric-card ${metrics.is_drowsy ? "alert" : ""}`}
            >
              <div
                className={`metric-accent ${
                  metrics.is_drowsy ? "accent-red" : "accent-neutral"
                }`}
              />
              <h3 className="metric-label">Drowsiness Alert</h3>
              <span
                className={`metric-value ${
                  metrics.is_drowsy ? "text-alert glow-red" : "text-neutral"
                }`}
              >
                {drowsinessLabel}
              </span>
              <div className="metric-footer">
                <span>
                  EAR:{" "}
                  {metrics.ear !== null ? metrics.ear.toFixed(3) : "—"}
                </span>
                <span>Threshold: {earThreshold.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* ──── Right Sidebar ─────────────────────────────────── */}
        <div className="sidebar">
          {/* Sensitivity Controls */}
          <div className="sidebar-card">
            <div className="card-header">
              <span className="material-symbols-outlined header-icon">
                tune
              </span>
              <h2 className="card-title">Sensitivity Controls</h2>
            </div>

            {/* EAR Threshold Slider */}
            <div className="slider-group">
              <div className="slider-header">
                <label>Fatigue Sensitivity (EAR Threshold)</label>
                <span className="slider-value">
                  {earThreshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.5"
                step="0.01"
                value={earThreshold}
                onChange={(e) =>
                  setEarThreshold(parseFloat(e.target.value))
                }
              />
              <div className="slider-labels">
                <span>0.1 (Strict)</span>
                <span>0.5 (Lenient)</span>
              </div>
            </div>

            {/* Posture Angle Slider */}
            <div className="slider-group">
              <div className="slider-header">
                <label>Posture Angle Tolerance</label>
                <span className="slider-value">{postureThreshold}°</span>
              </div>
              <input
                type="range"
                min="5"
                max="45"
                step="1"
                value={postureThreshold}
                onChange={(e) =>
                  setPostureThreshold(parseInt(e.target.value, 10))
                }
              />
              <div className="slider-labels">
                <span>5°</span>
                <span>45°</span>
              </div>
            </div>

            <button
              className="calibrate-btn"
              onClick={() =>
                addLog(
                  `Calibration applied — EAR: ${earThreshold.toFixed(
                    2
                  )}, Angle: ${postureThreshold}°`,
                  "primary"
                )
              }
            >
              <span
                className="material-symbols-outlined"
                style={{ fontSize: 16 }}
              >
                refresh
              </span>
              APPLY CALIBRATION
            </button>
          </div>

          {/* Telemetry Log */}
          <div className="sidebar-card telemetry-card">
            <div className="card-header">
              <span className="material-symbols-outlined header-icon neutral">
                terminal
              </span>
              <h2 className="card-title neutral">Telemetry Log</h2>
            </div>

            <div className="log-container">
              {logs.map((log, i) => (
                <div key={i} className={`log-entry ${log.type}`}>
                  [{log.time}] {log.message}
                </div>
              ))}
              <div className="log-cursor" ref={logEndRef}>
                _
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
