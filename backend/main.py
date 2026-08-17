"""
Real-Time Fatigue & Posture Tracker — FastAPI WebSocket Backend
================================================================
Receives base64-encoded JPEG frames from the React frontend over WebSocket,
runs MediaPipe Face Mesh (EAR) and Pose (neck-lean angle) detection,
annotates the frame with OpenCV, and streams back annotated frame + metrics.

Launch:  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import base64
import json
import math

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ──────────────────────────────────────────────────────────────────────
# MediaPipe solution references (module-level).
# Actual model instances are created PER WebSocket connection inside
# the endpoint to avoid thread-safety / AttributeError issues.
# ──────────────────────────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ── Face Mesh 468-point eye landmark indices ─────────────────────────
#   Each eye uses 6 points:  P1, P2, P3, P4, P5, P6
#     Horizontal axis: P1 ↔ P4     Vertical axes: P2 ↔ P6, P3 ↔ P5
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ── Pose landmark indices (shoulder & ear) ───────────────────────────
L_SHOULDER, R_SHOULDER = 11, 12
L_EAR, R_EAR = 7, 8


# ──────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def _ear(landmarks, indices, w, h):
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.

    EAR = (||P2-P6|| + ||P3-P5||) / (2 · ||P1-P4||)
    Returns ~0.25-0.35 for open eyes, drops toward 0 when closing.
    """
    p = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    v1 = math.dist(p[1], p[5])
    v2 = math.dist(p[2], p[4])
    hz = math.dist(p[0], p[3])
    return (v1 + v2) / (2.0 * hz) if hz > 1e-6 else 0.30


def _neck_angle(pose_lm, w, h):
    """
    Compute the angle of the shoulder→ear vector from the vertical axis.

    Uses midpoints of both shoulders and both ears for stability.
    Returns degrees — larger values indicate more forward head lean.
    """
    sx = (pose_lm[L_SHOULDER].x + pose_lm[R_SHOULDER].x) / 2.0 * w
    sy = (pose_lm[L_SHOULDER].y + pose_lm[R_SHOULDER].y) / 2.0 * h
    ex = (pose_lm[L_EAR].x + pose_lm[R_EAR].x) / 2.0 * w
    ey = (pose_lm[L_EAR].y + pose_lm[R_EAR].y) / 2.0 * h
    dx = ex - sx
    dy = sy - ey  # positive = upward (screen-y is inverted)
    return math.degrees(math.atan2(abs(dx), dy)) if abs(dy) > 1e-6 else 90.0


def _annotate(img, face_res, pose_res, ear_val, angle_val, is_drowsy, is_slouching):
    """
    Draw landmarks, posture vectors, heads-up display, and warning
    overlays directly onto the frame using OpenCV.
    """
    h, w = img.shape[:2]

    # ── Eye contours (neon cyan) ─────────────────────────────────────
    if face_res.multi_face_landmarks:
        fl = face_res.multi_face_landmarks[0].landmark
        for idx_set in (LEFT_EYE, RIGHT_EYE):
            pts = np.array(
                [(int(fl[i].x * w), int(fl[i].y * h)) for i in idx_set],
                dtype=np.int32,
            )
            cv2.polylines(img, [pts], True, (0, 255, 200), 2, cv2.LINE_AA)
            cv2.polylines(img, [pts], True, (180, 255, 255), 1, cv2.LINE_AA)

    # ── Posture vector (shoulder→ear) ────────────────────────────────
    if pose_res.pose_landmarks:
        pl = pose_res.pose_landmarks.landmark
        sh = (
            int((pl[L_SHOULDER].x + pl[R_SHOULDER].x) / 2 * w),
            int((pl[L_SHOULDER].y + pl[R_SHOULDER].y) / 2 * h),
        )
        er = (
            int((pl[L_EAR].x + pl[R_EAR].x) / 2 * w),
            int((pl[L_EAR].y + pl[R_EAR].y) / 2 * h),
        )
        vec_col = (0, 80, 255) if is_slouching else (0, 230, 118)
        cv2.line(img, sh, er, vec_col, 5, cv2.LINE_AA)
        cv2.line(img, sh, er, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, sh, 7, vec_col, -1, cv2.LINE_AA)
        cv2.circle(img, er, 7, vec_col, -1, cv2.LINE_AA)
        # Faint vertical reference line
        cv2.line(img, sh, (sh[0], sh[1] - 130), (80, 80, 90), 1, cv2.LINE_AA)

    # ── Semi-transparent HUD bar (top) ───────────────────────────────
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (16, 19, 26), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if ear_val is not None:
        c = (164, 230, 255) if not is_drowsy else (100, 100, 255)
        cv2.putText(img, f"EAR {ear_val:.3f}", (14, 33), font, 0.6, c, 2, cv2.LINE_AA)
    if angle_val is not None:
        c = (91, 255, 161) if not is_slouching else (0, 120, 255)
        cv2.putText(img, f"ANGLE {angle_val:.1f}", (w // 2 - 55, 33), font, 0.6, c, 2, cv2.LINE_AA)

    # Status text (right side of HUD)
    if is_drowsy:
        cv2.putText(img, "DROWSY", (w - 140, 33), font, 0.65, (80, 80, 255), 2, cv2.LINE_AA)
    elif is_slouching:
        cv2.putText(img, "SLOUCH", (w - 140, 33), font, 0.65, (0, 140, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "OPTIMAL", (w - 140, 33), font, 0.65, (0, 230, 180), 2, cv2.LINE_AA)

    # ── Full-frame alert borders ─────────────────────────────────────
    if is_drowsy:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
    if is_slouching:
        t = 8 if not is_drowsy else 6
        cv2.rectangle(img, (3, 3), (w - 4, h - 4), (0, 140, 255), t)

    return img


# ──────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Fatigue & Posture Tracker API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Quick liveness probe."""
    return {"status": "ok", "engine": "mediapipe"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main detection loop.

    Each WebSocket connection gets its OWN MediaPipe model instances
    (no shared mutable state → thread-safe).

    Protocol — client sends JSON:
        {
            "frame":             "<base64 JPEG>",
            "ear_threshold":     0.22,      // optional
            "posture_threshold": 30.0,      // optional
            "consec_frames":     15          // optional
        }

    Server responds with JSON:
        {
            "annotated_frame": "data:image/jpeg;base64,...",
            "ear":             0.2814,
            "is_drowsy":       false,
            "posture_score":   12.4,
            "is_slouching":    false,
            "status":          "OPTIMAL"
        }
    """
    await websocket.accept()

    # Per-connection model instances
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Detection state
    drowsy_counter = 0
    ear_thresh = 0.22
    posture_thresh = 30.0
    consec_limit = 15

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            # ── Dynamic threshold updates from sliders ───────
            if "ear_threshold" in msg:
                ear_thresh = float(msg["ear_threshold"])
            if "posture_threshold" in msg:
                posture_thresh = float(msg["posture_threshold"])
            if "consec_frames" in msg:
                consec_limit = int(msg["consec_frames"])

            # ── Decode base64 JPEG → numpy array ─────────────
            frame_b64 = msg.get("frame", "")
            if not frame_b64:
                continue
            # Strip data-URL prefix if present
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]

            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # ── MediaPipe inference ──────────────────────────
            face_res = face_mesh.process(rgb)
            pose_res = pose.process(rgb)

            ear_val = None
            angle_val = None
            is_drowsy = False
            is_slouching = False

            # EAR computation
            if face_res.multi_face_landmarks:
                fl = face_res.multi_face_landmarks[0].landmark
                ear_l = _ear(fl, LEFT_EYE, w, h)
                ear_r = _ear(fl, RIGHT_EYE, w, h)
                ear_val = (ear_l + ear_r) / 2.0

                if ear_val < ear_thresh:
                    drowsy_counter += 1
                else:
                    drowsy_counter = max(0, drowsy_counter - 1)
                is_drowsy = drowsy_counter >= consec_limit

            # Posture computation
            if pose_res.pose_landmarks:
                angle_val = _neck_angle(pose_res.pose_landmarks.landmark, w, h)
                is_slouching = angle_val > posture_thresh

            # ── Annotate & re-encode ─────────────────────────
            annotated = _annotate(
                img, face_res, pose_res,
                ear_val, angle_val, is_drowsy, is_slouching,
            )
            _, enc_buf = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            out_b64 = base64.b64encode(enc_buf).decode("utf-8")

            # Determine aggregate status
            status = "OPTIMAL"
            if is_drowsy:
                status = "DROWSY"
            elif is_slouching:
                status = "SLOUCHING"

            # ── Send response ────────────────────────────────
            await websocket.send_text(
                json.dumps(
                    {
                        "annotated_frame": f"data:image/jpeg;base64,{out_b64}",
                        "ear": round(ear_val, 4) if ear_val is not None else None,
                        "is_drowsy": is_drowsy,
                        "posture_score": round(angle_val, 2) if angle_val is not None else None,
                        "is_slouching": is_slouching,
                        "status": status,
                    }
                )
            )

    except WebSocketDisconnect:
        pass
    finally:
        face_mesh.close()
        pose.close()
