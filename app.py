"""
Real-Time Fatigue & Posture Tracker
====================================
A hackathon-ready application that uses MediaPipe Face Mesh and Pose
to detect drowsiness (via Eye Aspect Ratio) and slouching (via neck
lean angle) in a live webcam feed rendered through streamlit-webrtc.

Run:  streamlit run app.py
"""

import math
import threading
from collections import deque

import av
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ──────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit command)
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fatigue & Posture Tracker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# CUSTOM CSS — polished dark-theme dashboard
# ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Root variables ── */
    :root {
        --bg-primary:   #0f1117;
        --bg-card:      #1a1d29;
        --accent-cyan:  #00d4ff;
        --accent-green: #00e676;
        --accent-red:   #ff3d71;
        --accent-amber: #ffaa00;
        --text-primary: #e8eaf6;
        --text-muted:   #8892b0;
        --border-glow:  rgba(0, 212, 255, 0.15);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main title area ── */
    .main-title {
        text-align: center;
        padding: 1.2rem 0 0.4rem 0;
    }
    .main-title h1 {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #7b61ff 50%, #ff3d71 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
    }
    .main-title p {
        color: var(--text-muted);
        font-size: 0.95rem;
        margin: 0;
    }

    /* ── Metric cards ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-glow);
        border-radius: 14px;
        padding: 1rem 1.6rem;
        min-width: 180px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,212,255,0.12);
    }
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--text-muted);
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-value.cyan   { color: var(--accent-cyan);  }
    .metric-value.green  { color: var(--accent-green); }
    .metric-value.red    { color: var(--accent-red);   }
    .metric-value.amber  { color: var(--accent-amber); }

    /* ── Status badge ── */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 1.1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }
    .status-ok {
        background: rgba(0,230,118,0.12);
        color: var(--accent-green);
        border: 1px solid rgba(0,230,118,0.3);
    }
    .status-warn {
        background: rgba(255,61,113,0.12);
        color: var(--accent-red);
        border: 1px solid rgba(255,61,113,0.3);
        animation: pulse-warn 1.2s infinite;
    }
    @keyframes pulse-warn {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.65; }
    }

    /* ── Sidebar polish ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141824 0%, #0f1117 100%);
    }
    section[data-testid="stSidebar"] .stSlider label {
        color: var(--text-primary) !important;
        font-weight: 500;
    }

    /* ── Footer ── */
    .footer-text {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.75rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────
# CONSTANTS — MediaPipe Face Mesh eye landmark indices (468 model)
# Each eye is defined by 6 points: [P1, P2, P3, P4, P5, P6]
#   P1-P4 = horizontal axis, P2-P6 & P3-P5 = vertical axes
# ──────────────────────────────────────────────────────────────────────
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]

# Pose landmark indices (for posture)
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_EAR       = 7
RIGHT_EAR      = 8


# ──────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def _eye_aspect_ratio(face_landmarks, eye_indices, img_w, img_h):
    """
    Compute the Eye Aspect Ratio (EAR) for one eye.

    EAR = (||P2-P6|| + ||P3-P5||) / (2 · ||P1-P4||)

    A low EAR indicates the eye is closing.
    """
    pts = [
        (face_landmarks[i].x * img_w, face_landmarks[i].y * img_h)
        for i in eye_indices
    ]
    vert_a = math.dist(pts[1], pts[5])
    vert_b = math.dist(pts[2], pts[4])
    horiz  = math.dist(pts[0], pts[3])
    if horiz < 1e-6:
        return 0.30  # fallback to "open" if degenerate
    return (vert_a + vert_b) / (2.0 * horiz)


def _neck_lean_angle(pose_landmarks, img_w, img_h):
    """
    Compute the neck-lean angle from the vertical.

    Draws a vector from the midpoint of both shoulders to the midpoint
    of both ears, then measures its deviation from the vertical axis.
    A larger angle means more forward lean (slouching).
    """
    ls = pose_landmarks[LEFT_SHOULDER]
    rs = pose_landmarks[RIGHT_SHOULDER]
    le = pose_landmarks[LEFT_EAR]
    re = pose_landmarks[RIGHT_EAR]

    # Midpoints
    mid_sh_x = (ls.x + rs.x) / 2.0 * img_w
    mid_sh_y = (ls.y + rs.y) / 2.0 * img_h
    mid_ear_x = (le.x + re.x) / 2.0 * img_w
    mid_ear_y = (le.y + re.y) / 2.0 * img_h

    dx = mid_ear_x - mid_sh_x
    dy = mid_sh_y - mid_ear_y  # positive = upward (screen-y is inverted)

    if abs(dy) < 1e-6:
        return 90.0
    return math.degrees(math.atan2(abs(dx), dy))


def _draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=0.55):
    """Draw a semi-transparent rounded rectangle as an overlay."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    # Four corner circles + two filling rectangles
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_text_with_bg(img, text, org, scale, color, bg_color, thickness=2):
    """Draw text with a rounded semi-transparent background pill."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    pad = 10
    _draw_rounded_rect(
        img,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        bg_color,
        radius=8,
        thickness=-1,
        alpha=0.65,
    )
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────
# VIDEO PROCESSOR — runs in a background thread via streamlit-webrtc
# ──────────────────────────────────────────────────────────────────────

class FatiguePostureProcessor(VideoProcessorBase):
    """
    Processes each video frame through MediaPipe Face Mesh and Pose,
    computes fatigue (EAR) and posture (neck-lean angle) metrics, and
    draws visual overlays / warnings onto the frame.

    Threshold parameters are written from the Streamlit main thread and
    read here — all behind a threading lock for safety.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Tuneable thresholds (written by the UI thread)
        self.ear_threshold: float = 0.22
        self.posture_angle_threshold: float = 30.0
        self.drowsy_consec_frames: int = 15

        # Internal counters
        self._drowsy_counter: int = 0
        self._is_drowsy: bool = False
        self._is_slouching: bool = False
        self._current_ear: float = 0.0
        self._current_angle: float = 0.0
        self._frame_count: int = 0

        # Rolling history for a mini sparkline / smoothing
        self._ear_history: deque = deque(maxlen=90)
        self._angle_history: deque = deque(maxlen=90)

        # ── MediaPipe model initialisation (NO @st.cache) ──
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose = mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ─── Main per-frame callback ────────────────────────────────────
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._frame_count += 1

        # Snapshot thresholds
        with self._lock:
            ear_thr   = self.ear_threshold
            angle_thr = self.posture_angle_threshold
            consec    = self.drowsy_consec_frames

        # ── 1. FACE MESH — EAR computation ──────────────────────────
        face_results = self._face_mesh.process(rgb)
        ear = None

        if face_results.multi_face_landmarks:
            lm = face_results.multi_face_landmarks[0].landmark
            ear_l = _eye_aspect_ratio(lm, LEFT_EYE_IDX, w, h)
            ear_r = _eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
            ear = (ear_l + ear_r) / 2.0
            self._current_ear = ear
            self._ear_history.append(ear)

            # Draw eye contours with a neon glow effect
            for indices in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
                pts = np.array(
                    [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices],
                    dtype=np.int32,
                )
                # Outer glow
                cv2.polylines(img, [pts], True, (0, 255, 200), 3, cv2.LINE_AA)
                # Inner bright line
                cv2.polylines(img, [pts], True, (200, 255, 255), 1, cv2.LINE_AA)

            # Drowsiness counter logic
            if ear < ear_thr:
                self._drowsy_counter += 1
            else:
                self._drowsy_counter = max(0, self._drowsy_counter - 1)

            self._is_drowsy = self._drowsy_counter >= consec

        # ── 2. POSE — posture / slouch detection ────────────────────
        pose_results = self._pose.process(rgb)
        angle = None

        if pose_results.pose_landmarks:
            plm = pose_results.pose_landmarks.landmark
            angle = _neck_lean_angle(plm, w, h)
            self._current_angle = angle
            self._angle_history.append(angle)
            self._is_slouching = angle > angle_thr

            # Draw shoulder → ear vector
            mid_sh = (
                int((plm[LEFT_SHOULDER].x + plm[RIGHT_SHOULDER].x) / 2 * w),
                int((plm[LEFT_SHOULDER].y + plm[RIGHT_SHOULDER].y) / 2 * h),
            )
            mid_ear = (
                int((plm[LEFT_EAR].x + plm[RIGHT_EAR].x) / 2 * w),
                int((plm[LEFT_EAR].y + plm[RIGHT_EAR].y) / 2 * h),
            )
            vec_color = (0, 80, 255) if self._is_slouching else (0, 230, 118)
            # Thicker glow line behind
            cv2.line(img, mid_sh, mid_ear, vec_color, 6, cv2.LINE_AA)
            cv2.line(img, mid_sh, mid_ear, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, mid_sh, 7, vec_color, -1, cv2.LINE_AA)
            cv2.circle(img, mid_ear, 7, vec_color, -1, cv2.LINE_AA)

            # Draw a faint vertical reference line from shoulder
            vert_top = (mid_sh[0], mid_sh[1] - 120)
            cv2.line(img, mid_sh, vert_top, (100, 100, 100), 1, cv2.LINE_AA)

        # ── 3. HEADS-UP DISPLAY ─────────────────────────────────────
        self._draw_hud(img, w, h, ear, angle, ear_thr, angle_thr)

        # ── 4. FULLSCREEN WARNINGS ──────────────────────────────────
        if self._is_drowsy:
            # Pulsing red border
            border_alpha = 0.4 + 0.25 * math.sin(self._frame_count * 0.25)
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
            cv2.addWeighted(overlay, border_alpha, img, 1 - border_alpha, 0, img)
            _draw_text_with_bg(
                img,
                "!! DROWSY — WAKE UP !!",
                (w // 2 - 220, h // 2 - 10),
                1.1,
                (255, 255, 255),
                (0, 0, 200),
                thickness=3,
            )

        if self._is_slouching:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 120, 255), 10)
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
            y_pos = h // 2 + 50 if self._is_drowsy else h // 2 - 10
            _draw_text_with_bg(
                img,
                "!! SLOUCHING — SIT UP !!",
                (w // 2 - 230, y_pos),
                1.1,
                (255, 255, 255),
                (0, 100, 220),
                thickness=3,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # ─── HUD overlay drawing ────────────────────────────────────────
    def _draw_hud(self, img, w, h, ear, angle, ear_thr, angle_thr):
        """Draw a translucent heads-up display bar at the top of the frame."""
        # Top bar background
        _draw_rounded_rect(img, (8, 8), (w - 8, 70), (20, 20, 30), radius=10, alpha=0.70)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # EAR readout
        if ear is not None:
            ear_color = (0, 200, 255) if ear >= ear_thr else (80, 80, 255)
            cv2.putText(img, f"EAR: {ear:.3f}", (22, 50), font, 0.65, ear_color, 2, cv2.LINE_AA)
            # Threshold marker
            cv2.putText(
                img,
                f"(thr {ear_thr:.2f})",
                (170, 50),
                font,
                0.45,
                (140, 140, 160),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(img, "EAR: --", (22, 50), font, 0.65, (100, 100, 120), 2, cv2.LINE_AA)

        # Angle readout
        if angle is not None:
            ang_color = (0, 230, 118) if angle <= angle_thr else (0, 120, 255)
            cv2.putText(
                img,
                f"Neck: {angle:.1f} deg",
                (w // 2 - 70, 50),
                font,
                0.65,
                ang_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"(thr {angle_thr:.0f})",
                (w // 2 + 100, 50),
                font,
                0.45,
                (140, 140, 160),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                img, "Neck: --", (w // 2 - 70, 50), font, 0.65, (100, 100, 120), 2, cv2.LINE_AA
            )

        # Status pills (right side)
        status_x = w - 200

        if self._is_drowsy:
            cv2.putText(img, "DROWSY", (status_x, 35), font, 0.55, (80, 80, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "ALERT", (status_x, 35), font, 0.55, (0, 220, 180), 2, cv2.LINE_AA)

        if self._is_slouching:
            cv2.putText(
                img, "SLOUCH", (status_x, 58), font, 0.55, (0, 120, 255), 2, cv2.LINE_AA
            )
        else:
            cv2.putText(
                img, "UPRIGHT", (status_x, 58), font, 0.55, (0, 220, 180), 2, cv2.LINE_AA
            )

        # Mini EAR sparkline
        self._draw_sparkline(img, self._ear_history, (w - 380, 18), 100, 40, (0, 200, 255))

    # ─── Sparkline ──────────────────────────────────────────────────
    @staticmethod
    def _draw_sparkline(img, data, origin, width, height, color):
        """Draw a tiny sparkline graph from deque data."""
        if len(data) < 2:
            return
        ox, oy = origin
        vals = list(data)
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx - mn > 1e-6 else 1e-6
        step = width / (len(vals) - 1)
        pts = []
        for i, v in enumerate(vals):
            x = int(ox + i * step)
            y = int(oy + height - (v - mn) / rng * height)
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────
# STREAMLIT SIDEBAR — interactive controls
# ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Tracker Controls")
    st.markdown("---")

    st.markdown("#### 👁️ Fatigue Detection")
    ear_threshold = st.slider(
        "EAR Threshold",
        min_value=0.10,
        max_value=0.35,
        value=0.22,
        step=0.01,
        help="Eye Aspect Ratio below this value counts as 'eyes closing'. Lower = more sensitive.",
    )
    drowsy_frames = st.slider(
        "Consecutive Frames for Drowsy",
        min_value=3,
        max_value=60,
        value=15,
        step=1,
        help="Number of consecutive low-EAR frames before triggering a DROWSY alert.",
    )

    st.markdown("---")
    st.markdown("#### 🧍 Posture Detection")
    posture_threshold = st.slider(
        "Neck Lean Angle Threshold (°)",
        min_value=10.0,
        max_value=60.0,
        value=30.0,
        step=1.0,
        help="Angle of neck lean from vertical. Above this = slouching.",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="padding:0.7rem; background:rgba(0,212,255,0.06);
             border:1px solid rgba(0,212,255,0.15); border-radius:10px;
             font-size:0.78rem; color:#8892b0; line-height:1.5;">
        <strong style="color:#00d4ff;">How it works</strong><br>
        <b>EAR</b> — ratio of vertical to horizontal eye openness.
        When your eyes close, EAR drops sharply.<br><br>
        <b>Neck angle</b> — deviation of the shoulder→ear vector
        from vertical. Forward head posture increases this angle.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────
# MAIN CONTENT AREA
# ──────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="main-title">
        <h1>🧠 Real-Time Fatigue &amp; Posture Tracker</h1>
        <p>AI-powered drowsiness &amp; slouch detection — powered by MediaPipe &amp; OpenCV</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Metric cards placeholder (populated via session state from processor)
metric_placeholder = st.empty()

# ──────────────────────────────────────────────────────────────────────
# WEBRTC STREAMER
# ──────────────────────────────────────────────────────────────────────

RTC_CONFIG = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

ctx = webrtc_streamer(
    key="fatigue-posture-tracker",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=FatiguePostureProcessor,
    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    async_processing=True,
)

# Push slider values into the processor whenever the UI reruns
if ctx.video_processor is not None:
    with ctx.video_processor._lock:
        ctx.video_processor.ear_threshold = ear_threshold
        ctx.video_processor.posture_angle_threshold = posture_threshold
        ctx.video_processor.drowsy_consec_frames = drowsy_frames

    # Live metric cards (read from processor)
    cur_ear = ctx.video_processor._current_ear
    cur_ang = ctx.video_processor._current_angle
    is_drowsy = ctx.video_processor._is_drowsy
    is_slouch = ctx.video_processor._is_slouching

    ear_class = "red" if is_drowsy else "cyan"
    ang_class = "amber" if is_slouch else "green"
    drowsy_badge = (
        '<span class="status-badge status-warn">⚠ Drowsy</span>'
        if is_drowsy
        else '<span class="status-badge status-ok">✓ Alert</span>'
    )
    slouch_badge = (
        '<span class="status-badge status-warn">⚠ Slouching</span>'
        if is_slouch
        else '<span class="status-badge status-ok">✓ Upright</span>'
    )

    metric_placeholder.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Eye Aspect Ratio</div>
                <div class="metric-value {ear_class}">{cur_ear:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Neck Lean Angle</div>
                <div class="metric-value {ang_class}">{cur_ang:.1f}°</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Fatigue Status</div>
                <div>{drowsy_badge}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Posture Status</div>
                <div>{slouch_badge}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    metric_placeholder.markdown(
        """
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Eye Aspect Ratio</div>
                <div class="metric-value cyan">—</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Neck Lean Angle</div>
                <div class="metric-value green">—</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Fatigue Status</div>
                <div><span class="status-badge status-ok">Waiting…</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Posture Status</div>
                <div><span class="status-badge status-ok">Waiting…</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer-text">
        Built with Streamlit · streamlit-webrtc · MediaPipe · OpenCV &nbsp;|&nbsp;
        Hackathon 2026 🚀
    </div>
    """,
    unsafe_allow_html=True,
)
