# -*- coding: utf-8 -*-
"""
Sign Language Recognition - Webcam Real-Time Landmark Extraction
================================================================

Compatible with MediaPipe 0.10.x (Tasks API).

Connects to your webcam and visualises:
    - 33 Pose landmarks   (body skeleton)
    - 21 Left-hand landmarks
    - 21 Right-hand landmarks

Feature vector per frame: 225 values  (same layout as p_s_l.py)
    pose (99) | left_hand (63) | right_hand (63)

CONTROLS (keyboard):
    Q / ESC  – quit
    S        – start / stop recording a gesture sequence
    C        – clear current buffer

USAGE:
    python webcam_landmark.py

The script shares the same model paths and helper functions as p_s_l.py,
so make sure both files live in the same directory and the .task model
files have been downloaded (they auto-download on first run).
"""

import os
import time

# ── Prevent matplotlib frozen-dataclass crash on Python 3.10 ─────────────────
# mediapipe's drawing_utils imports matplotlib at load time.
# Setting the backend to the headless 'Agg' renderer BEFORE any mediapipe
# import avoids the GUI initialisation path that triggers the crash.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")          # non-interactive, no GUI window needed
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONFIGURATION  (mirrors p_s_l.py — keep in sync)
# ─────────────────────────────────────────────────────────────────────────────

POSE_LM        = 33
HAND_LM        = 21
COORDS_PER_LM  = 3
FEATURE_SIZE   = (POSE_LM + HAND_LM + HAND_LM) * COORDS_PER_LM   # 225
SEQUENCE_LEN   = 30   # frames to collect per gesture

POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"

# ── Drawing colours (BGR) ────────────────────
COLOR_POSE       = (0, 200, 255)    # orange-yellow
COLOR_LEFT_HAND  = (50, 255, 50)    # green
COLOR_RIGHT_HAND = (255, 80, 80)    # blue
COLOR_RECORDING  = (0, 0, 255)      # red indicator
COLOR_READY      = (0, 255, 150)    # teal ready indicator
COLOR_TEXT       = (255, 255, 255)  # white
COLOR_BORDER     = (60, 60, 60)     # dark shadow

# ── Pose connections (MediaPipe's body skeleton) ──────────────────────────────
# fmt: off
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]
# Hand connections (same topology for both hands)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),      # thumb
    (0,5),(5,6),(6,7),(7,8),      # index
    (0,9),(9,10),(10,11),(11,12), # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17),          # palm knuckles
]
# fmt: on


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def download_model(url: str, dest: str) -> None:
    """Auto-download a .task model file if missing."""
    if os.path.exists(dest):
        return
    import urllib.request
    print(f"[DL] Downloading {dest} …")
    urllib.request.urlretrieve(url, dest)
    print(f"[OK] {dest} ready.")


def build_pose_detector() -> mp_vision.PoseLandmarker:
    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def build_hand_detector() -> mp_vision.HandLandmarker:
    base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (same math as p_s_l.py)
# ─────────────────────────────────────────────────────────────────────────────

def _lm_list_to_array(landmark_list, n: int) -> np.ndarray:
    if not landmark_list:
        return np.zeros(n * COORDS_PER_LM, dtype=np.float32)
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmark_list],
        dtype=np.float32,
    ).flatten()


def extract_features(frame, pose_det, hand_det):
    """
    Returns:
        feature_vec  – (225,) float32 numpy array
        pose_result  – raw MediaPipe PoseLandmarkerResult
        hand_result  – raw MediaPipe HandLandmarkerResult
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pose_result = pose_det.detect(mp_image)
    hand_result = hand_det.detect(mp_image)

    pose_lms = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else []
    pose_vec  = _lm_list_to_array(pose_lms, POSE_LM)

    left_vec  = np.zeros(HAND_LM * COORDS_PER_LM, dtype=np.float32)
    right_vec = np.zeros(HAND_LM * COORDS_PER_LM, dtype=np.float32)

    for i, handedness_list in enumerate(hand_result.handedness):
        category = handedness_list[0].category_name
        lms = hand_result.hand_landmarks[i]
        arr = _lm_list_to_array(lms, HAND_LM)
        if category == "Left":
            left_vec = arr
        else:
            right_vec = arr

    feature_vec = np.concatenate([pose_vec, left_vec, right_vec])
    return feature_vec, pose_result, hand_result


def normalize_hand_section(fv: np.ndarray, hand_start: int) -> np.ndarray:
    vec = fv.copy()
    wrist = vec[hand_start: hand_start + COORDS_PER_LM].copy()
    hand_end = hand_start + HAND_LM * COORDS_PER_LM
    vec[hand_start:hand_end] -= np.tile(wrist, HAND_LM)
    return vec


def normalize_features(fv: np.ndarray) -> np.ndarray:
    LEFT_START  = POSE_LM * COORDS_PER_LM
    RIGHT_START = LEFT_START + HAND_LM * COORDS_PER_LM
    fv = normalize_hand_section(fv, LEFT_START)
    fv = normalize_hand_section(fv, RIGHT_START)
    return fv


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _px(lm, w, h):
    """Normalised landmark → pixel coords."""
    return int(lm.x * w), int(lm.y * h)


def draw_pose(frame, pose_result, color=COLOR_POSE):
    h, w = frame.shape[:2]
    if not pose_result.pose_landmarks:
        return
    lms = pose_result.pose_landmarks[0]
    # Connections
    for a, b in POSE_CONNECTIONS:
        if a < len(lms) and b < len(lms):
            cv2.line(frame, _px(lms[a], w, h), _px(lms[b], w, h), color, 2, cv2.LINE_AA)
    # Joints
    for lm in lms:
        cv2.circle(frame, _px(lm, w, h), 4, color, -1, cv2.LINE_AA)


def draw_hand(frame, hand_result, color_left=COLOR_LEFT_HAND, color_right=COLOR_RIGHT_HAND):
    h, w = frame.shape[:2]
    for i, handedness_list in enumerate(hand_result.handedness):
        category = handedness_list[0].category_name
        color    = color_left if category == "Left" else color_right
        lms      = hand_result.hand_landmarks[i]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, _px(lms[a], w, h), _px(lms[b], w, h), color, 2, cv2.LINE_AA)
        for lm in lms:
            cv2.circle(frame, _px(lm, w, h), 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, _px(lm, w, h), 5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, recording: bool, buffer_len: int, fps: float, feature_vec: np.ndarray):
    """Heads-up display: status, buffer count, FPS, mini bar chart."""
    h, w = frame.shape[:2]

    # ── Top-left panel ───────────────────────────────────────────────────────
    panel_h, panel_w = 110, 340
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    status_text  = "● REC" if recording else "○ IDLE"
    status_color = COLOR_RECORDING if recording else COLOR_READY
    cv2.putText(frame, status_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"Frames: {buffer_len:>3} / {SEQUENCE_LEN}",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS:    {fps:>5.1f}",
                (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Vec:    {FEATURE_SIZE} features",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_TEXT, 1, cv2.LINE_AA)

    # ── Buffer progress bar ───────────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 10, 115, 320, 10
    fill = int(bar_w * buffer_len / SEQUENCE_LEN)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), status_color, -1)

    # ── Bottom controls legend ────────────────────────────────────────────────
    legend = "[S] Record  [C] Clear  [Q/ESC] Quit"
    cv2.putText(frame, legend, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Mini feature heatbar (right side) ────────────────────────────────────
    bar_x2 = w - 22
    seg_h   = max(1, (h - 40) // len(feature_vec))
    norm    = feature_vec / (np.max(np.abs(feature_vec)) + 1e-6)
    for j, v in enumerate(norm):
        intensity = int(abs(v) * 255)
        color_h   = (0, intensity, 255 - intensity)  # blue→red gradient
        y1 = 20 + j * seg_h
        y2 = y1 + seg_h
        cv2.rectangle(frame, (bar_x2, y1), (bar_x2 + 14, y2), color_h, -1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WEBCAM LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_webcam(camera_index: int = 0) -> None:
    """
    Open the webcam, detect landmarks on every frame, and visualise results.

    Press S to start/stop buffering a gesture sequence (saves to sequence.npy).
    Press C to clear the current buffer.
    Press Q or ESC to quit.
    """
    print("=" * 60)
    print("  Sign Language — Webcam Landmark Extraction  [Tasks API]")
    print("=" * 60)

    # ── Download models if needed ─────────────────────────────────────────────
    download_model(POSE_MODEL_URL, POSE_MODEL_PATH)
    download_model(HAND_MODEL_URL, HAND_MODEL_PATH)

    # ── Build detectors ───────────────────────────────────────────────────────
    print("[INFO] Loading pose detector …")
    pose_det = build_pose_detector()
    print("[INFO] Loading hand detector …")
    hand_det = build_hand_detector()
    print("[INFO] Detectors ready.\n")

    # ── Open camera (DirectShow first — avoids MSMF grab bugs on Windows) ─────
    cap = None
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF,  "MSMF"),
        (cv2.CAP_ANY,   "Auto"),
    ]
    indices_to_try = [camera_index] if camera_index != 0 else [0, 1, 2]

    for idx in indices_to_try:
        for backend, name in backends:
            _cap = cv2.VideoCapture(idx, backend)
            if _cap.isOpened():
                ok, _test = _cap.read()   # verify we can actually grab a frame
                if ok:
                    cap = _cap
                    print(f"[OK] Camera #{idx} opened via {name} backend.")
                    break
                _cap.release()
        if cap is not None:
            break

    if cap is None:
        print("[ERROR] Could not open any camera. Check connections / permissions.")
        pose_det.close(); hand_det.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording         = False
    sequence_buf      = []          # list of (225,) feature vectors
    feature_vec       = np.zeros(FEATURE_SIZE, dtype=np.float32)
    fps               = 0.0
    prev_time         = time.perf_counter()
    saved_count       = 0
    consecutive_fails = 0
    MAX_FAILS         = 30         # exit after 30 consecutive bad frames

    print("Controls: [S] Start/Stop recording  [C] Clear  [Q/ESC] Quit")
    print("-" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_fails += 1
                print(f"[WARN] Frame grab failed ({consecutive_fails}/{MAX_FAILS}) …")
                cv2.waitKey(50)
                if consecutive_fails >= MAX_FAILS:
                    print("[ERROR] Too many consecutive frame failures. Exiting.")
                    break
                continue
            consecutive_fails = 0   # reset on success

            # Mirror for natural feel
            frame = cv2.flip(frame, 1)

            # ── Feature extraction ────────────────────────────────────────────
            try:
                feature_vec, pose_result, hand_result = extract_features(
                    frame, pose_det, hand_det
                )
                fv_norm = normalize_features(feature_vec)
            except Exception as exc:
                print(f"[WARN] Extraction error: {exc}")
                pose_result = type("R", (), {"pose_landmarks": []})()
                hand_result = type("R", (), {"handedness": [], "hand_landmarks": []})()
                fv_norm = feature_vec

            # ── Buffer gesture frames ─────────────────────────────────────────
            if recording:
                sequence_buf.append(fv_norm.copy())
                if len(sequence_buf) >= SEQUENCE_LEN:
                    recording = False
                    seq_array = np.array(sequence_buf[:SEQUENCE_LEN], dtype=np.float32)
                    out_file  = f"sequence_{saved_count:03d}.npy"
                    np.save(out_file, seq_array)
                    saved_count += 1
                    print(f"[OK] Saved {out_file}  shape={seq_array.shape}")
                    sequence_buf = []

            # ── Draw skeleton overlays ────────────────────────────────────────
            draw_pose(frame, pose_result)
            draw_hand(frame, hand_result)

            # ── FPS ───────────────────────────────────────────────────────────
            now       = time.perf_counter()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # ── HUD overlay ───────────────────────────────────────────────────
            draw_hud(frame, recording, len(sequence_buf), fps, feature_vec)

            cv2.imshow("Sign Language — Landmark Extraction  (Q to quit)", frame)

            # ── Keyboard handling ─────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # Q or ESC
                break
            elif key in (ord("s"), ord("S")):
                if recording:
                    print("[INFO] Recording stopped (manual).")
                    recording = False
                    sequence_buf = []
                else:
                    print("[INFO] Recording started — perform your gesture!")
                    recording    = True
                    sequence_buf = []
            elif key in (ord("c"), ord("C")):
                sequence_buf = []
                recording    = False
                print("[INFO] Buffer cleared.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_det.close()
        hand_det.close()
        print(f"\n[OK] Done.  Sequences saved: {saved_count}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Webcam landmark extraction for sign language")
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)"
    )
    args = parser.parse_args()
    run_webcam(camera_index=args.camera)
