# -*- coding: utf-8 -*-
"""
Sign Language Recognition - Data Preprocessing Pipeline (Holistic-Compatible)
==============================================================================

Compatible with MediaPipe 0.10.x (Tasks API).

Uses MediaPipe Pose + Hands to extract:
    - Pose:       33 landmarks × 3 = 99
    - Left Hand:  21 landmarks × 3 = 63
    - Right Hand: 21 landmarks × 3 = 63
    ─────────────────────────────────────
    Total per frame:        75 × 3 = 225 values
    (Face landmarks removed — not in Tasks API, Face Mesh is separate)

NOTE: If you need face landmarks, install mediapipe-model-maker separately.

Output shape per video : (30, 225)         [flattened]
Final stacked dataset  : (samples, 30, 225)

INSTALLATION:
    pip install opencv-python mediapipe numpy tqdm

DATASET STRUCTURE EXPECTED:
    downloads/
        hello/
            video1.mp4
            video2.mp4
        thanks/
            video1.mp4
        ...

USAGE:
    python p_s_l.py
"""

import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATASET_DIR    = "downloads"    # Root folder with gesture subfolders
OUTPUT_DIR     = "processed"    # Where .npy files will be saved
LABELS_FILE    = "labels.json"  # Gesture-to-integer mapping
FRAME_INTERVAL = 5              # Extract every Nth frame
SEQUENCE_LEN   = 30            # Fixed frames per video

# MediaPipe landmark counts
POSE_LM        = 33
HAND_LM        = 21
FACE_LM        = 0    # Face not available in standard Tasks API; set 0
TOTAL_LM       = POSE_LM + HAND_LM + HAND_LM + FACE_LM
COORDS_PER_LM  = 3
FEATURE_SIZE   = TOTAL_LM * COORDS_PER_LM   # 225

VIDEO_EXTS     = {".mp4", ".avi", ".mov", ".mkv"}

# ── Model download URLs ──────────────────────
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"


# ─────────────────────────────────────────────
# MODEL DOWNLOADER
# ─────────────────────────────────────────────

def download_model(url: str, dest: str) -> None:
    """Download a MediaPipe .task model file if not already present."""
    if os.path.exists(dest):
        print(f"  [OK] Model already present: {dest}")
        return
    import urllib.request
    print(f"  [DL] Downloading model: {dest} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  [OK] Downloaded: {dest}")


# ─────────────────────────────────────────────
# DETECTOR BUILDERS
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# VIDEO → FRAMES
# ─────────────────────────────────────────────

def extract_frames(video_path: str, interval: int) -> list:
    """Read a video and return every `interval`-th frame as a BGR ndarray."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames


# ─────────────────────────────────────────────
# LANDMARK EXTRACTION & FLATTENING
# ─────────────────────────────────────────────

def _lm_list_to_array(landmark_list, n: int) -> np.ndarray:
    """Convert a list of NormalizedLandmark objects to flat (n*3,) array."""
    if not landmark_list:
        return np.zeros(n * COORDS_PER_LM, dtype=np.float32)
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmark_list],
        dtype=np.float32,
    ).flatten()


def extract_features(
    frame: np.ndarray,
    pose_det: mp_vision.PoseLandmarker,
    hand_det: mp_vision.HandLandmarker,
) -> np.ndarray:
    """
    Run Pose + Hand detectors on a single BGR frame.

    Feature layout:
        pose (99) | left_hand (63) | right_hand (63) = 225
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # ── Pose ────────────────────────────────
    pose_result = pose_det.detect(mp_image)
    pose_lms = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else []
    pose_vec = _lm_list_to_array(pose_lms, POSE_LM)

    # ── Hands ───────────────────────────────
    hand_result = hand_det.detect(mp_image)
    left_vec  = np.zeros(HAND_LM * COORDS_PER_LM, dtype=np.float32)
    right_vec = np.zeros(HAND_LM * COORDS_PER_LM, dtype=np.float32)

    for i, handedness_list in enumerate(hand_result.handedness):
        category = handedness_list[0].category_name  # "Left" or "Right"
        lms = hand_result.hand_landmarks[i]
        arr = _lm_list_to_array(lms, HAND_LM)
        if category == "Left":
            left_vec = arr
        else:
            right_vec = arr

    return np.concatenate([pose_vec, left_vec, right_vec])  # (225,)


# ─────────────────────────────────────────────
# NORMALISATION (wrist-relative for hands)
# ─────────────────────────────────────────────

def normalize_hand_section(feature_vec: np.ndarray, hand_start: int) -> np.ndarray:
    """Translate hand landmarks so the wrist is at the origin."""
    vec = feature_vec.copy()
    wrist_xyz = vec[hand_start: hand_start + COORDS_PER_LM].copy()
    hand_end  = hand_start + HAND_LM * COORDS_PER_LM
    vec[hand_start:hand_end] -= np.tile(wrist_xyz, HAND_LM)
    return vec


def normalize_features(fv: np.ndarray) -> np.ndarray:
    LEFT_HAND_START  = POSE_LM * COORDS_PER_LM           # 99
    RIGHT_HAND_START = LEFT_HAND_START + HAND_LM * COORDS_PER_LM  # 162
    fv = normalize_hand_section(fv, LEFT_HAND_START)
    fv = normalize_hand_section(fv, RIGHT_HAND_START)
    return fv


# ─────────────────────────────────────────────
# SEQUENCE BUILDER
# ─────────────────────────────────────────────

def build_fixed_sequence(feature_list: list, sequence_len: int) -> np.ndarray:
    """Truncate or zero-pad feature list to `sequence_len` frames."""
    sequence = np.zeros((sequence_len, FEATURE_SIZE), dtype=np.float32)
    n = min(len(feature_list), sequence_len)
    for i in range(n):
        sequence[i] = feature_list[i]
    return sequence  # (30, 225)


# ─────────────────────────────────────────────
# PER-VIDEO PROCESSING
# ─────────────────────────────────────────────

def process_video(
    video_path: str,
    pose_det,
    hand_det,
    frame_interval: int,
    sequence_len: int,
) -> np.ndarray | None:
    frames = extract_frames(video_path, frame_interval)
    feature_list = []
    for frame in frames:
        fv = extract_features(frame, pose_det, hand_det)
        fv = normalize_features(fv)
        feature_list.append(fv)
    if not feature_list:
        return None
    return build_fixed_sequence(feature_list, sequence_len)


# ─────────────────────────────────────────────
# DATASET DISCOVERY
# ─────────────────────────────────────────────

def discover_dataset(dataset_dir: str) -> dict:
    gesture_map = {}
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    for label in sorted(os.listdir(dataset_dir)):
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_dir):
            continue
        videos = [
            os.path.join(label_dir, f)
            for f in os.listdir(label_dir)
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS
        ]
        if videos:
            gesture_map[label] = sorted(videos)
    return gesture_map


# ─────────────────────────────────────────────
# LABEL MAP
# ─────────────────────────────────────────────

def build_label_map(gesture_map: dict) -> dict:
    return {label: idx for idx, label in enumerate(sorted(gesture_map))}


def save_label_map(label_map: dict, output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\n[OK] Labels saved -> {output_path}")
    for gesture, idx in label_map.items():
        print(f"  {idx:>3} : {gesture}")


# ─────────────────────────────────────────────
# STACK ALL SAMPLES
# ─────────────────────────────────────────────

def stack_and_save_dataset(output_dir: str, label_map: dict, sequence_len: int) -> None:
    X_list, y_list = [], []
    for label, idx in sorted(label_map.items(), key=lambda x: x[1]):
        label_dir = os.path.join(output_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(".npy"):
                continue
            arr = np.load(os.path.join(label_dir, fname))
            if arr.shape != (sequence_len, FEATURE_SIZE):
                print(f"  [WARN] Unexpected shape {arr.shape} -- skipping {fname}")
                continue
            X_list.append(arr)
            y_list.append(idx)

    if not X_list:
        print("[WARN] No valid .npy samples found to stack.")
        return

    X = np.stack(X_list)
    y = np.array(y_list)

    X_path = os.path.join(output_dir, "X.npy")
    y_path = os.path.join(output_dir, "y.npy")
    np.save(X_path, X)
    np.save(y_path, y)

    print(f"\n[OK] Dataset stacked:")
    print(f"    X.npy -> {X.shape}   ({X_path})")
    print(f"    y.npy -> {y.shape}   ({y_path})")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(
    dataset_dir: str    = DATASET_DIR,
    output_dir: str     = OUTPUT_DIR,
    labels_file: str    = LABELS_FILE,
    frame_interval: int = FRAME_INTERVAL,
    sequence_len: int   = SEQUENCE_LEN,
) -> None:
    print("=" * 60)
    print("  Sign Language Preprocessing Pipeline  [Tasks API]")
    print("=" * 60)
    print(f"\n  Feature breakdown per frame:")
    print(f"    Pose       : {POSE_LM} lm x 3 = {POSE_LM * 3:>5}")
    print(f"    Left hand  : {HAND_LM} lm x 3 = {HAND_LM * 3:>5}")
    print(f"    Right hand : {HAND_LM} lm x 3 = {HAND_LM * 3:>5}")
    print(f"    ------------------------------")
    print(f"    Total      : {TOTAL_LM} lm x 3 = {FEATURE_SIZE:>5}")

    # ── Download Models ──────────────────────
    print("\n[INFO] Checking / downloading models...")
    download_model(POSE_MODEL_URL, POSE_MODEL_PATH)
    download_model(HAND_MODEL_URL, HAND_MODEL_PATH)

    # ── Discover ──────────────────────────────
    gesture_map = discover_dataset(dataset_dir)
    if not gesture_map:
        print("[WARN] No videos found. Check DATASET_DIR and VIDEO_EXTS.")
        return

    total_videos = sum(len(v) for v in gesture_map.values())
    print(f"\n  Dataset   : {dataset_dir}")
    print(f"  Gestures  : {len(gesture_map)}  ({', '.join(gesture_map)})")
    print(f"  Videos    : {total_videos}")
    print(f"  Interval  : every {frame_interval} frames")
    print(f"  Seq. len  : {sequence_len} frames\n")

    # ── Labels ────────────────────────────────
    label_map = build_label_map(gesture_map)
    save_label_map(label_map, labels_file)

    # ── Output dirs ───────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    for label in gesture_map:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    # ── Process ───────────────────────────────
    skipped, saved = 0, 0

    pose_det = build_pose_detector()
    hand_det = build_hand_detector()

    try:
        for label, video_paths in gesture_map.items():
            print(f"\n[{label_map[label]}] {label}  ({len(video_paths)} videos)")
            for video_path in tqdm(video_paths, desc=f"  {label}", unit="video"):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                out_path   = os.path.join(output_dir, label, f"{video_name}.npy")
                try:
                    sequence = process_video(
                        video_path, pose_det, hand_det, frame_interval, sequence_len
                    )
                except IOError as exc:
                    tqdm.write(f"    [SKIP] cannot open: {exc}")
                    skipped += 1
                    continue
                if sequence is None:
                    tqdm.write(f"    [SKIP] no frames: {video_path}")
                    skipped += 1
                    continue
                np.save(out_path, sequence)
                saved += 1
    finally:
        pose_det.close()
        hand_det.close()

    # ── Stack ─────────────────────────────────
    stack_and_save_dataset(output_dir, label_map, sequence_len)

    # ── Summary ───────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Saved   : {saved} files  ->  {output_dir}/")
    print(f"  Skipped : {skipped} videos")
    print(f"  Shape   : ({sequence_len}, {FEATURE_SIZE})  per sample")
    print(f"  Final   : X.npy -> (samples, {sequence_len}, {FEATURE_SIZE})")
    print("=" * 60)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline(
        dataset_dir    = DATASET_DIR,
        output_dir     = OUTPUT_DIR,
        labels_file    = LABELS_FILE,
        frame_interval = FRAME_INTERVAL,
        sequence_len   = SEQUENCE_LEN,
    )
