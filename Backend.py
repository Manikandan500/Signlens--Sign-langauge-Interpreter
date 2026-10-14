import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
import os
import warnings
import json

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths to your trained model files
CNN_MODEL_PATH  = "alphabet_model.h5"      # Keras CNN for static signs
LSTM_MODEL_PATH = "lstm_power30_final.h5"    # Keras LSTM for dynamic signs
LSTM_LABEL_MAP_PATH  = "sign_map_power30.json"   # Dict: {"0": "label"}

# CNN Label Map: Assumes standard 26-class alphabet A-Z (0=A, 1=B, ..., 25=Z)
# Modify this dictionary if your CNN includes 'Space', 'Del', or 'Nothing'
CNN_LABELS = {i: chr(65 + i) for i in range(26)}

# Motion Gate tuning
VELOCITY_THRESHOLD   = 0.015   # Normalised landmark velocity; above = dynamic sign
MOTION_WINDOW        = 10      # Number of frames to evaluate motion over
DYNAMIC_FRAME_COUNT  = 30      # Frames to collect for LSTM sequence
CNN_INPUT_SIZE       = (64, 64)  # Resize target for CNN static crops
LSTM_FEATURE_DIM     = 63       # 21 landmarks × 3 (x, y, z) per hand (one hand)

# =============================================================================
# MEDIAPIPE SETUP
# =============================================================================

mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils

# =============================================================================
# MODEL LOADER
# =============================================================================

_cnn_model   = None
_lstm_model  = None
_lstm_label_map   = None

def _load_models():
    """Lazily load CNN, LSTM, and LSTM label map from disk."""
    global _cnn_model, _lstm_model, _lstm_label_map

    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        raise ImportError(
            "TensorFlow is required. Install with: pip install tensorflow"
        )

    if _cnn_model is None:
        if not os.path.exists(CNN_MODEL_PATH):
            raise FileNotFoundError(f"CNN model not found at: {CNN_MODEL_PATH}")
        _cnn_model = load_model(CNN_MODEL_PATH)

    if _lstm_model is None:
        if not os.path.exists(LSTM_MODEL_PATH):
            raise FileNotFoundError(f"LSTM model not found at: {LSTM_MODEL_PATH}")
        _lstm_model = load_model(LSTM_MODEL_PATH)

    if _lstm_label_map is None:
        if not os.path.exists(LSTM_LABEL_MAP_PATH):
            raise FileNotFoundError(f"Label map not found at: {LSTM_LABEL_MAP_PATH}")
        with open(LSTM_LABEL_MAP_PATH, "r") as f:
            # json.load reads string keys; convert keys to integers for array indexing
            raw_map = json.load(f)
            _lstm_label_map = {int(k): v for k, v in raw_map.items()}

    return _cnn_model, _lstm_model, _lstm_label_map


# =============================================================================
# HELPER — LANDMARK EXTRACTION
# =============================================================================

def _extract_landmarks(hand_landmarks) -> np.ndarray:
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)

def _landmark_velocity(prev: np.ndarray, curr: np.ndarray) -> float:
    if prev is None or curr is None:
        return 0.0
    p = prev.reshape(21, 3)[:, :2]
    c = curr.reshape(21, 3)[:, :2]
    return float(np.mean(np.linalg.norm(c - p, axis=1)))


# =============================================================================
# MOTION GATE
# =============================================================================

class MotionGate:
    def __init__(self, threshold: float = VELOCITY_THRESHOLD, window: int = MOTION_WINDOW):
        self.threshold  = threshold
        self.window     = window
        self._velocities: list[float] = []

    def update(self, velocity: float) -> None:
        self._velocities.append(velocity)
        if len(self._velocities) > self.window:
            self._velocities.pop(0)

    @property
    def mean_velocity(self) -> float:
        if not self._velocities:
            return 0.0
        return float(np.mean(self._velocities))

    @property
    def is_dynamic(self) -> bool:
        return self.mean_velocity > self.threshold

    def reset(self) -> None:
        self._velocities.clear()


# =============================================================================
# CNN INFERENCE  (static sign)
# =============================================================================

def _predict_static(frame_bgr: np.ndarray,
                    hand_landmarks,
                    frame_h: int,
                    frame_w: int) -> str:
    cnn, _, _ = _load_models()

    xs = [lm.x * frame_w for lm in hand_landmarks.landmark]
    ys = [lm.y * frame_h for lm in hand_landmarks.landmark]
    pad = 20
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(frame_w, int(max(xs)) + pad)
    y2 = min(frame_h, int(max(ys)) + pad)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return "Unknown"

    crop_rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized   = cv2.resize(crop_rgb, CNN_INPUT_SIZE)
    normed    = resized.astype(np.float32) / 255.0
    input_arr = np.expand_dims(normed, axis=0)

    preds = cnn.predict(input_arr, verbose=0)
    idx   = int(np.argmax(preds[0]))
    
    # Use the dedicated CNN label map
    return CNN_LABELS.get(idx, f"Class_{idx}")


# =============================================================================
# LSTM INFERENCE  (dynamic sign)
# =============================================================================

def _predict_dynamic(sequence: list[np.ndarray]) -> str:
    _, lstm, lstm_label_map = _load_models()

    arr   = np.stack(sequence, axis=0)
    arr   = np.expand_dims(arr, axis=0)
    preds = lstm.predict(arr, verbose=0)
    idx   = int(np.argmax(preds[0]))
    return lstm_label_map.get(idx, f"Class_{idx}")


# =============================================================================
# CORE — ANALYSE VIDEO
# =============================================================================

def analyze_video_sign(video_path: str) -> str:
    if not os.path.exists(video_path):
        return "Error: Video file not found."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video."

    predictions: list[str] = []
    landmark_sequence: list[np.ndarray] = []
    gate = MotionGate()
    prev_landmarks = None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res  = hands.process(rgb)

            if not res.multi_hand_landmarks:
                gate.reset()
                prev_landmarks = None
                continue

            hand_lm  = res.multi_hand_landmarks[0]
            curr_lm  = _extract_landmarks(hand_lm)

            velocity = _landmark_velocity(prev_landmarks, curr_lm)
            gate.update(velocity)
            prev_landmarks = curr_lm

            if gate.is_dynamic:
                landmark_sequence.append(curr_lm)
                if len(landmark_sequence) >= DYNAMIC_FRAME_COUNT:
                    label = _predict_dynamic(landmark_sequence[-DYNAMIC_FRAME_COUNT:])
                    predictions.append(label)
                    landmark_sequence = landmark_sequence[DYNAMIC_FRAME_COUNT // 2:]
            else:
                label = _predict_static(frame, hand_lm, h, w)
                predictions.append(label)
                landmark_sequence.clear()

    cap.release()

    if not predictions:
        return "No sign detected."

    from collections import Counter
    most_common, _ = Counter(predictions).most_common(1)[0]
    return most_common


# =============================================================================
# VOICE (TTS)
# =============================================================================

def speak_text(text: str) -> None:
    """Reads text aloud in a separate daemon thread."""
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()