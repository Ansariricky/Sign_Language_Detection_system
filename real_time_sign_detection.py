import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from keras.models import load_model
import os

# ==============================
# Load best trained model
# ==============================
MODEL_PATH = "best_model.h5" if os.path.exists("best_model.h5") else "sign_language_model.h5"
print(f"[INFO] Loading model: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# ==============================
# Define the class labels (update if your dataset is a subset)
# ==============================
class_labels = [chr(i) for i in range(65, 91)]  # A-Z

# ==============================
# Mediapipe Hands setup (now supports TWO hands)
# ==============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Parameters
SEQ_LENGTH = 30
threshold = 0.75

# Keep a separate rolling sequence per hand
sequences = {
    "Left": deque(maxlen=SEQ_LENGTH),
    "Right": deque(maxlen=SEQ_LENGTH)
}

# Colors for overlays
colors = {
    "Left": (255, 0, 0),   # Blue-ish
    "Right": (0, 255, 0)   # Green
}

# ==============================
# Helper: Normalize landmarks relative to wrist (index 0)
# ==============================
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(21, 3)
    wrist = landmarks[0]
    normalized = landmarks - wrist
    return normalized.flatten()

# Draw a labeled box above a given bounding box
def draw_label_box(frame, text, x_min, y_min, color):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    box_left = max(x_min - 4, 0)
    box_top = max(y_min - th - 10, 0)
    box_right = min(box_left + tw + 8, frame.shape[1] - 1)
    box_bottom = min(box_top + th + 8, frame.shape[0] - 1)
    cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), color, -1)
    cv2.putText(frame, text, (box_left + 4, box_bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

# ==============================
# Open Webcam
# ==============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    raise SystemExit

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Zip landmarks with handedness so we know which is Left/Right
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'
            color = colors.get(hand_label, (0, 255, 255))

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collect pixel coords to make a bounding box
            xs, ys = [], []
            raw = []
            for lm in hand_landmarks.landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))
                raw.extend([lm.x, lm.y, lm.z])

            x_min, x_max = max(min(xs), 0), min(max(xs), w - 1)
            y_min, y_max = max(min(ys), 0), min(max(ys), h - 1)

            # Pad the box a bit
            pad = 20
            x_min = max(x_min - pad, 0)
            y_min = max(y_min - pad, 0)
            x_max = min(x_max + pad, w - 1)
            y_max = min(y_max + pad, h - 1)

            # Draw the bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

            # Normalize and append to that hand's sequence
            norm = normalize_landmarks(raw)
            sequences[hand_label].append(norm)

            # Predict when we have a full sequence for this hand
            if len(sequences[hand_label]) == SEQ_LENGTH:
                inp = np.expand_dims(np.array(sequences[hand_label]), axis=0)  # (1, 30, 63)
                probs = model.predict(inp, verbose=0)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])

                if conf > threshold:
                    text = f"{hand_label}: {class_labels[idx]} ({conf:.2f})"
                else:
                    text = f"{hand_label}: Uncertain"

                # Draw a filled label box above the hand
                draw_label_box(frame, text, x_min, y_min, color)
    else:
        cv2.putText(frame, "No Hand Detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Real-Time Sign Detection (Both Hands)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
