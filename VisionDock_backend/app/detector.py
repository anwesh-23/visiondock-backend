# detector.py - YOLO Model Loading and Object Detection Logic

import os
import numpy as np
from ultralytics import YOLO

# -----------------------------------------------------------------------
# Define paths
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best.pt")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "yolov8n.pt")

# -----------------------------------------------------------------------
# Load model safely
# -----------------------------------------------------------------------
if os.path.exists(CUSTOM_MODEL_PATH):
    print(f"✅ Loading custom model from: {CUSTOM_MODEL_PATH}")
    model = YOLO(CUSTOM_MODEL_PATH)

elif os.path.exists(DEFAULT_MODEL_PATH):
    print("⚠️ Custom model not found. Using local yolov8n.pt...")
    model = YOLO(DEFAULT_MODEL_PATH)

else:
    print("⚠️ No local model found. Downloading yolov8n from internet...")
    model = YOLO("yolov8n.pt")  # fallback (auto-download)

print("✅ Model loaded successfully.")


def run_detection(frame: np.ndarray) -> list[str]:
    """
    Run YOLO object detection on a single image frame.
    """

    if frame is None or not isinstance(frame, np.ndarray):
        raise ValueError("Invalid frame passed to run_detection.")

    # 🔥 IMPORTANT: lower confidence for better detection
    results = model(frame, conf=0.1)

    detected_labels = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_index = int(box.cls[0])
            label = model.names[class_index]
            detected_labels.append(label)

    # Remove duplicates
    unique_labels = list(set(detected_labels))

    print("🧠 Detected:", unique_labels)  # DEBUG

    return unique_labels