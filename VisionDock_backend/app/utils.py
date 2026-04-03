# utils.py - Image Processing Utilities for VisionDock

import cv2
import numpy as np
from werkzeug.datastructures import FileStorage


def decode_image(file: FileStorage) -> np.ndarray:
    """
    Convert an uploaded Flask file object into an OpenCV-compatible NumPy array.

    Args:
        file (FileStorage): The image file from request.files.

    Returns:
        np.ndarray: Decoded BGR image array ready for YOLO inference.

    Raises:
        ValueError: If the image cannot be decoded.
    """
    # Read raw bytes from the uploaded file
    file_bytes = file.read()

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    # Decode bytes into a NumPy array
    np_array = np.frombuffer(file_bytes, dtype=np.uint8)

    # Decode the image using OpenCV (returns BGR format)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError(
            "Could not decode the image. "
            "Ensure the file is a valid image (JPEG, PNG, etc.)."
        )

    return frame
