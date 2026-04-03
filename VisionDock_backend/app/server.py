# server.py - VisionDock Flask Backend Entry Point

from flask import Flask, request, jsonify
from flask_cors import CORS

from detector import run_detection
from utils import decode_image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (allows frontend to connect)


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return "Backend is running", 200


@app.route("/detect", methods=["POST"])
def detect():
    """
    Accepts an image file from the frontend, runs YOLO object detection,
    and returns detected object labels as JSON.

    Expected request: multipart/form-data with key 'image'
    Returns: { "detections": ["person", "car", ...] }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use key 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename. Please upload a valid image."}), 400

    try:
        # Convert uploaded file to a NumPy frame OpenCV can work with
        frame = decode_image(file)

        # Run YOLO detection on the frame
        detections = run_detection(frame)

        return jsonify({"detections": detections}), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422

    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("Starting VisionDock backend...")
    app.run(host="0.0.0.0", port=5000, debug=True)
