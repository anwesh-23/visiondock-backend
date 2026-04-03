# server.py - VisionDock Flask Backend Entry Point

from flask import Flask, request, jsonify
from flask_cors import CORS

from detector import run_detection
from utils import decode_image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route("/", methods=["GET"])
def health_check():
    return "Backend is running", 200


@app.route("/detect", methods=["POST"])
def detect():
    print("✅ Request received")   # DEBUG

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use key 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        # Decode image
        frame = decode_image(file)

        print("📸 Image received and decoded")  # DEBUG

        # Run detection
        detections = run_detection(frame)

        print("🧠 Detections:", detections)  # DEBUG (VERY IMPORTANT)

        return jsonify({"detections": detections}), 200

    except Exception as e:
        print("❌ ERROR:", str(e))  # DEBUG
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("🚀 Starting VisionDock backend...")
    app.run(host="0.0.0.0", port=5000, debug=True)