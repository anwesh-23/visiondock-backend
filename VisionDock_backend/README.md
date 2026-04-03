# VisionDock — AI Navigation Assistant Backend

A Flask-based backend that accepts images and returns detected obstacle labels
using a trained YOLO model.

---

## Project Structure

```
VisionDock_backend/
│
├── app/
│   ├── server.py      ← Flask app & API endpoints
│   ├── detector.py    ← YOLO model loading & inference
│   └── utils.py       ← Image decoding utility
│
├── models/
│   └── yolov8n.pt        ← ⚠️ YOU MUST ADD THIS (not included)
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your YOLO model
Place your trained model file at:
```
models/yolov8n.pt
```

### 3. Run the server
```bash
python app/server.py
```

The server starts at: `http://localhost:5000`

---

## API Endpoints

### `GET /`
Health check.

**Response:**
```
Backend is running
```

---

### `POST /detect`
Accepts an image and returns detected object labels.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `image` (image file — JPEG, PNG, etc.)

**Success Response (`200`):**
```json
{
  "detections": ["person", "car"]
}
```

**Error Response (`400` / `422` / `500`):**
```json
{
  "error": "Description of what went wrong"
}
```

---

## Example (curl)
```bash
curl -X POST http://localhost:5000/detect \
  -F "image=@/path/to/your/photo.jpg"
```

---

## Notes
- CORS is enabled — any frontend origin can connect.
- The YOLO model is loaded **once** at startup for performance.
- Detection returns **unique** labels (no duplicates).
