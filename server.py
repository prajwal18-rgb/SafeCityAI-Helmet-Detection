"""
SafeCityAI — Helmet Detection API
Built with FastAPI + YOLOv5
"""

import io
import torch
import pathlib
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ── Windows fix — model was trained on Linux (Colab) ───────
pathlib.PosixPath = pathlib.WindowsPath

# ── App setup ──────────────────────────────────────────────
app = FastAPI(
    title="SafeCityAI Helmet Detection API",
    description="Detects helmet violations using YOLOv5",
    version="1.0.0"
)

# ── CORS — allows browser to connect to the API ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ──────────────────────────────
MODEL_PATH = "best.pt"

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=MODEL_PATH,
    force_reload=False
)
model.conf = 0.5   # Confidence threshold — only show boxes above 50%
model.iou  = 0.45  # IoU threshold for NMS


# ── Health check endpoint ───────────────────────────────────
@app.get("/")
def root():
    return {"status": "SafeCityAI API is running!"}


# ── Main detection endpoint ─────────────────────────────────
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Send a POST request with an image file.
    Returns JSON with detected classes, confidence scores and bounding boxes.

    Example response:
    {
        "detections": [
            {"class": "Without Helmet", "confidence": 0.88, "box": [100, 200, 50, 60]},
            {"class": "With Helmet",    "confidence": 0.92, "box": [300, 150, 60, 70]}
        ],
        "total_detections": 2,
        "violations": 1
    }
    """

    # Read uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run YOLOv5 inference
    results = model(image)

    # Parse detections
    detections = []
    violations  = 0

    for *box, confidence, class_id in results.xyxy[0].tolist():
        x1, y1, x2, y2 = [int(c) for c in box]
        class_name      = model.names[int(class_id)]
        conf_score      = round(confidence, 2)

        detections.append({
            "class":      class_name,
            "confidence": conf_score,
            "box":        [x1, y1, x2 - x1, y2 - y1]
        })

        if class_name == "Without Helmet":
            violations += 1

    return JSONResponse(content={
        "detections":       detections,
        "total_detections": len(detections),
        "violations":       violations
    })


# ── Run server ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)