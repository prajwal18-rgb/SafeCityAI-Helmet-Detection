# SafeCityAI — Helmet Violation Detection System

> Real-time helmet violation detection using YOLOv5 + FastAPI  
> Built for smart city traffic enforcement automation

---

## Project Overview

SafeCityAI is an AI-powered traffic violation detection system built for the SafeCityAI case study. The system automatically watches traffic footage and identifies riders **with** and **without** helmets in real time — replacing thousands of hours of manual CCTV review by traffic police.

---

## Demo

The model detects helmet violations on real CCTV footage with bounding boxes drawn on every frame in real time.

- **With Helmet** — Blue bounding box
- **Without Helmet** — Light Blue bounding box + violation alert

---

## Model Performance

| Metric | Score |
|---|---|
| mAP@0.5 | **81.4%** |
| With Helmet Precision | **77.2%** |
| Without Helmet Precision | **75.0%** |
| With Helmet Recall | **89.8%** |
| Training Time | **~7 minutes** (Tesla T4 GPU) |
| Epochs | 50 |
| Architecture | YOLOv5s (Transfer Learning from COCO) |

![Training Results](Screenshot (275).png)

---

## Dataset

- **Source**: Roboflow Universe — Helmet Detector dataset
- **Images**: 629 total (447 train / 104 val / 78 test)
- **Classes**: `With Helmet`, `Without Helmet`
- **Format**: YOLOv5 PyTorch (normalized .txt annotations)
- **Augmentations**: Mosaic, horizontal flip, brightness/contrast

---

## Project Structure

```
SafeCityAI/
  server.py        — FastAPI inference endpoint
  index.html       — Dashboard UI (streetlight theme)
  results.png      — Training loss and mAP charts
  SafeCityAI.ipynb — Full training notebook (Google Colab)
  best.pt          — Trained model weights (available on request)
```

---

## API Usage

Start the server:
```bash
pip install fastapi uvicorn torch pillow python-multipart
python server.py
```

Send a POST request with an image:
```bash
curl -X POST "http://localhost:8000/detect" -F "file=@image.jpg"
```

Example JSON response:
```json
{
  "detections": [
    {
      "class": "Without Helmet",
      "confidence": 0.88,
      "box": [100, 200, 50, 60]
    },
    {
      "class": "With Helmet",
      "confidence": 0.92,
      "box": [300, 150, 60, 70]
    }
  ],
  "total_detections": 2,
  "violations": 1
}
```

Interactive API docs available at `http://localhost:8000/docs`

---

## Dashboard UI

Open `index.html` in your browser while `server.py` is running.

Features:
- Drag and drop image upload
- Live AI detection with scan animation
- Side by side image preview and results
- Confidence bars for each detection
- Violation alert vs all clear status
- Detection stats — total, safe, violations

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv5s (Ultralytics) |
| Training | Google Colab + Tesla T4 GPU |
| Backend API | FastAPI + Uvicorn |
| Frontend | HTML + CSS + Vanilla JS |
| Dataset Management | Roboflow |
| Deep Learning | PyTorch |

---

## How to Run Locally

**1. Clone the repo:**
```bash
git clone https://github.com/PrajwalDhoke/SafeCityAI-Helmet-Detection
cd SafeCityAI-Helmet-Detection
```

**2. Install dependencies:**
```bash
pip install fastapi uvicorn torch pillow python-multipart
```

**3. Add your model weights:**
Place `best.pt` in the project folder (available on request)

**4. Start the API:**
```bash
python server.py
```

**5. Open the dashboard:**
Double click `index.html` in your browser

---

## Key Concepts Used

- **Transfer Learning** — fine-tuned YOLOv5s pretrained on COCO dataset
- **mAP@0.5** — Mean Average Precision at 0.5 IoU threshold
- **Non-Maximum Suppression (NMS)** — removes duplicate bounding boxes
- **Mosaic Augmentation** — combines 4 images for better small object detection
- **Confidence Threshold** — only detections above 50% confidence are shown

---

*Built as part of AI/ML Internship — SafeCityAI Case Study*
