# MetroVision 🏙️
### Urban Scene Perception Engine

> Real-time computer vision system for intelligent urban mobility — vehicle detection, multi-object tracking, traffic flow analytics, road anomaly detection, and bird's eye view scene understanding.

---

## Overview

MetroVision is a production-structured computer vision pipeline designed for urban scene understanding. It targets the perception layer that underlies modern Advanced Driver Assistance Systems (ADAS) and smart city infrastructure .

Built with: **YOLOv8 · ByteTrack · OpenCV · Streamlit · Plotly · PyTorch (MPS)**

---

## Features

| Module | Description |
|--------|-------------|
| **Object Detection** | YOLOv8n optimised for urban classes: cars, trucks, buses, motorcycles, bicycles, pedestrians |
| **Multi-Object Tracking** | ByteTrack — persistent track IDs, trajectory history, unique vehicle counting |
| **Traffic Flow Analytics** | Per-frame vehicle counts, rolling averages, congestion scoring (0–100) |
| **Activity Heatmap** | Cumulative spatial density map of all detected activity |
| **Road Anomaly Detection** | Pothole and speed hump detection via fine-tuned YOLOv8 |
| **Bird's Eye View** | Perspective transform to overhead 2D scene map |

---

## Project Structure

```
metrovision/
├── backend/
│   ├── core/
│   │   ├── detector.py          # YOLOv8 wrapper (Facade pattern)
│   │   └── tracker.py           # ByteTrack wrapper with trajectory history
│   ├── analytics/
│   │   ├── traffic_flow.py      # Vehicle counting and congestion scoring
│   │   ├── heatmap.py           # Spatial density accumulator
│   │   ├── anomaly.py           # Road surface anomaly detector
│   │   └── bird_eye_view.py     # Perspective transform (BEV)
│   ├── pipeline/
│   │   ├── video_pipeline.py    # Main orchestrator (Generator pattern)
│   │   └── frame_utils.py       # Drawing utilities
│   └── api/
│       ├── routes.py            # FastAPI REST endpoints
│       └── schemas.py           # Pydantic request/response models
├── frontend/
│   ├── app.py                   # Streamlit entry point
│   ├── pages/
│   │   ├── 1_dashboard.py       # Live detection + metrics
│   │   ├── 2_heatmap.py         # Heatmap visualisation
│   │   ├── 3_anomaly.py         # Road anomaly viewer
│   │   └── 4_bird_eye_view.py   # BEV map
│   └── components/
│       ├── sidebar.py           # Shared sidebar (DRY principle)
│       ├── charts.py            # Plotly chart functions
│       ├── metrics_card.py      # KPI metric cards
│       └── video_player.py      # OpenCV → Streamlit bridge
├── configs/
│   └── settings.yaml            # All parameters — no hardcoded values
├── data/
│   ├── models/                  # YOLOv8 weights (not in Git)
│   ├── samples/                 # Test videos (not in Git)
│   └── outputs/                 # Processed results
├── tests/
│   ├── test_detector.py         # Unit tests: detection + analytics
│   └── test_pipeline.py         # Integration tests: full pipeline
├── requirements.txt
└── .env
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/bcherishma/metrovision.git
cd metrovision
python3 -m venv venv
source venv/bin/activate       # Mac / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Apple Silicon GPU (MPS) support

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

### 4. Download model weights

YOLOv8 weights download automatically on first run. For the road anomaly model:

```bash
# Download from Roboflow (requires account):
# https://universe.roboflow.com/pothole-detection
# Place as: data/models/pothole_yolov8.pt
```

### 5. Run the dashboard

```bash
streamlit run frontend/app.py
```

---

## Configuration

All parameters live in `configs/settings.yaml` — no hardcoded values in source code.

```yaml
model:
  device: "mps"               # mps | cpu | cuda
  confidence_threshold: 0.45
analytics:
  congestion_threshold: 10    # vehicles → congestion alert
```

---

## Running Tests

```bash
pytest tests/ -v --cov=backend --cov-report=term-missing
```

Tests cover detection geometry, congestion scoring logic, and heatmap state management — all without requiring model weights, so they run fast in CI.

---

## Technical Architecture

```
Video Input
    ↓
OpenCV Frame Reader
    ↓
YOLOv8 Detector  ──→  Urban class filtering (6 classes)
    ↓
ByteTrack Tracker  ──→  Persistent IDs + trajectory history
    ↓
┌────────────────────────────────────┐
│  Analytics (parallel)              │
│  ├── Traffic Flow (congestion)     │
│  ├── Heatmap (density)             │
│  ├── Anomaly (potholes)            │
│  └── Bird's Eye View (BEV)         │
└────────────────────────────────────┘
    ↓
Streamlit Dashboard  ──→  Live video + Plotly charts
```

---

## Relevance to ADAS & Intelligent Mobility

MetroVision implements the perception layer of an autonomous driving stack:

- **Detection** → equivalent to sensor fusion output in ADAS
- **Tracking** → object permanence, required for trajectory prediction
- **BEV Transform** → standard coordinate space in parking-assist and surround-view systems
- **Congestion scoring** → feeds navigation re-routing in connected vehicles
- **Anomaly detection** → road condition data, similar in purpose to BMW ConnectedDrive / HERE Maps road quality pipelines

---

## Roadmap

- [ ] Fine-tune anomaly model on a labelled pothole dataset
- [ ] Add FastAPI REST layer for headless/batch processing
- [ ] Export analytics session as downloadable PDF report
- [ ] Multi-camera fusion for intersection-level coverage

---

## Author

**Cherishma Bodapati**
[LinkedIn](#) · [GitHub](#)