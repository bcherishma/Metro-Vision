from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np
import yaml
from loguru import logger
from ultralytics import YOLO

@dataclass
class RoadAnomaly:
    """Represents a detected road surface anomaly."""
    bbox: np.ndarray            
    confidence: float
    anomaly_type: str           # "pothole", "hump", "debris", etc.
    severity: str    

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
 
    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

class AnomalyDetector:
    
    ANOMALY_CLASSES: dict[int, str] = {
        0: "pothole",
        1: "hump",
        2: "debris",
    }
 
    SEVERITY_THRESHOLDS = {
        "low":    5_000,    
        "medium": 20_000,   
        "high":   float("inf"),
    }
 
    # Only run anomaly detection on the bottom N% of the frame (road region)
    ROAD_REGION_FRACTION: float = 0.55

    def __init__(self, config_path: str = "configs/settings.yaml") -> None:
        config = yaml.safe_load(Path(config_path).read_text())
        model_cfg = config["model"]
 
        model_path = model_cfg["anomaly_model"]
        self.device = model_cfg["device"]
        self.conf   = model_cfg["confidence_threshold"]
 
        logger.info(f"Loading anomaly model: {model_path}")
        self.model = YOLO(model_path)
        logger.success("AnomalyDetector ready.")
 
    def detect(self, frame: np.ndarray) -> List[RoadAnomaly]:
    
        h, w = frame.shape[:2]
 
        # Crop to road region only (bottom ROAD_REGION_FRACTION of frame)
        road_start_y = int(h * (1 - self.ROAD_REGION_FRACTION))
        road_crop = frame[road_start_y:, :]
 
        results = self.model(
            road_crop,
            device=self.device,
            conf=self.conf,
            verbose=False,
        )[0]
 
        anomalies: List[RoadAnomaly] = []
 
        for box in results.boxes:
            cls_id = int(box.cls.item())
            bbox = box.xyxy.cpu().numpy().squeeze().astype(float)
 
            # Translate Y coordinates back to full-frame space
            bbox[1] += road_start_y
            bbox[3] += road_start_y
 
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            severity = self._classify_severity(area)
 
            anomalies.append(RoadAnomaly(
                bbox=bbox,
                confidence=float(box.conf.item()),
                anomaly_type=self.ANOMALY_CLASSES.get(cls_id, "pothole"),
                severity=severity,
            ))
 
        return anomalies
 
    def _classify_severity(self, area_px: float) -> str:
        
        if area_px < self.SEVERITY_THRESHOLDS["low"]:
            return "low"
        elif area_px < self.SEVERITY_THRESHOLDS["medium"]:
            return "medium"
        return "high"
 