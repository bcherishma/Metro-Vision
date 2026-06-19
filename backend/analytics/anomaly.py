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