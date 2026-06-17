from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
 
import numpy as np
import yaml
from loguru import logger
from ultralytics import YOLO

@dataclass
class Detection:
    """
    Represents a single detected object in one video frame.
    """
    bbox: np.ndarray        
    confidence: float       
    class_id: int           
    class_name: str         

    @property
    def center(self) -> tuple[float, float]:
        """Returns the (cx, cy) center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
 
    @property
    def area(self) -> float:
        """Returns bounding box area in pixels squared."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
class ObjectDetector:
    """
    Loads a YOLOv8 model and runs inference on BGR frames from OpenCV.
    """
    URBAN_CLASSES: dict[int, str] = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        9: "traffic light",
        11: "stop sign"
    }
 
    def __init__(self, config_path: str = "configs/settings.yaml") -> None:

        config = yaml.safe_load(Path(config_path).read_text())
        model_cfg = config["model"]
 
        model_path = model_cfg["detection_model"]
        self.device     = model_cfg["device"]
        self.conf       = model_cfg["confidence_threshold"]
        self.iou        = model_cfg["iou_threshold"]
        self.class_ids  = list(self.URBAN_CLASSES.keys())
 
        logger.info(f"Loading detection model: {model_path} on device={self.device}")
        self.model = YOLO(model_path)
        logger.success("ObjectDetector ready.")
 
    def detect(self, frame: np.ndarray) -> List[Detection]:
        
        results = self.model(
            frame,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            classes=self.class_ids,
            verbose=False,      # suppress YOLO's per-frame console output
        )[0]                    # [0] because we process one frame at a time
 
        detections: List[Detection] = []
 
        for box in results.boxes:
            cls_id = int(box.cls.item())
            detections.append(Detection(
                bbox=box.xyxy.cpu().numpy().squeeze().astype(float),
                confidence=float(box.conf.item()),
                class_id=cls_id,
                class_name=self.URBAN_CLASSES.get(cls_id, "unknown"),
            ))
 
        return detections
 
    def get_class_counts(self, detections: List[Detection]) -> dict[str, int]:
    
        counts: dict[str, int] = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts
 