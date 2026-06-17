from __future__ import annotations
 
from dataclasses import dataclass
from typing import List
 
import numpy as np
import supervision as sv
from loguru import logger
 
from backend.core.detector import Detection

@dataclass
class TrackedObject:
    
    track_id: int
    bbox: np.ndarray        
    confidence: float
    class_id: int
    class_name: str
 
    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
 
    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

class ObjectTracker:
    
    MAX_HISTORY_LEN: int = 30
 
    def __init__(self) -> None:
        self._tracker = sv.ByteTrack()
 
        # Used by the frontend to draw trajectory trails
        self.track_history: dict[int, list[tuple[float, float]]] = {}
 
        logger.info("ObjectTracker (ByteTrack) initialised.")
 
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        
        if not detections:
            return []
 
        # Convert our Detection dataclass → supervision's Detections format.
        # supervision is the bridge between YOLO output and ByteTrack input.
        sv_detections = sv.Detections(
            xyxy=np.array([d.bbox for d in detections], dtype=float),
            confidence=np.array([d.confidence for d in detections], dtype=float),
            class_id=np.array([d.class_id for d in detections], dtype=int),
        )
 
        tracked_sv = self._tracker.update_with_detections(sv_detections)
 
        id_to_name = {d.class_id: d.class_name for d in detections}
 
        tracked_objects: List[TrackedObject] = []
 
        for i, track_id in enumerate(tracked_sv.tracker_id):
            tid = int(track_id)
            cls_id = int(tracked_sv.class_id[i])
            bbox = tracked_sv.xyxy[i]
 
            obj = TrackedObject(
                track_id=tid,
                bbox=bbox,
                confidence=float(tracked_sv.confidence[i]),
                class_id=cls_id,
                class_name=id_to_name.get(cls_id, "vehicle"),
            )
            tracked_objects.append(obj)
 
            cx, cy = obj.center
            if tid not in self.track_history:
                self.track_history[tid] = []
            history = self.track_history[tid]
            history.append((cx, cy))
            if len(history) > self.MAX_HISTORY_LEN:
                history.pop(0)
 
        return tracked_objects
 
    def get_unique_count(self) -> int:
        return len(self.track_history)
 
    def reset(self) -> None:
        """
        Resets tracker state completely.
        MUST be called between different videos to avoid ID contamination.
        """
        self._tracker = sv.ByteTrack()
        self.track_history.clear()
        logger.info("ObjectTracker reset.")
 