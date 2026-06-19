from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter
from backend.core.tracker import TrackedObject

class HeatmapGenerator:
    """
    Accumulates tracked object positions into a spatial density heatmap.
    """
 
    def __init__(self, analytics_cfg: dict) -> None:
        self.alpha: float = analytics_cfg["heatmap_alpha"]
        self._grid: Optional[np.ndarray] = None
        self._frame_count: int = 0
 
        logger.info("HeatmapGenerator initialised.")
 
    def update(
        self,
        tracked_objects: List[TrackedObject],
        frame_shape: Tuple[int, int, int],
    ) -> None:
        h, w = frame_shape[:2]
        
        if self._grid is None:
            self._grid = np.zeros((h, w), dtype=np.float32)
            logger.debug(f"Heatmap grid initialised: {w}x{h}")
 
        for obj in tracked_objects:
            cx, cy = obj.bottom_center
            cx, cy = int(cx), int(cy)
 
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
 
            cv2.circle(self._grid, (cx, cy), radius=20, color=1.0, thickness=-1)
 
        self._frame_count += 1
 
   