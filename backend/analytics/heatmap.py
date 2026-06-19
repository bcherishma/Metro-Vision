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
 
    def get_overlay(self, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Renders the current accumulated heatmap as a colored BGRA overlay.
        """
        h, w = frame_shape[:2]
 
        if self._grid is None or self._grid.max() == 0:
            return np.zeros((h, w, 4), dtype=np.uint8)
 
        # Apply Gaussian blur 
        blurred = gaussian_filter(self._grid, sigma=15)
 
        normalised = cv2.normalize(
            blurred, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
 
        colored = cv2.applyColorMap(normalised, cv2.COLORMAP_JET)  
 
        alpha_channel = normalised  
        bgra = cv2.merge([
            colored[:, :, 0],   # B
            colored[:, :, 1],   # G
            colored[:, :, 2],   # R
            alpha_channel,      # A
        ])
 
        return bgra
 
    def blend_onto_frame(
        self, frame: np.ndarray, frame_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Blends the heatmap overlay onto a BGR frame.
        """
        overlay_bgra = self.get_overlay(frame_shape)
        overlay_bgr = overlay_bgra[:, :, :3]
        alpha_mask = overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
 
        blended = (
            frame.astype(np.float32) * (1 - alpha_mask * self.alpha)
            + overlay_bgr.astype(np.float32) * alpha_mask * self.alpha
        ).astype(np.uint8)
 
        return blended
 
    def get_heatmap_image(self) -> Optional[np.ndarray]:
        
        if self._grid is None or self._grid.max() == 0:
            return None
 
        blurred = gaussian_filter(self._grid, sigma=15)
        normalised = cv2.normalize(
            blurred, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        return cv2.applyColorMap(normalised, cv2.COLORMAP_JET)
 
    def reset(self) -> None:
        self._grid = None
        self._frame_count = 0
        logger.info("HeatmapGenerator reset.")
 