from __future__ import annotations
from typing import List, Optional
import cv2
import numpy as np
from backend.core.tracker import TrackedObject

#BGR is used in OpenCV
COLORS: dict[str, tuple[int, int, int]] = {
    "car":        (255, 180,  50),   # amber
    "truck":      (255,  80,  80),   # coral red
    "bus":        ( 80, 200, 255),   # sky blue
    "person":     ( 80, 255, 150),   # mint green
    "bicycle":    (200, 100, 255),   # purple
    "motorcycle": (255, 220,  80),   # yellow
    "default":    (200, 200, 200),   # gray fallback
}
 
CONGESTION_COLOR = (0, 0, 220)

def draw_tracks(
    frame: np.ndarray,
    tracked_objects: List[TrackedObject],
    track_history: Optional[dict] = None,
    show_trails: bool = True,
) -> np.ndarray:
    
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.bbox.astype(int)
        color = COLORS.get(obj.class_name, COLORS["default"])
 
        # --- Bounding box ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
 
        # --- Label background pill ---
        label = f"#{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        # Dark pill behind the text for readability
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - baseline - 4),
            (x1 + label_w + 6, y1),
            color,
            thickness=-1,   # -1 = filled rectangle
        )
        
        cv2.putText(
            frame, label,
            (x1 + 3, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255),
            thickness=1, lineType=cv2.LINE_AA,
        )
 
        # --- Trajectory trail ---
        if show_trails and track_history and obj.track_id in track_history:
            trail = track_history[obj.track_id]
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
                pt2 = (int(trail[i][0]),     int(trail[i][1]))
                thickness = max(1, int(alpha * 3))
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
 
    return frame
 
 
def draw_congestion_overlay(
    frame: np.ndarray,
    vehicle_count: int,
    threshold: int,
    roi_points: Optional[np.ndarray] = None,
) -> np.ndarray:

    if vehicle_count < threshold:
        return frame
 
    overlay = frame.copy()
 
    if roi_points is not None:
        cv2.fillPoly(overlay, [roi_points], CONGESTION_COLOR)
    else:
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                      CONGESTION_COLOR, -1)
 
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
 
    # Warning banner at the top
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 180), -1)
    cv2.putText(
        frame,
        f"  CONGESTION ALERT — {vehicle_count} vehicles detected",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255),
        thickness=1, lineType=cv2.LINE_AA,
    )
 
    return frame
 
 
def draw_fps_counter(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draws an FPS counter in the bottom-right corner."""
    label = f"FPS: {fps:.1f}"
    h, w = frame.shape[:2]
    cv2.putText(
        frame, label,
        (w - 110, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0, 255, 100),
        thickness=1, lineType=cv2.LINE_AA,
    )
    return frame
 
 
def resize_with_aspect(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_LINEAR)
 
 
def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 