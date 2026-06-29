from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np
import yaml
from loguru import logger
from pathlib import Path
from backend.core.tracker import TrackedObject
 
 
class BirdEyeViewTransformer:
    """
    Computes and applies a perspective warp to generate a top-down
    bird's eye view of the road scene.
    """
    def __init__(self, config_path: str = "configs/settings.yaml") -> None:
        config = yaml.safe_load(Path(config_path).read_text())
        analytics_cfg = config["analytics"]
 
        src = np.array(
            analytics_cfg["bev_src_points"], dtype=np.float32
        )
 
        dst = np.array(
            analytics_cfg["bev_dst_points"], dtype=np.float32
        )
 
        self._M = cv2.getPerspectiveTransform(src, dst)
 
        # Inverse matrix — used to project BEV coordinates back to camera space
        self._M_inv = cv2.getPerspectiveTransform(dst, src)
 
        self._out_w = int(max(p[0] for p in analytics_cfg["bev_dst_points"])) + 20
        self._out_h = int(max(p[1] for p in analytics_cfg["bev_dst_points"])) + 20
 
        logger.info(f"BirdEyeViewTransformer ready. Output size: {self._out_w}x{self._out_h}")
 
    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Warps the full camera frame into bird's eye view.
        """
        return cv2.warpPerspective(
            frame, self._M,
            (self._out_w, self._out_h),
            flags=cv2.INTER_LINEAR,
        )
 
    def transform_points(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """ 
        Used to place vehicle positions on the BEV map without
        warping the entire frame (more efficient for analytics).
        """
        if not points:
            return []
 
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self._M)
        return [(float(p[0][0]), float(p[0][1])) for p in transformed]
 
    def draw_bev_vehicles(
        self,
        bev_frame: np.ndarray,
        tracked_objects: List[TrackedObject],
    ) -> np.ndarray:
        """
        Draws vehicle dots on a bird's eye view frame.
        """
        from backend.pipeline.frame_utils import COLORS
 
        ground_points = [obj.bottom_center for obj in tracked_objects]
        bev_points = self.transform_points(ground_points)
 
        for obj, (bx, by) in zip(tracked_objects, bev_points):
            bx, by = int(bx), int(by)
            h, w = bev_frame.shape[:2]
            if not (0 <= bx < w and 0 <= by < h):
                continue
 
            color = COLORS.get(obj.class_name, COLORS["default"])
 
            cv2.circle(bev_frame, (bx, by), radius=8, color=color, thickness=-1)
 
            cv2.putText(
                bev_frame, f"#{obj.track_id}",
                (bx + 10, by + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                color, thickness=1, lineType=cv2.LINE_AA,
            )
 
        return bev_frame
 