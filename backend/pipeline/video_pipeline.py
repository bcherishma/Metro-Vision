from __future__ import annotations
import time
from pathlib import Path
from typing import Generator, Optional
import cv2
import numpy as np
import yaml
from loguru import logger
from backend.core.detector import ObjectDetector
from backend.core.tracker import ObjectTracker
from backend.analytics.traffic_flow import TrafficFlowAnalyser
from backend.analytics.heatmap import HeatmapGenerator
from backend.analytics.anomaly import AnomalyDetector
from backend.analytics.bird_eye_view import BirdEyeViewTransformer
from backend.pipeline.frame_utils import (
    draw_tracks,
    draw_congestion_overlay,
    draw_fps_counter,
    resize_with_aspect,
    bgr_to_rgb,
)

class VideoPipeline:
    def __init__(self, config_path: str = "configs/settings.yaml") -> None:
        logger.info("Initialising MetroVision pipeline...")
 
        cfg = yaml.safe_load(Path(config_path).read_text())
        self._cfg = cfg
        self._video_cfg    = cfg["video"]
        self._analytics_cfg = cfg["analytics"]
 
        self.detector    = ObjectDetector(config_path)
        self.tracker     = ObjectTracker()
        self.flow        = TrafficFlowAnalyser(cfg["analytics"])
        self.heatmap     = HeatmapGenerator(cfg["analytics"])
        self.anomaly     = AnomalyDetector(config_path)
        self.bev         = BirdEyeViewTransformer(config_path)
 
        logger.success("MetroVision pipeline ready.")
 
    def process_video(
        self,
        video_path: str,
        show_trails: bool = True,
        show_bev: bool = True,
    ) -> Generator[dict, None, None]:
        
        video_path = str(video_path)
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
 
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")
 
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_skip = self._video_cfg["frame_skip"]
        display_width = self._video_cfg["display_width"]
        congestion_threshold = self._analytics_cfg["congestion_threshold"]
 
        # Reset all stateful modules for this new video
        self._reset_all()
 
        frame_number = 0
        t_prev = time.perf_counter()
 
        logger.info(f"Processing video: {video_path}")
 
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
 
                frame_number += 1
 
                if frame_number % frame_skip != 0:
                    continue
 
                frame = resize_with_aspect(frame, display_width)
 
                detections = self.detector.detect(frame)
 
                tracked = self.tracker.update(detections)
 
                flow_stats = self.flow.update(tracked)
 
                self.heatmap.update(tracked, frame.shape)
 
                anomalies = self.anomaly.detect(frame)
 
                bev_frame = None
                if show_bev:
                    bev_raw = self.bev.transform_frame(frame.copy())
                    bev_frame = self.bev.draw_bev_vehicles(bev_raw, tracked)
 
                annotated = draw_tracks(
                    frame.copy(), tracked,
                    track_history=self.tracker.track_history,
                    show_trails=show_trails,
                )
                annotated = draw_congestion_overlay(
                    annotated,
                    vehicle_count=flow_stats["vehicle_count"],
                    threshold=congestion_threshold,
                )
 
                t_now = time.perf_counter()
                fps = 1.0 / max(t_now - t_prev, 1e-6)
                t_prev = t_now
                annotated = draw_fps_counter(annotated, fps)
 
                annotated_rgb = bgr_to_rgb(annotated)
                bev_rgb = bgr_to_rgb(bev_frame) if bev_frame is not None else None
 
                timestamp_sec = (frame_number / video_fps)
 
                yield {
                    "annotated_frame_rgb": annotated_rgb,
                    "bev_frame_rgb":       bev_rgb,
                    "flow_stats":          flow_stats,
                    "heatmap_overlay":     self.heatmap.get_overlay(frame.shape),
                    "anomalies":           anomalies,
                    "tracked_objects":     tracked,
                    "frame_number":        frame_number,
                    "fps":                 round(fps, 1),
                    "timestamp_sec":       round(timestamp_sec, 2),
                    "unique_count":        self.tracker.get_unique_count(),
                }
 
        finally:
            cap.release()
            logger.info(f"Pipeline finished. Frames processed: {frame_number // frame_skip}")
 
    def get_heatmap_image(self) -> Optional[np.ndarray]:
        return self.heatmap.get_heatmap_image()
 
    def _reset_all(self) -> None:
        self.tracker.reset()
        self.flow.reset()
        self.heatmap.reset()
        logger.debug("All pipeline modules reset.")
 