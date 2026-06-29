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