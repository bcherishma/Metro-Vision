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