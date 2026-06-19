from __future__ import annotations
import time
from collections import deque
from typing import List
from loguru import logger
from backend.core.tracker import TrackedObject
 
class TrafficFlowAnalyser:
    def __init__(self, analytics_cfg: dict) -> None:
        """
        The analytics  section from settings.yaml
        """
        self.congestion_threshold: int  = analytics_cfg["congestion_threshold"]
        self.window_seconds: float      = analytics_cfg["stats_window_seconds"]
        self._count_history: deque[tuple[float, int]] = deque()
        self._total_frames_processed: int = 0
        self._peak_count: int = 0
 
        logger.info("TrafficFlowAnalyser initialised.")
 
 
    def update(self, tracked_objects: List[TrackedObject]) -> dict:
        
        now = time.time()
        vehicle_count = len(tracked_objects)
        self._count_history.append((now, vehicle_count))
        self._prune_old_history(now)
        self._total_frames_processed += 1
        if vehicle_count > self._peak_count:
            self._peak_count = vehicle_count
 
        class_breakdown  = self._compute_class_breakdown(tracked_objects)
        rolling_avg      = self._compute_rolling_average()
        congestion_score = self._compute_congestion_score(vehicle_count)
 
        return {
            "vehicle_count":    vehicle_count,
            "class_breakdown":  class_breakdown,
            "congestion_score": congestion_score,
            "is_congested":     vehicle_count >= self.congestion_threshold,
            "rolling_avg":      round(rolling_avg, 1),
            "peak_count":       self._peak_count,
            "frames_processed": self._total_frames_processed,
        }
 
    def get_count_timeseries(self) -> list[dict]:
        return [
            {"time": t, "count": c}
            for t, c in self._count_history
        ]
 
    def reset(self) -> None:
        self._count_history.clear()
        self._total_frames_processed = 0
        self._peak_count = 0
        logger.info("TrafficFlowAnalyser reset.")
 
    def _prune_old_history(self, now: float) -> None:
        """
        Removes entries older than window_seconds from the rolling deque.
        This keeps memory usage bounded regardless of video length.
        """
        cutoff = now - self.window_seconds
        while self._count_history and self._count_history[0][0] < cutoff:
            self._count_history.popleft()
 
    def _compute_class_breakdown(
        self, tracked_objects: List[TrackedObject]
    ) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for obj in tracked_objects:
            breakdown[obj.class_name] = breakdown.get(obj.class_name, 0) + 1
        return breakdown
 
    def _compute_rolling_average(self) -> float:
        if not self._count_history:
            return 0.0
        counts = [c for _, c in self._count_history]
        return sum(counts) / len(counts)
 
    def _compute_congestion_score(self, vehicle_count: int) -> float:
        """
        Maps vehicle count to a 0–100 congestion score.
 
        Formula: min(100, (count / threshold) * 100)
 
        A score of 100 means the scene is at or beyond the congestion
        threshold. A score of 50 means half the threshold is reached.
        This normalized score is more meaningful than a raw count
        for dashboard gauges and alerts.
        """
        score = (vehicle_count / self.congestion_threshold) * 100
        return round(min(100.0, score), 1)
 