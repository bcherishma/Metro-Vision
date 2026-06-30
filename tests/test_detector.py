import sys
from pathlib import Path 
import numpy as np
import pytest
from backend.core.detector import Detection, ObjectDetector
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
class TestDetection: 
    def test_center_calculation(self):
        det = Detection(
            bbox=np.array([100.0, 200.0, 300.0, 400.0]),
            confidence=0.9,
            class_id=2,
            class_name="car",
        )
        cx, cy = det.center
        assert cx == pytest.approx(200.0)
        assert cy == pytest.approx(300.0)
 
    def test_area_calculation(self):
        det = Detection(
            bbox=np.array([0.0, 0.0, 100.0, 50.0]),
            confidence=0.8,
            class_id=2,
            class_name="car",
        )
        assert det.area == pytest.approx(5000.0)
 
    def test_class_name_stored(self):
        det = Detection(
            bbox=np.array([0.0, 0.0, 50.0, 50.0]),
            confidence=0.75,
            class_id=0,
            class_name="person",
        )
        assert det.class_name == "person"
        assert det.class_id == 0
 
class TestObjectDetectorUrbanClasses:
    def test_urban_classes_contains_car(self):
        assert 2 in ObjectDetector.URBAN_CLASSES
        assert ObjectDetector.URBAN_CLASSES[2] == "car"
 
    def test_urban_classes_contains_person(self):
        assert 0 in ObjectDetector.URBAN_CLASSES
        assert ObjectDetector.URBAN_CLASSES[0] == "person"
 
    def test_urban_classes_excludes_animals(self):
        assert 16 not in ObjectDetector.URBAN_CLASSES
        assert 17 not in ObjectDetector.URBAN_CLASSES
 
    def test_urban_classes_count(self):
        assert len(ObjectDetector.URBAN_CLASSES) == 6
 
class TestTrafficFlowAnalyser: 
    def _make_analyser(self):
        from backend.analytics.traffic_flow import TrafficFlowAnalyser
        cfg = {
            "congestion_threshold": 10,
            "stats_window_seconds": 10,
        }
        return TrafficFlowAnalyser(cfg)
 
    def test_empty_frame_returns_zero(self):
        analyser = self._make_analyser()
        result = analyser.update([])
        assert result["vehicle_count"] == 0
        assert result["congestion_score"] == 0.0
        assert result["is_congested"] is False
 
    def test_congestion_threshold(self):
        from backend.core.tracker import TrackedObject
        analyser = self._make_analyser()
        fake_objects = [
            TrackedObject(
                track_id=i,
                bbox=np.array([0.0, 0.0, 50.0, 50.0]),
                confidence=0.9,
                class_id=2,
                class_name="car",
            )
            for i in range(15)
        ]
 
        result = analyser.update(fake_objects)
        assert result["vehicle_count"] == 15
        assert result["is_congested"] is True
        assert result["congestion_score"] == pytest.approx(100.0)
 
    def test_peak_count_updates(self):
        from backend.core.tracker import TrackedObject
        analyser = self._make_analyser()
 
        def make_objects(n):
            return [
                TrackedObject(i, np.array([0.0, 0.0, 50.0, 50.0]), 0.9, 2, "car")
                for i in range(n)
            ]
 
        analyser.update(make_objects(5))
        analyser.update(make_objects(12))
        analyser.update(make_objects(3))
 
        result = analyser.update(make_objects(1))
        assert result["peak_count"] == 12
 
    def test_reset_clears_history(self):
        analyser = self._make_analyser()
        from backend.core.tracker import TrackedObject
        fake = [TrackedObject(0, np.array([0.0, 0.0, 50.0, 50.0]), 0.9, 2, "car")]
        analyser.update(fake)
        analyser.reset()
        result = analyser.update([])
        assert result["peak_count"] == 0
 
class TestHeatmapGenerator:
 
    def _make_gen(self):
        from backend.analytics.heatmap import HeatmapGenerator
        return HeatmapGenerator({"heatmap_alpha": 0.6})
 
    def test_initial_overlay_is_transparent(self):
        gen = self._make_gen()
        overlay = gen.get_overlay((480, 640, 3))
        assert overlay[:, :, 3].max() == 0
 
    def test_update_populates_grid(self):
        from backend.core.tracker import TrackedObject
        gen = self._make_gen()
        fake = [TrackedObject(0, np.array([100.0, 100.0, 200.0, 200.0]), 0.9, 2, "car")]
        gen.update(fake, (480, 640, 3))
        overlay = gen.get_overlay((480, 640, 3))
        assert overlay[:, :, 3].max() > 0
 
    def test_reset_clears_grid(self):
        from backend.core.tracker import TrackedObject
        gen = self._make_gen()
        fake = [TrackedObject(0, np.array([100.0, 100.0, 200.0, 200.0]), 0.9, 2, "car")]
        gen.update(fake, (480, 640, 3))
        gen.reset()
        overlay = gen.get_overlay((480, 640, 3))
        assert overlay[:, :, 3].max() == 0
 