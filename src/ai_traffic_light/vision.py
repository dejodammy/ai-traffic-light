from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(slots=True)
class LaneObservation:
    approach: str
    vehicle_count: float
    waiting_time: float = 0.0
    density: float = 0.0


@dataclass(slots=True)
class DetectionZone:
    approach: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(slots=True)
class VehicleDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    approach: str
    confidence: float


DEFAULT_APPROACHES = ("north", "east", "south", "west")


def default_zones(width: int, height: int, labels: Iterable[str] = DEFAULT_APPROACHES) -> list["DetectionZone"]:
    """Split a frame into equal vertical bands, one detection zone per approach."""
    labels = list(labels)
    band = max(width // max(len(labels), 1), 1)
    zones = []
    for idx, label in enumerate(labels):
        x1 = idx * band
        x2 = width if idx == len(labels) - 1 else (idx + 1) * band
        zones.append(DetectionZone(approach=label, x1=x1, y1=0, x2=x2, y2=height))
    return zones


def build_state_from_observations(
    observations: Iterable[LaneObservation],
    approach_order: list[str],
    queue_norm: float = 20.0,
    wait_norm: float = 120.0,
    max_approaches: int = 4,
) -> np.ndarray:
    grouped = {approach: LaneObservation(approach=approach, vehicle_count=0.0, waiting_time=0.0, density=0.0) for approach in approach_order}
    for obs in observations:
        current = grouped.setdefault(obs.approach, LaneObservation(obs.approach, 0.0, 0.0, 0.0))
        current.vehicle_count += obs.vehicle_count
        current.waiting_time += obs.waiting_time
        current.density = max(current.density, obs.density)

    state = []
    for approach in approach_order[:max_approaches]:
        obs = grouped[approach]
        state.extend(
            [
                obs.vehicle_count / max(queue_norm, 1.0),
                obs.waiting_time / max(wait_norm, 1.0),
                obs.density,
            ]
        )

    while len(state) < max_approaches * 3:
        state.extend([0.0, 0.0, 0.0])
    return np.asarray(state[: max_approaches * 3], dtype=np.float32)


@dataclass(slots=True)
class EmergencyAlert:
    approach: str
    vehicle_type: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class EmergencyVehicleDetector:
    """Detects emergency vehicles using a fine-tuned YOLOv8 model.

    To get a model:
      1. Download from Roboflow Universe — search "emergency vehicles" and
         export as YOLOv8. Save the .pt file and pass its path here.
      2. Or fine-tune your own:
         yolo train model=yolov8n.pt data=emergency_vehicles.yaml epochs=100
    """

    DEFAULT_CLASSES = {"ambulance", "fire truck", "fire_truck", "police", "emergency"}

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.40,
        emergency_classes: set[str] | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.emergency_classes = emergency_classes or self.DEFAULT_CLASSES

    def detect_frame(self, image: np.ndarray, zones: list[DetectionZone]) -> list[EmergencyAlert]:
        results = self.model.predict(image, conf=self.confidence, verbose=False)
        alerts: list[EmergencyAlert] = []
        for box in results[0].boxes:
            cls_name = self.model.names[int(box.cls[0].item())].lower()
            if not any(ec in cls_name for ec in self.emergency_classes):
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0].item())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for zone in zones:
                if zone.x1 <= cx <= zone.x2 and zone.y1 <= cy <= zone.y2:
                    alerts.append(EmergencyAlert(zone.approach, cls_name, conf, x1, y1, x2, y2))
                    break
        return alerts


class ColorFlashDetector:
    """Detects emergency vehicles by their red/blue flashing lights.

    Requires no training — scans detection zones for high-saturation red or
    blue regions that indicate emergency lights. Works on any camera feed.

    Tune ``min_flash_fraction`` if you get false positives (raise it) or miss
    real emergency vehicles (lower it). Default 0.008 works well indoors;
    real outdoor cameras may need 0.005.
    """

    def __init__(self, min_flash_fraction: float = 0.008) -> None:
        self.min_flash_fraction = min_flash_fraction

    def detect_frame(self, image: np.ndarray, zones: list[DetectionZone]) -> list[EmergencyAlert]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # red wraps around 0/180 in HSV
        red_lo = cv2.inRange(hsv, np.array([0,   150, 150]), np.array([10,  255, 255]))
        red_hi = cv2.inRange(hsv, np.array([170, 150, 150]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_lo, red_hi)

        blue_mask = cv2.inRange(hsv, np.array([100, 150, 150]), np.array([130, 255, 255]))
        flash_mask = cv2.bitwise_or(red_mask, blue_mask)

        alerts: list[EmergencyAlert] = []
        for zone in zones:
            zone_area = max((zone.x2 - zone.x1) * (zone.y2 - zone.y1), 1)
            roi = flash_mask[zone.y1:zone.y2, zone.x1:zone.x2]
            flash_pixels = int(roi.sum() / 255)
            fraction = flash_pixels / zone_area
            if fraction < self.min_flash_fraction:
                continue
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            rx, ry, rw, rh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            alerts.append(EmergencyAlert(
                zone.approach, "emergency_light", round(fraction, 4),
                zone.x1 + rx, zone.y1 + ry,
                zone.x1 + rx + rw, zone.y1 + ry + rh,
            ))
        return alerts


class MockDetector:
    def __init__(self, observations: Iterable[LaneObservation]):
        self._observations = list(observations)

    def detect(self, _image_path: str, _zones=None):
        return list(self._observations)

    def detect_frame(self, _image, _zones=None):
        return list(self._observations)


class YOLOVehicleDetector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.25):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.vehicle_classes = {2, 3, 5, 7}

    def detect(self, image_path: str, zones: list[DetectionZone]) -> list[LaneObservation]:
        image = cv2.imread(str(Path(image_path)))
        if image is None:
            raise FileNotFoundError(f"Could not read image '{image_path}'.")
        return self.detect_frame(image, zones)

    def detect_frame(self, image: np.ndarray, zones: list[DetectionZone]) -> list[LaneObservation]:
        observations, _ = self.detect_frame_with_boxes(image, zones)
        return observations

    def detect_frame_with_boxes(
        self, image: np.ndarray, zones: list[DetectionZone]
    ) -> tuple[list[LaneObservation], list[VehicleDetection]]:
        results = self.model.predict(image, conf=self.confidence, verbose=False)
        observations = {zone.approach: LaneObservation(zone.approach, 0.0, 0.0, 0.0) for zone in zones}
        detections: list[VehicleDetection] = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in self.vehicle_classes:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0].item())
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            for zone in zones:
                if zone.x1 <= center_x <= zone.x2 and zone.y1 <= center_y <= zone.y2:
                    obs = observations[zone.approach]
                    obs.vehicle_count += 1
                    zone_area = max((zone.x2 - zone.x1) * (zone.y2 - zone.y1), 1)
                    bbox_area = max((x2 - x1) * (y2 - y1), 1)
                    obs.density = min(1.0, obs.density + (bbox_area / zone_area))
                    detections.append(VehicleDetection(x1, y1, x2, y2, zone.approach, conf))
                    break

        return list(observations.values()), detections
