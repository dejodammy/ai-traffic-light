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


class MockDetector:
    def __init__(self, observations: Iterable[LaneObservation]):
        self._observations = list(observations)

    def detect(self, _image_path: str):
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

        results = self.model.predict(image, conf=self.confidence, verbose=False)
        observations = {zone.approach: LaneObservation(zone.approach, 0.0, 0.0, 0.0) for zone in zones}

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in self.vehicle_classes:
                continue
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            for zone in zones:
                if zone.x1 <= center_x <= zone.x2 and zone.y1 <= center_y <= zone.y2:
                    obs = observations[zone.approach]
                    obs.vehicle_count += 1
                    zone_area = max((zone.x2 - zone.x1) * (zone.y2 - zone.y1), 1)
                    bbox_area = max((x2 - x1) * (y2 - y1), 1)
                    obs.density = min(1.0, obs.density + (bbox_area / zone_area))
                    break

        return list(observations.values())
