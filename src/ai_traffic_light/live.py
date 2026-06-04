from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .vision import (
    DEFAULT_APPROACHES,
    DetectionZone,
    EmergencyAlert,
    LaneObservation,
    VehicleDetection,
    build_state_from_observations,
    default_zones,
)


@dataclass(slots=True)
class Decision:
    """One real-time control decision, suitable for logging or hardware output."""

    timestamp: float
    observations: list[LaneObservation]
    state: np.ndarray
    action: int
    detections: list[VehicleDetection] = None
    emergency_alerts: list[EmergencyAlert] = None
    emergency_override: bool = False


# action -> approaches it serves (matches phase 0 = E-W, phase 1 = N-S)
_ACTION_APPROACHES: dict[int, frozenset[str]] = {
    0: frozenset({"east", "west"}),
    1: frozenset({"north", "south"}),
}


class LiveController:
    """Turns a single camera frame into a phase decision: detect -> state -> DQN.

    Capture, display, hardware actuation and logging are deliberately kept out of
    this class so they can be layered on top — a SQLite logger or a GPIO bridge is
    just an ``on_decision`` callback passed to :func:`run_live`.

    Pass an ``emergency_detector`` (EmergencyVehicleDetector or ColorFlashDetector)
    to enable emergency vehicle preemption — the DQN decision is overridden and the
    approach with the emergency vehicle gets green immediately.
    """

    def __init__(
        self,
        detector,
        agent,
        approach_order: Sequence[str] = DEFAULT_APPROACHES,
        zones: Optional[list[DetectionZone]] = None,
        queue_norm: float = 20.0,
        wait_norm: float = 120.0,
        max_approaches: int = 4,
        emergency_detector=None,
        emergency_hold: float = 30.0,
    ) -> None:
        self.detector = detector
        self.agent = agent
        self.approach_order = list(approach_order)
        self.zones = zones
        self.queue_norm = queue_norm
        self.wait_norm = wait_norm
        self.max_approaches = max_approaches
        self.emergency_detector = emergency_detector
        self.emergency_hold = emergency_hold
        self._emergency_until: float = 0.0
        self._emergency_action: int | None = None

    def ensure_zones(self, frame) -> list[DetectionZone]:
        """Build equal-band detection zones from the frame size on first use."""
        if self.zones is None:
            height, width = frame.shape[:2]
            self.zones = default_zones(width, height, self.approach_order)
        return self.zones

    def _emergency_action_for(self, approach: str) -> int | None:
        for action, approaches in _ACTION_APPROACHES.items():
            if approach in approaches:
                return action
        return None

    def decide(self, frame) -> Decision:
        zones = self.ensure_zones(frame)
        if hasattr(self.detector, "detect_frame_with_boxes"):
            observations, detections = self.detector.detect_frame_with_boxes(frame, zones)
        else:
            observations = self.detector.detect_frame(frame, zones)
            detections = []

        state = build_state_from_observations(
            observations,
            self.approach_order,
            queue_norm=self.queue_norm,
            wait_norm=self.wait_norm,
            max_approaches=self.max_approaches,
        )

        # emergency vehicle check
        alerts: list[EmergencyAlert] = []
        override = False
        now = time.time()

        if self.emergency_detector is not None:
            alerts = self.emergency_detector.detect_frame(frame, zones)
            if alerts:
                action_for_alert = self._emergency_action_for(alerts[0].approach)
                if action_for_alert is not None:
                    self._emergency_action = action_for_alert
                    self._emergency_until = now + self.emergency_hold

        if now < self._emergency_until and self._emergency_action is not None:
            action = self._emergency_action
            override = True
        else:
            action = self.agent.select_action(state, greedy=True)

        return Decision(
            timestamp=now,
            observations=list(observations),
            state=state,
            action=int(action),
            detections=detections,
            emergency_alerts=alerts,
            emergency_override=override,
        )


class MultiCameraController:
    """One camera per approach — each camera covers its road arm in full.

    Instead of splitting one frame into zones, each camera's full frame is
    one detection zone. This is more accurate for real deployments.

    Pass an ``emergency_detector`` to enable emergency vehicle preemption.
    """

    def __init__(
        self,
        detector,
        agent,
        approach_order: Sequence[str] = DEFAULT_APPROACHES,
        queue_norm: float = 20.0,
        wait_norm: float = 120.0,
        max_approaches: int = 4,
        emergency_detector=None,
        emergency_hold: float = 30.0,
    ) -> None:
        self.detector = detector
        self.agent = agent
        self.approach_order = list(approach_order)
        self.queue_norm = queue_norm
        self.wait_norm = wait_norm
        self.max_approaches = max_approaches
        self.emergency_detector = emergency_detector
        self.emergency_hold = emergency_hold
        self._emergency_until: float = 0.0
        self._emergency_action: int | None = None

    def _full_frame_zone(self, frame, approach: str) -> DetectionZone:
        h, w = frame.shape[:2]
        return DetectionZone(approach=approach, x1=0, y1=0, x2=w, y2=h)

    def _emergency_action_for(self, approach: str) -> int | None:
        for action, approaches in _ACTION_APPROACHES.items():
            if approach in approaches:
                return action
        return None

    def decide(self, frames: dict[str, np.ndarray]) -> Decision:
        all_observations: list[LaneObservation] = []
        all_detections: list[VehicleDetection] = []
        all_alerts: list[EmergencyAlert] = []

        for approach in self.approach_order:
            frame = frames.get(approach)
            if frame is None:
                all_observations.append(LaneObservation(approach, 0.0, 0.0, 0.0))
                continue
            zone = self._full_frame_zone(frame, approach)
            if hasattr(self.detector, "detect_frame_with_boxes"):
                obs_list, dets = self.detector.detect_frame_with_boxes(frame, [zone])
            else:
                obs_list = self.detector.detect_frame(frame, [zone])
                dets = []
            all_observations.extend(obs_list)
            all_detections.extend(dets)

            if self.emergency_detector is not None:
                all_alerts.extend(self.emergency_detector.detect_frame(frame, [zone]))

        state = build_state_from_observations(
            all_observations,
            self.approach_order,
            queue_norm=self.queue_norm,
            wait_norm=self.wait_norm,
            max_approaches=self.max_approaches,
        )

        override = False
        now = time.time()
        if all_alerts:
            action_for_alert = self._emergency_action_for(all_alerts[0].approach)
            if action_for_alert is not None:
                self._emergency_action = action_for_alert
                self._emergency_until = now + self.emergency_hold

        if now < self._emergency_until and self._emergency_action is not None:
            action = self._emergency_action
            override = True
        else:
            action = self.agent.select_action(state, greedy=True)

        return Decision(
            timestamp=now,
            observations=all_observations,
            state=state,
            action=int(action),
            detections=all_detections,
            emergency_alerts=all_alerts,
            emergency_override=override,
        )


def _annotate_single(frame, approach: str, obs: LaneObservation, detections: list, action: int):
    """Annotate one camera's frame with its bounding boxes and count."""
    import cv2

    out = frame.copy()
    colour = _ZONE_COLOURS.get(approach, _DEFAULT_COLOUR)

    for det in detections:
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), colour, 2)
        cv2.putText(out, f"{det.confidence:.2f}", (det.x1, max(det.y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    cv2.putText(out, f"{approach}: {int(obs.vehicle_count)}", (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return out


def _make_grid(annotated_frames: dict[str, np.ndarray], target_h: int = 360, target_w: int = 640) -> np.ndarray:
    """Combine 4 annotated frames into a 2x2 grid."""
    import cv2

    order = ["north", "east", "south", "west"]
    cells = []
    for approach in order:
        frame = annotated_frames.get(approach)
        if frame is None:
            frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        cells.append(cv2.resize(frame, (target_w, target_h)))

    top = np.hstack([cells[0], cells[1]])
    bot = np.hstack([cells[2], cells[3]])
    return np.vstack([top, bot])


def run_live_multi(
    controller: MultiCameraController,
    sources: dict[str, str | int],
    decision_interval: float = 2.0,
    max_decisions: int = 0,
    display: bool = True,
    on_decision: Optional[Callable[[Decision], None]] = None,
    window_name: str = "AI Traffic Light — 4 Cameras",
) -> list[Decision]:
    """Run the controller with one camera per approach.

    Args:
        sources: mapping of approach name to camera index or video path,
                 e.g. ``{"north": 0, "east": 1, "south": 2, "west": 3}``.
    """
    import cv2

    caps: dict[str, cv2.VideoCapture] = {}
    for approach, src in sources.items():
        cap_src = int(src) if str(src).isdigit() else str(src)
        cap = cv2.VideoCapture(cap_src)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera for approach '{approach}' (source '{src}').")
        caps[approach] = cap

    decisions: list[Decision] = []
    last_decision_time = 0.0
    last_decision: Optional[Decision] = None

    try:
        while True:
            frames: dict[str, np.ndarray] = {}
            for approach, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    frames[approach] = frame

            if not frames:
                break

            now = time.monotonic()
            if now - last_decision_time >= decision_interval:
                last_decision = controller.decide(frames)
                last_decision_time = now
                decisions.append(last_decision)
                if on_decision is not None:
                    on_decision(last_decision)
                if max_decisions and len(decisions) >= max_decisions:
                    break

            if display and last_decision is not None:
                obs_map = {obs.approach: obs for obs in last_decision.observations}
                det_map: dict[str, list] = {a: [] for a in caps}
                for det in (last_decision.detections or []):
                    det_map.setdefault(det.approach, []).append(det)

                annotated: dict[str, np.ndarray] = {}
                for approach, frame in frames.items():
                    obs = obs_map.get(approach, LaneObservation(approach, 0.0))
                    annotated[approach] = _annotate_single(
                        frame, approach, obs, det_map.get(approach, []), last_decision.action
                    )

                grid = _make_grid(annotated)
                phase_label = "E-W GREEN" if last_decision.action == 0 else "N-S GREEN"
                cv2.putText(grid, phase_label, (10, grid.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                cv2.imshow(window_name, grid)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        for cap in caps.values():
            cap.release()
        if display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    return decisions


def load_controller_from_checkpoint(
    checkpoint_path: str,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    approach_order: Sequence[str] = DEFAULT_APPROACHES,
    device: str = "cpu",
) -> LiveController:
    """Build a LiveController from a trained DQN checkpoint and a YOLO detector."""
    import torch

    from .dqn import DQNAgent
    from .vision import YOLOVehicleDetector

    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent = DQNAgent.from_checkpoint(checkpoint, device=device)
    detector = YOLOVehicleDetector(model_name=model_name, confidence=confidence)
    return LiveController(detector=detector, agent=agent, approach_order=approach_order)


_ZONE_COLOURS = {
    "north": (255, 100, 0),
    "east":  (0, 200, 255),
    "south": (0, 255, 100),
    "west":  (200, 0, 255),
}
_DEFAULT_COLOUR = (0, 255, 0)


def draw_overlay(frame, decision: Decision, zones: list[DetectionZone]):
    """Annotate a frame with detection zones, bounding boxes and the phase."""
    import cv2

    annotated = frame.copy()
    counts = {obs.approach: obs.vehicle_count for obs in decision.observations}

    # zone outlines and vehicle counts
    for zone in zones:
        colour = _ZONE_COLOURS.get(zone.approach, _DEFAULT_COLOUR)
        cv2.rectangle(annotated, (zone.x1, zone.y1), (zone.x2, zone.y2), colour, 2)
        label = f"{zone.approach}: {int(counts.get(zone.approach, 0))}"
        cv2.putText(annotated, label, (zone.x1 + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    # per-vehicle bounding boxes
    if decision.detections:
        for det in decision.detections:
            colour = _ZONE_COLOURS.get(det.approach, _DEFAULT_COLOUR)
            cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), colour, 2)
            cv2.putText(
                annotated,
                f"{det.confidence:.2f}",
                (det.x1, max(det.y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
            )

    # emergency vehicle bounding boxes (bright red outline, thicker)
    if decision.emergency_alerts:
        for alert in decision.emergency_alerts:
            cv2.rectangle(annotated, (alert.x1, alert.y1), (alert.x2, alert.y2), (0, 0, 255), 3)
            cv2.putText(annotated, f"EMERGENCY {alert.approach.upper()}",
                        (alert.x1, max(alert.y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # current phase — flash red background if emergency override active
    phase_label = "E-W GREEN" if decision.action == 0 else "N-S GREEN"
    if decision.emergency_override:
        phase_label = "EMERGENCY OVERRIDE — " + phase_label
        text_colour = (0, 0, 255)
    else:
        text_colour = (0, 200, 255)
    cv2.putText(annotated, phase_label, (10, annotated.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_colour, 2)
    return annotated


def run_live(
    controller: LiveController,
    source: str | int = "0",
    decision_interval: float = 2.0,
    max_decisions: int = 0,
    display: bool = True,
    on_decision: Optional[Callable[[Decision], None]] = None,
    window_name: str = "AI Traffic Light",
) -> list[Decision]:
    """Capture a webcam/video feed and run the controller on a wall-clock cadence.

    Args:
        source: webcam index (e.g. ``0`` or ``"0"``) or a path to a video file.
        decision_interval: minimum seconds between phase decisions.
        max_decisions: stop after this many decisions; ``0`` runs until the source
            ends or the user presses ``q`` (display mode only).
        display: show an annotated preview window; set ``False`` for headless
            operation, e.g. on a Raspberry Pi without a monitor.
        on_decision: callback invoked with each :class:`Decision` — the seam where
            a SQLite logger or GPIO bridge hooks in.

    Returns the list of decisions made during the session.
    """
    import cv2

    cap_source = int(source) if str(source).isdigit() else str(source)
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source '{source}'.")

    decisions: list[Decision] = []
    last_decision_time = 0.0
    last_decision: Optional[Decision] = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.monotonic()
            if now - last_decision_time >= decision_interval:
                last_decision = controller.decide(frame)
                last_decision_time = now
                decisions.append(last_decision)
                if on_decision is not None:
                    on_decision(last_decision)
                if max_decisions and len(decisions) >= max_decisions:
                    break

            if display:
                zones = controller.ensure_zones(frame)
                annotated = draw_overlay(frame, last_decision, zones) if last_decision else frame
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    return decisions
