from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import traci


@dataclass(slots=True)
class EnvConfig:
    name: str = "ideal"
    sumocfg: str = "sumo_scenarios/ideal.sumocfg"
    tls_id: str = "c"
    decision_interval: int = 10
    episode_steps: int = 600
    use_gui: bool = False
    seed: Optional[int] = None
    action_phases: tuple[int, ...] | None = None
    use_yellow: bool = True
    yellow_duration: int = 3
    queue_weight: float = 1.0
    wait_weight: float = 0.1
    switch_penalty: float = 0.05
    queue_norm: float = 20.0
    wait_norm: float = 500.0
    max_approaches: int = 4


class SumoRLEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._step = 0
        self._approach_lanes: dict[str, list[str]] = {}
        self._approach_ids: list[str] = []
        self._prev_action: int | None = None
        self._action_phases: tuple[int, ...] = ()
        self._phase_count = 0

    @property
    def action_size(self) -> int:
        return len(self._action_phases)

    @property
    def state_size(self) -> int:
        return self.cfg.max_approaches * 3

    @property
    def action_phases(self) -> tuple[int, ...]:
        return self._action_phases

    def _sumo_cmd(self) -> list[str]:
        binary = "sumo-gui" if self.cfg.use_gui else "sumo"
        sumocfg = str(Path(self.cfg.sumocfg).resolve())
        cmd = [binary, "-c", sumocfg, "--start", "--quit-on-end"]
        if self.cfg.seed is not None:
            cmd += ["--seed", str(self.cfg.seed)]
        return cmd

    @staticmethod
    def _approach_name(lane_id: str) -> str:
        if "_" not in lane_id:
            return lane_id
        return lane_id.rsplit("_", 1)[0]

    def _init_lanes(self) -> None:
        lanes = traci.trafficlight.getControlledLanes(self.cfg.tls_id)
        unique_lanes = list(dict.fromkeys(lanes))
        groups: dict[str, list[str]] = {}
        for lane in unique_lanes:
            groups.setdefault(self._approach_name(lane), []).append(lane)
        self._approach_lanes = groups
        self._approach_ids = sorted(groups)[: self.cfg.max_approaches]

    @staticmethod
    def _is_green_phase(state: str) -> bool:
        return "y" not in state and any(ch in {"G", "g"} for ch in state)

    @staticmethod
    def _is_yellow_phase(state: str) -> bool:
        return "y" in state

    def _discover_action_phases(self) -> tuple[int, ...]:
        logic = traci.trafficlight.getAllProgramLogics(self.cfg.tls_id)[0]
        self._phase_count = len(logic.phases)
        if self.cfg.action_phases:
            return self.cfg.action_phases

        green_phases = []
        for idx, phase in enumerate(logic.phases):
            if self._is_green_phase(phase.state):
                green_phases.append(idx)

        if not green_phases:
            raise RuntimeError(f"No controllable green phases discovered for TLS '{self.cfg.tls_id}'.")

        return tuple(green_phases)

    def _approach_metrics(self, approach: str) -> tuple[float, float, float]:
        lanes = self._approach_lanes[approach]
        queue = float(sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes))
        wait = float(sum(traci.lane.getWaitingTime(lane) for lane in lanes))
        density = float(np.mean([traci.lane.getLastStepOccupancy(lane) for lane in lanes]))
        return queue, wait, density

    def _get_state(self) -> np.ndarray:
        features: list[float] = []
        for approach in self._approach_ids:
            queue, wait, density = self._approach_metrics(approach)
            features.extend(
                [
                    queue / max(self.cfg.queue_norm, 1.0),
                    wait / max(self.cfg.wait_norm, 1.0),
                    density / 100.0,
                ]
            )

        while len(features) < self.state_size:
            features.extend([0.0, 0.0, 0.0])

        return np.asarray(features[: self.state_size], dtype=np.float32)

    def _total_queue(self) -> float:
        return float(sum(self._approach_metrics(approach)[0] for approach in self._approach_ids))

    def _total_wait(self) -> float:
        return float(sum(self._approach_metrics(approach)[1] for approach in self._approach_ids))

    def _run_yellow_if_needed(self, current_phase: int, target_phase: int) -> None:
        if not self.cfg.use_yellow or current_phase == target_phase:
            return

        next_phase = (current_phase + 1) % max(self._phase_count, 1)
        logic = traci.trafficlight.getAllProgramLogics(self.cfg.tls_id)[0]
        if next_phase < len(logic.phases) and self._is_yellow_phase(logic.phases[next_phase].state):
            traci.trafficlight.setPhase(self.cfg.tls_id, next_phase)
            for _ in range(self.cfg.yellow_duration):
                traci.simulationStep()
                self._step += 1

    def reset(self) -> np.ndarray:
        if traci.isLoaded():
            traci.close()

        traci.start(self._sumo_cmd())
        self._step = 0
        self._init_lanes()
        self._action_phases = self._discover_action_phases()
        self._prev_action = None
        return self._get_state()

    def step(self, action: int):
        if not self._action_phases:
            raise RuntimeError("Environment must be reset before calling step().")
        if action < 0 or action >= self.action_size:
            raise AssertionError(f"Action must be in [0, {self.action_size - 1}].")

        target_phase = self._action_phases[action]
        current_phase = traci.trafficlight.getPhase(self.cfg.tls_id)
        switched = self._prev_action is not None and action != self._prev_action

        self._run_yellow_if_needed(current_phase, target_phase)
        traci.trafficlight.setPhase(self.cfg.tls_id, target_phase)

        for _ in range(self.cfg.decision_interval):
            traci.simulationStep()
            self._step += 1

        total_queue = self._total_queue()
        total_wait = self._total_wait()

        reward = -(total_queue / max(self.cfg.queue_norm, 1.0) +
                   self.cfg.wait_weight * total_wait / max(self.cfg.wait_norm, 1.0))
        if switched:
            reward -= self.cfg.switch_penalty

        self._prev_action = action

        state = self._get_state()
        done = self._step >= self.cfg.episode_steps
        info = {
            "step": self._step,
            "phase": traci.trafficlight.getPhase(self.cfg.tls_id),
            "target_phase": target_phase,
            "total_queue": total_queue,
            "total_wait": total_wait,
            "switched": switched,
            "action_phases": self._action_phases,
        }
        return state, float(reward), done, info

    def close(self) -> None:
        if traci.isLoaded():
            traci.close()
