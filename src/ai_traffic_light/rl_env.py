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
    phase_features: bool = False  # add current-phase + time-in-phase to the state
    use_acc_wait: bool = False    # penalise ACCUMULATED waiting time (anti-starvation)
    acc_wait_norm: float = 600.0
    use_pressure: bool = False    # add per-approach downstream queue (max-pressure signal)
    use_pressure_reward: bool = False  # PressLight-style reward: minimise intersection pressure
    pressure_norm: float = 20.0
    dynamic_duration: bool = False     # PDLight-style: agent chooses phase AND green duration
    duration_options: tuple = (5, 10, 20)  # green durations (sim steps) the agent can choose


class SumoRLEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._step = 0
        self._approach_lanes: dict[str, list[str]] = {}
        self._approach_ids: list[str] = []
        self._prev_action: int | None = None
        self._action_phases: tuple[int, ...] = ()
        self._phase_count = 0
        self._current_action = 0
        self._time_in_phase = 0
        self._arrived = 0

    @property
    def action_size(self) -> int:
        base = len(self._action_phases)
        if self.cfg.dynamic_duration and base:
            return base * len(self.cfg.duration_options)
        return base

    def _decode_action(self, action: int) -> tuple[int, int]:
        """Return (green-phase index, green duration in sim steps) for an action."""
        if self.cfg.dynamic_duration:
            n = len(self.cfg.duration_options)
            return action // n, int(self.cfg.duration_options[action % n])
        return action, self.cfg.decision_interval

    @property
    def state_size(self) -> int:
        # max_approaches × (queue, wait, density), +1 downstream-queue (pressure) per approach,
        # + 2 phase-awareness features, each if enabled
        size = self.cfg.max_approaches * 3
        if self.cfg.use_pressure:
            size += self.cfg.max_approaches
        if self.cfg.phase_features:
            size += 2
        return size

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

        # map each approach to its downstream (outgoing) lanes for max-pressure features
        self._approach_downstream: dict[str, list[str]] = {}
        for link_list in traci.trafficlight.getControlledLinks(self.cfg.tls_id):
            if not link_list or not link_list[0]:
                continue
            in_lane, out_lane = link_list[0][0], link_list[0][1]
            self._approach_downstream.setdefault(self._approach_name(in_lane), set()).add(out_lane)
        self._approach_downstream = {a: list(s) for a, s in self._approach_downstream.items()}

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

    def _approach_acc_wait(self, approach: str) -> float:
        """Accumulated waiting time of the vehicles currently on this approach's lanes."""
        total = 0.0
        for lane in self._approach_lanes.get(approach, []):
            for veh in traci.lane.getLastStepVehicleIDs(lane):
                total += traci.vehicle.getAccumulatedWaitingTime(veh)
        return total

    def _approach_downstream_queue(self, approach: str) -> float:
        """Queue waiting on this approach's DOWNSTREAM lanes (max-pressure signal: a full
        downstream means serving this approach won't help)."""
        lanes = self._approach_downstream.get(approach, [])
        if not lanes:
            return 0.0
        return float(sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes))

    def _get_state(self) -> np.ndarray:
        approach_dim = self.cfg.max_approaches * 3
        features: list[float] = []
        for approach in self._approach_ids:
            queue, wait, density = self._approach_metrics(approach)
            # When optimising accumulated wait, the state must EXPOSE accumulated wait so the
            # agent can observe (and learn to relieve) a starved approach.
            if self.cfg.use_acc_wait:
                wait_feat = self._approach_acc_wait(approach) / max(self.cfg.acc_wait_norm, 1.0)
            else:
                wait_feat = wait / max(self.cfg.wait_norm, 1.0)
            features.extend([queue / max(self.cfg.queue_norm, 1.0), wait_feat, density / 100.0])

        while len(features) < approach_dim:
            features.extend([0.0, 0.0, 0.0])
        features = features[:approach_dim]

        if self.cfg.use_pressure:
            # downstream queue per approach (lets the agent learn pressure = upstream - downstream)
            press = [self._approach_downstream_queue(a) / max(self.cfg.queue_norm, 1.0)
                     for a in self._approach_ids]
            while len(press) < self.cfg.max_approaches:
                press.append(0.0)
            features.extend(press[: self.cfg.max_approaches])

        if self.cfg.phase_features:
            # which phase is green now, and how long it has been held
            phase_feat = self._current_action / max(len(self._action_phases) - 1, 1)
            time_feat = min(self._time_in_phase / 10.0, 1.0)
            features.extend([phase_feat, time_feat])

        return np.asarray(features, dtype=np.float32)

    def _total_queue(self) -> float:
        return float(sum(self._approach_metrics(approach)[0] for approach in self._approach_ids))

    def _total_wait(self) -> float:
        return float(sum(self._approach_metrics(approach)[1] for approach in self._approach_ids))

    def _total_accumulated_wait(self) -> float:
        """Sum each vehicle's accumulated waiting time (survives stop-and-go, does not
        reset when the car briefly moves) across all controlled approach lanes."""
        total = 0.0
        for approach in self._approach_ids:
            for lane in self._approach_lanes.get(approach, []):
                for veh in traci.lane.getLastStepVehicleIDs(lane):
                    total += traci.vehicle.getAccumulatedWaitingTime(veh)
        return total

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
                self._arrived += traci.simulation.getArrivedNumber()

    def reset(self) -> np.ndarray:
        if traci.isLoaded():
            traci.close()

        traci.start(self._sumo_cmd())
        self._step = 0
        self._init_lanes()
        self._action_phases = self._discover_action_phases()
        self._prev_action = None
        self._current_action = 0
        self._time_in_phase = 0
        self._arrived = 0
        return self._get_state()

    def step(self, action: int):
        if not self._action_phases:
            raise RuntimeError("Environment must be reset before calling step().")
        if action < 0 or action >= self.action_size:
            raise AssertionError(f"Action must be in [0, {self.action_size - 1}].")

        phase_idx, duration = self._decode_action(action)
        target_phase = self._action_phases[phase_idx]
        current_phase = traci.trafficlight.getPhase(self.cfg.tls_id)
        switched = self._prev_action is not None and phase_idx != self._prev_action

        self._run_yellow_if_needed(current_phase, target_phase)
        traci.trafficlight.setPhase(self.cfg.tls_id, target_phase)

        for _ in range(duration):
            traci.simulationStep()
            self._step += 1
            self._arrived += traci.simulation.getArrivedNumber()

        total_queue = self._total_queue()
        total_wait = self._total_wait()

        if self.cfg.use_pressure_reward:
            # PressLight (Wei et al., 2019): minimise intersection pressure = sum over approaches
            # of (incoming queue - downstream queue). Provably throughput-optimal. A small
            # accumulated-wait term is kept for fairness (anti-starvation).
            pressure = sum(self._approach_metrics(a)[0] - self._approach_downstream_queue(a)
                           for a in self._approach_ids)
            acc = self._total_accumulated_wait() / max(self.cfg.acc_wait_norm, 1.0)
            reward = -(abs(pressure) / max(self.cfg.pressure_norm, 1.0) + 0.3 * acc)
        else:
            if self.cfg.use_acc_wait:
                # penalise accumulated waiting time so the agent cannot starve an approach
                wait_term = self._total_accumulated_wait() / max(self.cfg.acc_wait_norm, 1.0)
            else:
                wait_term = total_wait / max(self.cfg.wait_norm, 1.0)
            reward = -(total_queue / max(self.cfg.queue_norm, 1.0) + self.cfg.wait_weight * wait_term)
        if switched:
            reward -= self.cfg.switch_penalty

        # track phase-awareness features for the next state
        if switched or self._prev_action is None:
            self._time_in_phase = 0
        else:
            self._time_in_phase += 1
        self._current_action = phase_idx   # store PHASE index (not the raw action) for state/switching
        self._prev_action = phase_idx

        state = self._get_state()
        done = self._step >= self.cfg.episode_steps
        info = {
            "step": self._step,
            "phase": traci.trafficlight.getPhase(self.cfg.tls_id),
            "target_phase": target_phase,
            "total_queue": total_queue,
            "total_wait": total_wait,
            "total_acc_wait": self._total_accumulated_wait(),
            "arrived_total": self._arrived,
            "switched": switched,
            "action_phases": self._action_phases,
        }
        return state, float(reward), done, info

    @property
    def approach_ids(self) -> list[str]:
        return list(self._approach_ids)

    def approach_queue_map(self) -> dict[str, float]:
        return {a: self._approach_metrics(a)[0] for a in self._approach_ids}

    def served_approaches(self, action: int) -> list[str]:
        """Return the approach ids that get a green signal for this action."""
        phase_idx, _ = self._decode_action(action)
        if not self._action_phases or phase_idx >= len(self._action_phases):
            return []
        phase_index = self._action_phases[phase_idx]
        logic = traci.trafficlight.getAllProgramLogics(self.cfg.tls_id)[0]
        links = traci.trafficlight.getControlledLinks(self.cfg.tls_id)
        phase = logic.phases[phase_index]
        served: set[str] = set()
        for link_index, signal_state in enumerate(phase.state):
            if signal_state not in {"G", "g"}:
                continue
            if link_index >= len(links) or not links[link_index]:
                continue
            lane_id = links[link_index][0][0]
            served.add(self._approach_name(lane_id))
        return [a for a in self._approach_ids if a in served]

    def emergency_approach(self) -> str | None:
        """Return the approach id of the nearest emergency vehicle, or None.

        Scans all approach lanes for vehicles with vClass='emergency'. The first
        approach found (in approach_id order) is returned so the caller can
        preempt the signal for that direction.
        """
        for approach in self._approach_ids:
            for lane in self._approach_lanes.get(approach, []):
                for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                    if traci.vehicle.getVehicleClass(veh_id) == "emergency":
                        return approach
        return None

    def close(self) -> None:
        if traci.isLoaded():
            traci.close()
            
