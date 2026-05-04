from __future__ import annotations

from dataclasses import dataclass, field

import traci

from .rl_env import SumoRLEnv


@dataclass(slots=True)
class PhasePressureController:
    """Selects the green phase serving the largest queued/waiting demand."""

    wait_weight: float = 0.05
    min_hold_decisions: int = 3
    _last_action: int | None = field(default=None, init=False)
    _held_for: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._last_action: int | None = None
        self._held_for = 0

    def _phase_score(self, env: SumoRLEnv, phase_index: int) -> float:
        logic = traci.trafficlight.getAllProgramLogics(env.cfg.tls_id)[0]
        links = traci.trafficlight.getControlledLinks(env.cfg.tls_id)
        phase = logic.phases[phase_index]

        score = 0.0
        for link_index, signal_state in enumerate(phase.state):
            if signal_state not in {"G", "g"}:
                continue
            if link_index >= len(links) or not links[link_index]:
                continue

            lane_id = links[link_index][0][0]
            score += traci.lane.getLastStepHaltingNumber(lane_id)
            score += self.wait_weight * traci.lane.getWaitingTime(lane_id)
        return score

    def select_action(self, env: SumoRLEnv, _state) -> int:
        scores = [
            self._phase_score(env, phase_index)
            for phase_index in env.action_phases
        ]
        candidate = max(range(len(scores)), key=scores.__getitem__) if scores else 0

        if self._last_action is None:
            self._last_action = candidate
            self._held_for = 1
            return candidate

        if self._held_for < self.min_hold_decisions:
            self._held_for += 1
            return self._last_action

        if candidate != self._last_action:
            self._last_action = candidate
            self._held_for = 1
            return candidate

        self._held_for += 1
        return self._last_action
