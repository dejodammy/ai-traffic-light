from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .dqn import DQNAgent
from .rl_env import EnvConfig, SumoRLEnv


class FixedTimeController:
    def __init__(self, action_size: int):
        self.action_size = action_size
        self._next_action = 0

    def select_action(self, _state):
        action = self._next_action
        self._next_action = (self._next_action + 1) % self.action_size
        return action


class LearnedController:
    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def select_action(self, state):
        return self.agent.select_action(state, greedy=True)


def _run_episode(env: SumoRLEnv, controller) -> dict:
    state = env.reset()
    done = False
    total_reward = 0.0
    queue_sum = 0.0
    wait_sum = 0.0
    decisions = 0
    while not done:
        action = controller.select_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        queue_sum += info["total_queue"]
        wait_sum += info["total_wait"]
        decisions += 1

    env.close()
    return {
        "reward": total_reward,
        "avg_queue": queue_sum / max(decisions, 1),
        "avg_wait": wait_sum / max(decisions, 1),
        "decisions": decisions,
    }


def evaluate_checkpoint(checkpoint_path: str, env_config: EnvConfig, episodes: int = 5, output_dir: str | None = None) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    agent = DQNAgent.from_checkpoint(checkpoint, device="cpu")
    env = SumoRLEnv(env_config)

    learned_metrics = [_run_episode(env, LearnedController(agent)) for _ in range(episodes)]
    fixed_time_metrics = [_run_episode(env, FixedTimeController(agent.action_dim)) for _ in range(episodes)]

    def aggregate(rows: list[dict]) -> dict:
        return {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("reward", "avg_queue", "avg_wait", "decisions")
        }

    summary = {
        "scenario": env_config.name,
        "episodes": episodes,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "learned": aggregate(learned_metrics),
        "fixed_time": aggregate(fixed_time_metrics),
    }

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary_path = Path(output_dir) / "evaluation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary
