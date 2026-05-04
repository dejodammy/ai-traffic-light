from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .dqn import DQNAgent, DQNConfig
from .rl_env import EnvConfig, SumoRLEnv

_BEST_CHECKPOINT_MIN_EPISODE = 30


@dataclass(slots=True)
class TrainingConfig:
    episodes: int = 300
    output_dir: str = "results/latest"
    save_every: int = 10
    pretrained_checkpoint: Optional[str] = None


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_dqn(env_config: EnvConfig, train_config: TrainingConfig, dqn_config: DQNConfig | None = None) -> dict:
    _set_seed(env_config.seed)
    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = SumoRLEnv(env_config)
    initial_state = env.reset()
    agent = DQNAgent(initial_state.shape[0], env.action_size, dqn_config)
    env.close()

    if train_config.pretrained_checkpoint is not None:
        ckpt = torch.load(train_config.pretrained_checkpoint, map_location="cpu")
        agent.load_pretrained_weights(ckpt)
        print(f"Loaded pretrained weights from {train_config.pretrained_checkpoint}")

    rewards_history: list[float] = []
    queue_history: list[float] = []
    wait_history: list[float] = []
    loss_history: list[float] = []
    best_rolling_reward = float("-inf")
    best_checkpoint_path = output_dir / "best_dqn_model.pt"

    for episode in range(1, train_config.episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        queue_sum = 0.0
        wait_sum = 0.0
        decisions = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.global_step += 1
            loss = agent.optimize()
            if loss is not None:
                loss_history.append(loss)

            state = next_state
            episode_reward += reward
            queue_sum += info["total_queue"]
            wait_sum += info["total_wait"]
            decisions += 1

        env.close()

        average_queue = queue_sum / max(decisions, 1)
        average_wait = wait_sum / max(decisions, 1)
        rewards_history.append(episode_reward)
        queue_history.append(average_queue)
        wait_history.append(average_wait)

        rolling_rewards = rewards_history[-10:]
        rolling_reward = sum(rolling_rewards) / len(rolling_rewards)

        # Only track best checkpoint after the agent has warmed up
        if episode >= _BEST_CHECKPOINT_MIN_EPISODE and rolling_reward >= best_rolling_reward:
            best_rolling_reward = rolling_reward
            checkpoint = agent.checkpoint()
            checkpoint["env_config"] = asdict(env_config)
            checkpoint["best_rolling_reward"] = best_rolling_reward
            torch.save(checkpoint, best_checkpoint_path)

        print(
            f"episode={episode:03d} "
            f"reward={episode_reward:8.2f} "
            f"avg_queue={average_queue:7.2f} "
            f"avg_wait={average_wait:9.2f} "
            f"epsilon={agent.epsilon():.3f} "
            f"rolling10={rolling_reward:8.2f}"
        )

        if episode % train_config.save_every == 0:
            np.savez(
                output_dir / "training_logs.npz",
                rewards=np.asarray(rewards_history),
                avg_queue=np.asarray(queue_history),
                avg_wait=np.asarray(wait_history),
                losses=np.asarray(loss_history, dtype=np.float32),
            )

    final_checkpoint_path = output_dir / "final_dqn_model.pt"
    final_checkpoint = agent.checkpoint()
    final_checkpoint["env_config"] = asdict(env_config)
    torch.save(final_checkpoint, final_checkpoint_path)
    np.savez(
        output_dir / "training_logs.npz",
        rewards=np.asarray(rewards_history),
        avg_queue=np.asarray(queue_history),
        avg_wait=np.asarray(wait_history),
        losses=np.asarray(loss_history, dtype=np.float32),
    )

    summary = {
        "output_dir": str(output_dir.resolve()),
        "episodes": train_config.episodes,
        "best_model": str(best_checkpoint_path.resolve()),
        "final_model": str(final_checkpoint_path.resolve()),
        "average_reward": float(np.mean(rewards_history)) if rewards_history else 0.0,
        "best_rolling_reward": float(best_rolling_reward),
        "scenario": env_config.name,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
