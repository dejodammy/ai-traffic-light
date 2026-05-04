from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .dqn import DQNAgent, DQNConfig
from .expert import PhasePressureController
from .rl_env import EnvConfig, SumoRLEnv


@dataclass(slots=True)
class ImitationConfig:
    episodes: int = 8
    epochs: int = 250
    batch_size: int = 64
    learning_rate: float = 1e-3
    expert_wait_weight: float = 0.05
    expert_min_hold_decisions: int = 3
    output_dir: str = "results/expert_pretrain"


def collect_expert_demonstrations(
    env_config: EnvConfig,
    imitation_config: ImitationConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    queues: list[float] = []
    waits: list[float] = []

    for _ in range(imitation_config.episodes):
        env = SumoRLEnv(env_config)
        controller = PhasePressureController(
            wait_weight=imitation_config.expert_wait_weight,
            min_hold_decisions=imitation_config.expert_min_hold_decisions,
        )
        state = env.reset()
        done = False
        episode_reward = 0.0
        queue_sum = 0.0
        wait_sum = 0.0
        decisions = 0

        while not done:
            action = controller.select_action(env, state)
            states.append(state.copy())
            actions.append(action)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            queue_sum += info["total_queue"]
            wait_sum += info["total_wait"]
            decisions += 1

        env.close()
        rewards.append(episode_reward)
        queues.append(queue_sum / max(decisions, 1))
        waits.append(wait_sum / max(decisions, 1))

    metadata = {
        "demonstration_episodes": imitation_config.episodes,
        "expert_wait_weight": imitation_config.expert_wait_weight,
        "expert_min_hold_decisions": imitation_config.expert_min_hold_decisions,
        "samples": len(states),
        "expert_average_reward": float(np.mean(rewards)) if rewards else 0.0,
        "expert_average_queue": float(np.mean(queues)) if queues else 0.0,
        "expert_average_wait": float(np.mean(waits)) if waits else 0.0,
    }
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.int64), metadata


def pretrain_from_expert(
    env_config: EnvConfig,
    imitation_config: ImitationConfig,
    dqn_config: DQNConfig | None = None,
) -> dict:
    output_dir = Path(imitation_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    states, actions, metadata = collect_expert_demonstrations(env_config, imitation_config)
    if len(states) == 0:
        raise RuntimeError("No expert demonstrations were collected.")

    probe_env = SumoRLEnv(env_config)
    initial_state = probe_env.reset()
    action_size = probe_env.action_size
    probe_env.close()

    dqn_config = dqn_config or DQNConfig()
    agent = DQNAgent(initial_state.shape[0], action_size, dqn_config)
    agent.optimizer = torch.optim.Adam(agent.online_net.parameters(), lr=imitation_config.learning_rate)

    dataset = TensorDataset(
        torch.as_tensor(states, dtype=torch.float32),
        torch.as_tensor(actions, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=imitation_config.batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    loss_history: list[float] = []
    accuracy_history: list[float] = []

    for _ in range(imitation_config.epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_states, batch_actions in loader:
            batch_states = batch_states.to(agent.device)
            batch_actions = batch_actions.to(agent.device)
            logits = agent.online_net(batch_states)
            loss = loss_fn(logits, batch_actions)

            agent.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.online_net.parameters(), dqn_config.grad_clip_norm)
            agent.optimizer.step()

            total_loss += float(loss.item()) * batch_actions.numel()
            correct += int((logits.argmax(dim=1) == batch_actions).sum().item())
            total += batch_actions.numel()

        loss_history.append(total_loss / max(total, 1))
        accuracy_history.append(correct / max(total, 1))

    agent.target_net.load_state_dict(agent.online_net.state_dict())
    checkpoint = agent.checkpoint()
    checkpoint["env_config"] = asdict(env_config)
    checkpoint["training_method"] = "expert_imitation_pretraining"
    checkpoint_path = output_dir / "expert_pretrained_dqn_model.pt"
    torch.save(checkpoint, checkpoint_path)

    np.savez(
        output_dir / "imitation_logs.npz",
        loss=np.asarray(loss_history, dtype=np.float32),
        accuracy=np.asarray(accuracy_history, dtype=np.float32),
        actions=actions,
    )

    summary = {
        "output_dir": str(output_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "scenario": env_config.name,
        "epochs": imitation_config.epochs,
        "final_loss": float(loss_history[-1]),
        "final_accuracy": float(accuracy_history[-1]),
        **metadata,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
