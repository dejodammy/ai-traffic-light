from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass(slots=True)
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 128
    replay_capacity: int = 100_000
    min_replay_size: int = 1_000
    target_update_every: int = 1_000
    hidden_dim: int = 256          # was 128 — more capacity for complex patterns
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay: int = 2_000     # was 5_000 — calibrated to ~200-episode runs
    grad_clip_norm: float = 10.0
    per_alpha: float = 0.6         # priority exponent for PER
    per_beta_start: float = 0.4    # IS weight start (annealed to 1.0)
    per_beta_steps: int = 10_000   # steps to anneal beta over
    per_eps: float = 1e-6          # minimum priority floor


class SumTree:
    """Binary heap enabling O(log n) priority-proportional sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list = [None] * capacity
        self.size = 0
        self._write = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        idx = self._write + self.capacity - 1
        self.data[self._write] = data
        self._set(idx, priority)
        self._write = (self._write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _set(self, idx: int, priority: float) -> None:
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def update(self, idx: int, priority: float) -> None:
        self._set(idx, priority)

    def get(self, s: float) -> tuple[int, float, object]:
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


class PrioritizedReplayBuffer:
    """Experience replay with priority-proportional sampling and IS correction."""

    def __init__(self, capacity: int, alpha: float, beta_start: float, beta_steps: int, eps: float) -> None:
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.eps = eps
        self.max_priority = 1.0
        self._sample_step = 0

    def push(self, state, action, reward, next_state, done) -> None:
        self.tree.add(self.max_priority ** self.alpha, (state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        self._sample_step += 1
        beta = min(1.0, self.beta_start + self._sample_step * (1.0 - self.beta_start) / max(self.beta_steps, 1))

        segment = self.tree.total / batch_size
        indices, priorities, samples = [], [], []

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            if data is None:
                idx, priority, data = self.tree.get(random.uniform(0, self.tree.total))
            indices.append(idx)
            priorities.append(max(priority, 1e-8))
            samples.append(data)

        probs = np.array(priorities, dtype=np.float64) / max(self.tree.total, 1e-8)
        weights = (self.tree.size * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        states, actions, rewards, next_states, dones = map(np.asarray, zip(*samples))
        return states, actions, rewards, next_states, dones, np.array(indices, dtype=np.int64), weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for idx, p in zip(indices, priorities):
            self.tree.update(int(idx), float(p))
            self.max_priority = max(self.max_priority, float(p))

    def __len__(self) -> int:
        return self.tree.size


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig | None = None, device: str | None = None):
        self.config = config or DQNConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online_net = QNet(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.target_net = QNet(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.config.learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.replay_capacity,
            alpha=self.config.per_alpha,
            beta_start=self.config.per_beta_start,
            beta_steps=self.config.per_beta_steps,
            eps=self.config.per_eps,
        )
        self.global_step = 0

    def epsilon(self, frame_idx: int | None = None) -> float:
        idx = self.global_step if frame_idx is None else frame_idx
        cfg = self.config
        return cfg.epsilon_final + (cfg.epsilon_start - cfg.epsilon_final) * math.exp(-idx / cfg.epsilon_decay)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon():
            return random.randrange(self.action_dim)
        with torch.no_grad():
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(torch.argmax(self.online_net(tensor_state), dim=1).item())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def optimize(self) -> float | None:
        if len(self.replay_buffer) < self.config.min_replay_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.config.batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online_net(states).gather(1, actions)
        with torch.no_grad():
            # Double DQN: online net picks the action, target net evaluates it
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            targets = rewards + self.config.gamma * (1.0 - dones) * next_q_values

        td_errors = (q_values - targets).detach().cpu().numpy().squeeze(1)
        self.replay_buffer.update_priorities(indices, td_errors)

        loss = (weights * (q_values - targets).pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        if self.global_step % self.config.target_update_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def load_pretrained_weights(self, checkpoint: dict) -> None:
        """Copy network weights from a pretrained checkpoint; keeps fresh RL state."""
        self.online_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(checkpoint["model_state_dict"])

    def checkpoint(self) -> dict:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "global_step": self.global_step,
            "dqn_config": asdict(self.config),
            "model_state_dict": self.online_net.state_dict(),
            "target_model_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, device: str | None = None):
        saved_cfg = checkpoint["dqn_config"]
        known_fields = {f.name for f in DQNConfig.__dataclass_fields__.values()}
        filtered = {k: v for k, v in saved_cfg.items() if k in known_fields}
        config = DQNConfig(**filtered)
        agent = cls(
            state_dim=checkpoint["state_dim"],
            action_dim=checkpoint["action_dim"],
            config=config,
            device=device,
        )
        agent.global_step = checkpoint["global_step"]
        agent.online_net.load_state_dict(checkpoint["model_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return agent
