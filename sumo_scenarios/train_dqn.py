from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_traffic_light.config import build_scenario_config
from ai_traffic_light.dqn import DQNConfig
from ai_traffic_light.training import TrainingConfig, train_dqn


def main():
    summary = train_dqn(
        env_config=build_scenario_config(
            "ideal",
            use_gui=False,
            decision_interval=20,
            episode_steps=600,
            seed=42,
        ),
        train_config=TrainingConfig(
            episodes=200,
            output_dir=str(Path(__file__).resolve().parent),
            save_every=10,
        ),
        dqn_config=DQNConfig(
            batch_size=128,
            min_replay_size=2000,
            target_update_every=2000,
        ),
    )
    print(summary)


if __name__ == "__main__":
    main()
