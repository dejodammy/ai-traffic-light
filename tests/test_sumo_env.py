import shutil

import pytest

from ai_traffic_light.config import build_scenario_config
from ai_traffic_light.rl_env import SumoRLEnv


pytestmark = pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO binary is not installed")


def test_sumo_env_smoke_runs_headless():
    env = SumoRLEnv(
        build_scenario_config(
            "ideal",
            use_gui=False,
            episode_steps=20,
            decision_interval=5,
            seed=7,
        )
    )

    try:
        state = env.reset()
        assert state.shape == (12,)
        assert env.action_size >= 2

        next_state, reward, done, info = env.step(0)
        assert next_state.shape == (12,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "total_queue" in info
    finally:
        env.close()
