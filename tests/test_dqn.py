import numpy as np

from ai_traffic_light.dqn import DQNAgent, DQNConfig


def test_agent_selects_valid_action():
    agent = DQNAgent(state_dim=12, action_dim=4, config=DQNConfig(epsilon_start=0.0, epsilon_final=0.0))
    state = np.zeros(12, dtype=np.float32)
    action = agent.select_action(state, greedy=True)
    assert 0 <= action < 4


def test_replay_buffer_gates_optimization():
    agent = DQNAgent(state_dim=12, action_dim=2, config=DQNConfig(min_replay_size=4, batch_size=4))
    sample_state = np.ones(12, dtype=np.float32)
    for _ in range(3):
        agent.remember(sample_state, 0, 1.0, sample_state, False)
    assert agent.optimize() is None
