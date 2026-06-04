import numpy as np

from ai_traffic_light.dqn import DQNAgent, DQNConfig
from ai_traffic_light.live import Decision, LiveController
from ai_traffic_light.vision import LaneObservation, MockDetector, default_zones


def test_default_zones_partition_width():
    zones = default_zones(400, 200, ["north", "east", "south", "west"])
    assert len(zones) == 4
    assert zones[0].x1 == 0
    assert zones[-1].x2 == 400
    assert all(zone.y1 == 0 and zone.y2 == 200 for zone in zones)


def test_live_controller_produces_valid_decision():
    detector = MockDetector(
        [
            LaneObservation("north", vehicle_count=8, waiting_time=20.0, density=0.5),
            LaneObservation("east", vehicle_count=1, waiting_time=2.0, density=0.1),
        ]
    )
    agent = DQNAgent(state_dim=12, action_dim=4, config=DQNConfig(epsilon_start=0.0, epsilon_final=0.0))
    controller = LiveController(detector=detector, agent=agent)
    frame = np.zeros((200, 400, 3), dtype=np.uint8)

    decision = controller.decide(frame)

    assert isinstance(decision, Decision)
    assert decision.state.shape == (12,)
    assert 0 <= decision.action < 4
    assert decision.observations[0].approach == "north"
    # zones are auto-built from the first frame's dimensions
    assert controller.zones is not None
    assert len(controller.zones) == 4
