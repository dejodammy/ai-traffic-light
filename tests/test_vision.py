from ai_traffic_light.vision import LaneObservation, build_state_from_observations


def test_build_state_from_observations_pads_and_orders():
    state = build_state_from_observations(
        [
            LaneObservation("north", vehicle_count=6, waiting_time=15.0, density=0.4),
            LaneObservation("east", vehicle_count=2, waiting_time=5.0, density=0.2),
        ],
        approach_order=["north", "east", "south", "west"],
    )

    assert state.shape == (12,)
    assert state[0] > state[3]
    assert state[1] > 0.0
    assert state[-1] == 0.0
