"""Watch a trained DQN model drive the traffic lights in the SUMO GUI.

Usage (from the src/ folder):
    ..\\venv\\Scripts\\python.exe watch_model.py --scenario lagos_peak \
        --checkpoint ..\\results\\lagos_peak_rl\\best_dqn_model.pt
"""
from __future__ import annotations

import argparse

import torch

from ai_traffic_light.config import build_scenario_config, list_scenario_presets
from ai_traffic_light.dqn import DQNAgent
from ai_traffic_light.evaluation import FixedTimeController
from ai_traffic_light.rl_env import SumoRLEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch the trained model in the SUMO GUI")
    parser.add_argument("--scenario", choices=list_scenario_presets(), default="lagos_peak")
    parser.add_argument("--checkpoint", default="../results/lagos_peak_rl/best_dqn_model.pt")
    parser.add_argument("--episode-steps", type=int, default=900)
    parser.add_argument("--decision-interval", type=int, default=5)
    parser.add_argument("--delay", type=int, default=250,
                        help="Milliseconds SUMO waits between frames (higher = slower/easier to watch)")
    parser.add_argument("--fixed", action="store_true",
                        help="Use a dumb fixed-time controller instead of the trained model")
    args = parser.parse_args()

    # use_gui=True opens the SUMO graphical window
    env_config = build_scenario_config(
        args.scenario,
        use_gui=True,
        episode_steps=args.episode_steps,
        decision_interval=args.decision_interval,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    agent = DQNAgent.from_checkpoint(checkpoint, device="cpu")

    env = SumoRLEnv(env_config)
    # Slow the GUI down so it is watchable: append SUMO's --delay flag to the launch command.
    _orig_cmd = env._sumo_cmd
    env._sumo_cmd = lambda: _orig_cmd() + ["--delay", str(args.delay)]
    state = env.reset()

    if args.fixed:
        controller = FixedTimeController(env.action_size)
        mode = "FIXED-TIME (dumb timer)"
    else:
        controller = None  # use the trained agent directly
        mode = "TRAINED MODEL"
    print(f">>> Running scenario '{args.scenario}' with: {mode}\n")

    done = False
    total_queue = 0.0
    steps = 0
    try:
        while not done:
            if args.fixed:
                action = controller.select_action(state)
            else:
                action = agent.select_action(state, greedy=True)  # greedy = use what it learned, no random
            state, reward, done, info = env.step(action)
            total_queue += info["total_queue"]
            steps += 1
            print(
                f"step={info['step']:04d} phase={info['phase']} "
                f"queue={info['total_queue']:.1f} reward={reward:.2f}"
            )
    finally:
        env.close()

    print(f"\nAverage total queue while the model was driving: {total_queue / max(steps, 1):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
