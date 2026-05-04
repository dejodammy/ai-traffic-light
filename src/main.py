from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ai_traffic_light.config import build_scenario_config, list_scenario_presets
from ai_traffic_light.dqn import DQNAgent, DQNConfig
from ai_traffic_light.evaluation import evaluate_checkpoint
from ai_traffic_light.imitation import ImitationConfig, pretrain_from_expert
from ai_traffic_light.rl_env import SumoRLEnv
from ai_traffic_light.training import TrainingConfig, train_dqn
from ai_traffic_light.vision import DetectionZone, YOLOVehicleDetector, build_state_from_observations


def _default_zones(image_path: str) -> list[DetectionZone]:
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image '{image_path}'.")

    height, width = image.shape[:2]
    band = width // 4
    labels = ["north", "east", "south", "west"]
    zones = []
    for idx, label in enumerate(labels):
        x1 = idx * band
        x2 = width if idx == 3 else (idx + 1) * band
        zones.append(DetectionZone(approach=label, x1=x1, y1=0, x2=x2, y2=height))
    return zones


def run_smoke_test(args) -> int:
    env = SumoRLEnv(
        build_scenario_config(
            args.scenario,
            use_gui=args.gui,
            episode_steps=args.steps,
            decision_interval=args.decision_interval,
        )
    )

    try:
        state = env.reset()
        print(f"state_dim={state.shape[0]} action_count={env.action_size} action_phases={env.action_phases}")
        done = False
        total_reward = 0.0
        action = 0
        while not done:
            _, reward, done, info = env.step(action)
            total_reward += reward
            action = (action + 1) % env.action_size
            print(
                f"step={info['step']:03d} phase={info['phase']} "
                f"queue={info['total_queue']:.1f} wait={info['total_wait']:.1f} reward={reward:.2f}"
            )
        print(f"smoke_test_total_reward={total_reward:.2f}")
    finally:
        env.close()

    return 0


def run_train(args) -> int:
    env_config = build_scenario_config(
        args.scenario,
        use_gui=args.gui,
        episode_steps=args.episode_steps,
        decision_interval=args.decision_interval,
        seed=args.seed,
    )
    train_config = TrainingConfig(
        episodes=args.episodes,
        output_dir=args.output_dir,
        save_every=args.save_every,
        pretrained_checkpoint=args.pretrained_checkpoint,
    )
    dqn_config = DQNConfig(
        batch_size=args.batch_size,
        min_replay_size=args.min_replay_size,
        target_update_every=args.target_update_every,
    )
    summary = train_dqn(env_config, train_config, dqn_config)
    print(json.dumps(summary, indent=2))
    return 0


def run_pretrain_expert(args) -> int:
    env_config = build_scenario_config(
        args.scenario,
        use_gui=args.gui,
        episode_steps=args.episode_steps,
        decision_interval=args.decision_interval,
        seed=args.seed,
    )
    imitation_config = ImitationConfig(
        episodes=args.episodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        expert_wait_weight=args.expert_wait_weight,
        expert_min_hold_decisions=args.expert_min_hold_decisions,
        output_dir=args.output_dir,
    )
    dqn_config = DQNConfig(hidden_dim=args.hidden_dim)
    summary = pretrain_from_expert(env_config, imitation_config, dqn_config)
    print(json.dumps(summary, indent=2))
    return 0


def run_evaluate(args) -> int:
    env_config = build_scenario_config(
        args.scenario,
        use_gui=False,
        episode_steps=args.episode_steps,
        decision_interval=args.decision_interval,
    )
    summary = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        env_config=env_config,
        episodes=args.episodes,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


def run_detect(args) -> int:
    zones = _default_zones(args.image)
    detector = YOLOVehicleDetector(model_name=args.model_name, confidence=args.confidence)
    observations = detector.detect(args.image, zones)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    agent = DQNAgent.from_checkpoint(checkpoint, device="cpu")
    state = build_state_from_observations(observations, [zone.approach for zone in zones])
    action = agent.select_action(state, greedy=True)

    print("observations:")
    for observation in observations:
        print(
            f"  {observation.approach}: vehicles={observation.vehicle_count:.0f} "
            f"density={observation.density:.2f}"
        )
    print(f"recommended_action={action}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intelligent traffic light project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    scenario_choices = list_scenario_presets()

    smoke = subparsers.add_parser("smoke-test", help="Run a short headless SUMO smoke test")
    smoke.add_argument("--scenario", choices=scenario_choices, default="ideal")
    smoke.add_argument("--steps", type=int, default=40)
    smoke.add_argument("--decision-interval", type=int, default=5)
    smoke.add_argument("--gui", action="store_true")
    smoke.set_defaults(handler=run_smoke_test)

    train = subparsers.add_parser("train", help="Train the DQN controller")
    train.add_argument("--scenario", choices=scenario_choices, default="ideal")
    train.add_argument("--episodes", type=int, default=60)
    train.add_argument("--episode-steps", type=int, default=600)
    train.add_argument("--decision-interval", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--min-replay-size", type=int, default=1000)
    train.add_argument("--target-update-every", type=int, default=1000)
    train.add_argument("--save-every", type=int, default=10)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--output-dir", default=str(Path("results") / "latest"))
    train.add_argument("--pretrained-checkpoint", default=None, help="Path to a pretrained checkpoint to warm-start RL training")
    train.add_argument("--gui", action="store_true")
    train.set_defaults(handler=run_train)

    pretrain = subparsers.add_parser(
        "pretrain-expert",
        help="Pretrain a DQN controller from a phase-pressure expert policy",
    )
    pretrain.add_argument("--scenario", choices=scenario_choices, default="ideal")
    pretrain.add_argument("--episodes", type=int, default=8)
    pretrain.add_argument("--epochs", type=int, default=250)
    pretrain.add_argument("--episode-steps", type=int, default=600)
    pretrain.add_argument("--decision-interval", type=int, default=10)
    pretrain.add_argument("--batch-size", type=int, default=64)
    pretrain.add_argument("--learning-rate", type=float, default=1e-3)
    pretrain.add_argument("--hidden-dim", type=int, default=256)
    pretrain.add_argument("--expert-wait-weight", type=float, default=0.05)
    pretrain.add_argument("--expert-min-hold-decisions", type=int, default=3)
    pretrain.add_argument("--seed", type=int, default=42)
    pretrain.add_argument("--output-dir", default=str(Path("results") / "expert_pretrain"))
    pretrain.add_argument("--gui", action="store_true")
    pretrain.set_defaults(handler=run_pretrain_expert)

    evaluate = subparsers.add_parser("evaluate", help="Compare a trained agent against fixed-time control")
    evaluate.add_argument("--scenario", choices=scenario_choices, default="ideal")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--episodes", type=int, default=5)
    evaluate.add_argument("--episode-steps", type=int, default=600)
    evaluate.add_argument("--decision-interval", type=int, default=10)
    evaluate.add_argument("--output-dir", default=str(Path("results") / "evaluation"))
    evaluate.set_defaults(handler=run_evaluate)

    detect = subparsers.add_parser("detect", help="Run YOLO on an image and recommend a phase")
    detect.add_argument("--image", required=True)
    detect.add_argument("--checkpoint", required=True)
    detect.add_argument("--model-name", default="yolov8n.pt")
    detect.add_argument("--confidence", type=float, default=0.25)
    detect.set_defaults(handler=run_detect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
