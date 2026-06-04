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
from ai_traffic_light.vision import YOLOVehicleDetector, build_state_from_observations, default_zones


def _default_zones(image_path: str):
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image '{image_path}'.")

    height, width = image.shape[:2]
    return default_zones(width, height)


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
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
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


def run_stream_test(args) -> int:
    import cv2

    urls = [u.strip() for u in args.urls.split(",") if u.strip()]
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if len(labels) < len(urls):
        labels += [f"stream_{i}" for i in range(len(labels), len(urls))]

    print(f"Testing {len(urls)} stream(s)...\n")
    caps = {}
    for label, url in zip(labels, urls):
        src = int(url) if url.isdigit() else url
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"  FAIL  {label:10s}  {url}")
        else:
            ret, _ = cap.read()
            status = "OK  " if ret else "OPEN but no frame"
            print(f"  {status}  {label:10s}  {url}")
        caps[label] = (cap, url)

    if args.preview:
        print("\nShowing live preview for each stream — press 'q' to close all")
        while True:
            for label, (cap, url) in caps.items():
                ret, frame = cap.read()
                if ret:
                    cv2.imshow(label, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    for cap, _ in caps.values():
        cap.release()
    return 0


def run_detect_live(args) -> int:
    import cv2
    from ultralytics import YOLO

    VEHICLE_CLASSES = {2, 3, 5, 7}
    model = YOLO(args.model_name)

    cap_src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"ERROR: could not open camera source '{args.source}'")
        return 1

    print("Running live detection — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=args.confidence, verbose=False)
        annotated = frame.copy()

        for box in results[0].boxes:
            cls_id   = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            conf     = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            if args.vehicles_only and cls_id not in VEHICLE_CLASSES:
                continue

            colour = (0, 255, 0) if cls_id in VEHICLE_CLASSES else (180, 180, 180)
            label  = f"{cls_name} {conf:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            # filled label background for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ly = max(y1 - 6, th + 4)
            cv2.rectangle(annotated, (x1, ly - th - 4), (x1 + tw + 4, ly + 2), colour, -1)
            cv2.putText(annotated, label, (x1 + 2, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        cv2.imshow("Live Detection — q to quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def run_detect_test(args) -> int:
    import cv2
    from pathlib import Path
    from ultralytics import YOLO

    model = YOLO(args.model_name)
    image_paths = args.images

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: could not read '{image_path}'")
            continue

        results = model.predict(image, conf=args.confidence, verbose=False)
        annotated = image.copy()

        print(f"\n--- {Path(image_path).name} ---")
        found = []
        for box in results[0].boxes:
            cls_id   = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            conf     = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            is_vehicle = cls_id in {2, 3, 5, 7}
            colour = (0, 255, 0) if is_vehicle else (180, 180, 180)
            label  = f"{cls_name} {conf:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

            tag = " <-- vehicle counted" if is_vehicle else ""
            print(f"  [{cls_id:2d}] {cls_name:<20s}  conf={conf:.2f}{tag}")
            found.append(cls_name)

        if not found:
            print("  (nothing detected)")

        # save annotated image next to the original
        out_path = str(Path(image_path).with_stem(Path(image_path).stem + "_detected"))
        cv2.imwrite(out_path, annotated)
        print(f"  saved: {out_path}")

        if args.show:
            cv2.imshow(Path(image_path).name, annotated)
            print("  press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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


def run_live_multi(args) -> int:
    from ai_traffic_light.live import MultiCameraController, run_live_multi as run_live_multi_loop
    from ai_traffic_light.dqn import DQNAgent
    from ai_traffic_light.vision import YOLOVehicleDetector
    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    agent = DQNAgent.from_checkpoint(checkpoint, device="cpu")
    detector = YOLOVehicleDetector(model_name=args.model_name, confidence=args.confidence)

    approaches = [a.strip() for a in args.approaches.split(",") if a.strip()]
    raw_sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    if len(raw_sources) != len(approaches):
        raise ValueError(
            f"--sources must have the same number of entries as --approaches "
            f"({len(approaches)} approaches, {len(raw_sources)} sources)."
        )
    sources = dict(zip(approaches, raw_sources))

    controller = MultiCameraController(
        detector=detector,
        agent=agent,
        approach_order=approaches,
    )

    if args.emergency_model:
        from ai_traffic_light.vision import EmergencyVehicleDetector
        controller.emergency_detector = EmergencyVehicleDetector(args.emergency_model)
    elif args.emergency_color:
        from ai_traffic_light.vision import ColorFlashDetector
        controller.emergency_detector = ColorFlashDetector()

    callbacks = []
    if args.gpio:
        from ai_traffic_light.gpio_bridge import TrafficLightGPIO
        gpio = TrafficLightGPIO(yellow_duration=args.yellow_duration)
        callbacks.append(gpio)
    if args.log_db:
        from ai_traffic_light.logger import DecisionLogger
        db_logger = DecisionLogger(args.log_db)
        callbacks.append(db_logger)

    def on_decision(decision) -> None:
        counts = ", ".join(f"{obs.approach}={int(obs.vehicle_count)}" for obs in decision.observations)
        print(f"[{decision.timestamp:.0f}] {counts} -> phase {decision.action}")
        for cb in callbacks:
            cb(decision)

    try:
        decisions = run_live_multi_loop(
            controller,
            sources=sources,
            decision_interval=args.decision_interval,
            max_decisions=args.max_decisions,
            display=not args.no_display,
            on_decision=on_decision,
        )
    finally:
        for cb in callbacks:
            if hasattr(cb, "cleanup"):
                cb.cleanup()
            if hasattr(cb, "close"):
                cb.close()

    print(f"live_session_decisions={len(decisions)}")
    return 0


def run_sim_server(args) -> int:
    import asyncio
    import torch
    from ai_traffic_light.dqn import DQNAgent
    from ai_traffic_light.sim_server import run_server

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    agent = DQNAgent.from_checkpoint(checkpoint, device="cpu")

    spawn_rates = {
        "north": args.spawn_ns,
        "south": args.spawn_ns,
        "east":  args.spawn_ew,
        "west":  args.spawn_ew,
    }

    asyncio.run(run_server(
        agent=agent,
        spawn_rates=spawn_rates,
        http_port=args.http_port,
        ws_port=args.ws_port,
    ))
    return 0


def run_compare(args) -> int:
    import json
    from ai_traffic_light.config import build_scenario_config
    from ai_traffic_light.visualize import make_comparison

    env_config = build_scenario_config(
        args.scenario,
        use_gui=False,
        episode_steps=args.episode_steps,
        decision_interval=args.decision_interval,
        seed=args.seed,
    )
    result = make_comparison(
        checkpoint_path=args.checkpoint,
        env_config=env_config,
        output_path=args.output,
        fps=args.fps,
    )
    print(json.dumps(result, indent=2))
    return 0


def run_live(args) -> int:
    from ai_traffic_light.live import load_controller_from_checkpoint, run_live as run_live_loop

    approaches = [a.strip() for a in args.approaches.split(",") if a.strip()]
    controller = load_controller_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        confidence=args.confidence,
        approach_order=approaches,
    )

    if args.emergency_model:
        from ai_traffic_light.vision import EmergencyVehicleDetector
        controller.emergency_detector = EmergencyVehicleDetector(args.emergency_model)
    elif args.emergency_color:
        from ai_traffic_light.vision import ColorFlashDetector
        controller.emergency_detector = ColorFlashDetector()

    callbacks = []

    if args.gpio:
        from ai_traffic_light.gpio_bridge import TrafficLightGPIO
        gpio = TrafficLightGPIO(yellow_duration=args.yellow_duration)
        callbacks.append(gpio)

    if args.log_db:
        from ai_traffic_light.logger import DecisionLogger
        db_logger = DecisionLogger(args.log_db)
        callbacks.append(db_logger)

    def on_decision(decision) -> None:
        counts = ", ".join(f"{obs.approach}={int(obs.vehicle_count)}" for obs in decision.observations)
        print(f"[{decision.timestamp:.0f}] {counts} -> phase {decision.action}")
        for cb in callbacks:
            cb(decision)

    try:
        decisions = run_live_loop(
            controller,
            source=args.source,
            decision_interval=args.decision_interval,
            max_decisions=args.max_decisions,
            display=not args.no_display,
            on_decision=on_decision,
        )
    finally:
        for cb in callbacks:
            if hasattr(cb, "cleanup"):
                cb.cleanup()
            if hasattr(cb, "close"):
                cb.close()

    print(f"live_session_decisions={len(decisions)}")
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
    train.add_argument("--epsilon-start", type=float, default=1.0, help="Starting epsilon (use ~0.2 when warm-starting from a pretrained checkpoint)")
    train.add_argument("--epsilon-final", type=float, default=0.05, help="Final epsilon after decay")
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

    stream_test = subparsers.add_parser("stream-test", help="Test and preview IP camera streams (e.g. iPads over WiFi)")
    stream_test.add_argument("--urls", required=True, help="Comma-separated stream URLs or camera indices")
    stream_test.add_argument("--labels", default="north,east,south,west", help="Comma-separated label for each stream")
    stream_test.add_argument("--preview", action="store_true", help="Show a live preview window for each stream")
    stream_test.set_defaults(handler=run_stream_test)

    detect_live = subparsers.add_parser("detect-live", help="Live camera feed with YOLO labels on every detected object")
    detect_live.add_argument("--source", default="0", help="Webcam index or video file path")
    detect_live.add_argument("--model-name", default="yolov8n.pt")
    detect_live.add_argument("--confidence", type=float, default=0.25)
    detect_live.add_argument("--vehicles-only", action="store_true", help="Only show vehicle detections, hide everything else")
    detect_live.set_defaults(handler=run_detect_live)

    detect_test = subparsers.add_parser("detect-test", help="Run YOLO on one or more images and show what each vehicle was classified as")
    detect_test.add_argument("images", nargs="+", help="Image file paths to test")
    detect_test.add_argument("--model-name", default="yolov8n.pt")
    detect_test.add_argument("--confidence", type=float, default=0.25)
    detect_test.add_argument("--show", action="store_true", help="Display annotated image in a window")
    detect_test.set_defaults(handler=run_detect_test)

    detect = subparsers.add_parser("detect", help="Run YOLO on an image and recommend a phase")
    detect.add_argument("--image", required=True)
    detect.add_argument("--checkpoint", required=True)
    detect.add_argument("--model-name", default="yolov8n.pt")
    detect.add_argument("--confidence", type=float, default=0.25)
    detect.set_defaults(handler=run_detect)

    live = subparsers.add_parser("live", help="Run the controller live on a webcam or video feed")
    live.add_argument("--checkpoint", required=True)
    live.add_argument("--source", default="0", help="Webcam index (e.g. 0) or path to a video file")
    live.add_argument("--decision-interval", type=float, default=2.0, help="Seconds between phase decisions")
    live.add_argument(
        "--max-decisions",
        type=int,
        default=0,
        help="Stop after N decisions (0 = run until the source ends or 'q' is pressed)",
    )
    live.add_argument("--model-name", default="yolov8n.pt")
    live.add_argument("--confidence", type=float, default=0.25)
    live.add_argument("--approaches", default="north,east,south,west")
    live.add_argument("--no-display", action="store_true", help="Headless mode, e.g. on a Raspberry Pi")
    live.add_argument("--emergency-model", default=None, help="Path to a fine-tuned YOLOv8 emergency vehicle model (.pt)")
    live.add_argument("--emergency-color", action="store_true", help="Use colour-flash detection for emergency vehicles (no model needed)")
    live.add_argument("--gpio", action="store_true", help="Enable Raspberry Pi GPIO output to physical traffic lights")
    live.add_argument("--yellow-duration", type=float, default=3.0, help="Seconds to hold yellow before switching phase")
    live.add_argument("--log-db", default=None, help="Path to SQLite database for logging decisions (e.g. results/session.db)")
    live.set_defaults(handler=run_live)

    live_multi = subparsers.add_parser("live-multi", help="Run the controller with one USB camera per approach")
    live_multi.add_argument("--checkpoint", required=True)
    live_multi.add_argument("--sources", default="0,1,2,3", help="Comma-separated camera indices or video paths, one per approach")
    live_multi.add_argument("--approaches", default="north,east,south,west")
    live_multi.add_argument("--decision-interval", type=float, default=2.0)
    live_multi.add_argument("--max-decisions", type=int, default=0)
    live_multi.add_argument("--model-name", default="yolov8n.pt")
    live_multi.add_argument("--confidence", type=float, default=0.25)
    live_multi.add_argument("--no-display", action="store_true")
    live_multi.add_argument("--emergency-model", default=None, help="Path to fine-tuned YOLOv8 emergency vehicle model (.pt)")
    live_multi.add_argument("--emergency-color", action="store_true", help="Use colour-flash detection for emergency vehicles")
    live_multi.add_argument("--gpio", action="store_true")
    live_multi.add_argument("--yellow-duration", type=float, default=3.0)
    live_multi.add_argument("--log-db", default=None)
    live_multi.set_defaults(handler=run_live_multi)

    sim = subparsers.add_parser("sim-server", help="Run the multi-screen intersection simulation server")
    sim.add_argument("--checkpoint", required=True, help="Path to trained DQN checkpoint")
    sim.add_argument("--spawn-ew", type=float, default=0.5, help="Vehicle spawn rate on E-W approaches (0-1 per tick)")
    sim.add_argument("--spawn-ns", type=float, default=0.2, help="Vehicle spawn rate on N-S approaches (0-1 per tick)")
    sim.add_argument("--http-port", type=int, default=8765, help="HTTP port for serving the browser client")
    sim.add_argument("--ws-port",   type=int, default=8766, help="WebSocket port for state broadcast")
    sim.set_defaults(handler=run_sim_server)

    compare = subparsers.add_parser("compare", help="Generate a DQN vs fixed-time side-by-side GIF")
    compare.add_argument("--checkpoint", required=True)
    compare.add_argument("--scenario", choices=scenario_choices, default="lagos_peak")
    compare.add_argument("--output", default=str(Path("results") / "comparison.gif"))
    compare.add_argument("--fps", type=int, default=10)
    compare.add_argument("--episode-steps", type=int, default=900)
    compare.add_argument("--decision-interval", type=int, default=5)
    compare.add_argument("--seed", type=int, default=42)
    compare.set_defaults(handler=run_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
