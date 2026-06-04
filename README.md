# AI Traffic Light

This repository now implements the final-year project described in `OLADEJO OLUWADAMILOLA FINAL YEAR PROJECT (1).pdf` as a runnable baseline:

- single-intersection intelligent signal control
- SUMO simulation
- Deep Q-Network training
- fixed-time baseline evaluation
- camera/YOLO-based inference prototype
- automated tests

## PDF goal translated into code

The PDF defines a closed-loop system:

1. detect traffic from a camera
2. turn that into a traffic state
3. let a DQN agent choose the next phase
4. test and train in SUMO
5. keep the final system lightweight enough for Raspberry Pi style inference

The codebase now matches that structure:

- [`src/ai_traffic_light/rl_env.py`](/f:/ai-traffic-light/src/ai_traffic_light/rl_env.py): SUMO traffic-light environment
- [`src/ai_traffic_light/dqn.py`](/f:/ai-traffic-light/src/ai_traffic_light/dqn.py): replay buffer, Q-network, agent
- [`src/ai_traffic_light/training.py`](/f:/ai-traffic-light/src/ai_traffic_light/training.py): DQN training loop
- [`src/ai_traffic_light/evaluation.py`](/f:/ai-traffic-light/src/ai_traffic_light/evaluation.py): learned vs fixed-time comparison
- [`src/ai_traffic_light/vision.py`](/f:/ai-traffic-light/src/ai_traffic_light/vision.py): image detection and state construction
- [`src/main.py`](/f:/ai-traffic-light/src/main.py): CLI entry point

## What the implementation is doing

### State representation

The controller uses a 12-value state vector:

- north: queue, wait, density
- east: queue, wait, density
- south: queue, wait, density
- west: queue, wait, density

In SUMO, those come from:

- halting vehicles
- accumulated waiting time
- lane occupancy

In the vision pipeline, the same layout is built from vehicle detections.

### Action representation

The action is the choice of the next green phase. Instead of hard-coding phase numbers, the environment inspects the SUMO traffic-light logic and discovers valid green phases. That matters because the original prototype used a hard-coded phase mapping that does not generalize safely.

### Reward

The reward is based on congestion reduction:

- positive when queue length decreases
- positive when waiting time decreases
- slightly negative when the agent switches too often

That matches the project objective of reducing congestion and delay.

## Scenario presets

- `ideal`: [`sumo_scenarios/ideal.sumocfg`](/f:/ai-traffic-light/sumo_scenarios/ideal.sumocfg)
- `lagos_intersection`: [`sumo_scenarios1/intersection.sumocfg`](/f:/ai-traffic-light/sumo_scenarios1/intersection.sumocfg)

The Lagos preset is still a single-intersection case study, which matches the project scope in the PDF.

## Setup

1. Activate the virtual environment if you want to reuse the existing one.

```powershell
.\venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Make sure SUMO is installed and `sumo` is on your `PATH`.

## Main commands

Run a smoke test:

```powershell
python src/main.py smoke-test --scenario ideal
```

Train the DQN:

```powershell
python src/main.py train --scenario ideal --episodes 60 --output-dir results/ideal_run
```

Evaluate a checkpoint against fixed-time control:

```powershell
python src/main.py evaluate --scenario ideal --checkpoint results/ideal_run/best_dqn_model.pt
```

Run image-based inference:

```powershell
python src/main.py detect --image sumo_scenarios1/cropped_traffic.png --checkpoint results/ideal_run/best_dqn_model.pt
```

Generate a side-by-side DQN vs fixed-time comparison GIF:

```powershell
python src/main.py compare --checkpoint results/lagos_peak_pretrain/expert_pretrained_dqn_model.pt --scenario lagos_peak --output results/comparison.gif
```

The GIF shows both controllers on the same traffic simultaneously: an intersection diagram with queued car blocks and traffic-light indicators, plus a live total-queue chart underneath. The output is at `results/comparison.gif`.

Run live on a webcam or video feed (real-time control loop):

```powershell
python src/main.py live --checkpoint results/ideal_run/best_dqn_model.pt --source 0 --decision-interval 2
```

`--source` accepts a webcam index (e.g. `0`) or a path to a video file. Use
`--no-display` for headless operation (e.g. on a Raspberry Pi), and
`--max-decisions N` to stop after N decisions. Each decision is printed as the
per-approach vehicle counts and the chosen phase.

## Tests

```powershell
pytest
```

The test suite covers:

- DQN action selection
- replay-buffer training gate behaviour
- state construction from detections
- a real headless SUMO smoke test

## Legacy scripts

The old prototype scripts remain under `sumo_scenarios/`, but the supported workflow is now the CLI in [`src/main.py`](/f:/ai-traffic-light/src/main.py).
