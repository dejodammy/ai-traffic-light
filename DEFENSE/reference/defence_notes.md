# Project Defence Notes

This document covers the full codebase breakdown, design decisions, expected examiner questions and answers, and result interpretation for the AI Traffic Light Control project.

---

## System Overview

The system uses a Deep Q-Network (DQN) to control a traffic light at a single intersection. Instead of a fixed repeating cycle, the AI observes how many vehicles are queued and waiting at each approach, then chooses which signal phase to set next.

The pipeline is:

```
Camera / SUMO simulator
        |
        v
  vision.py          — detects vehicles, counts per approach lane
        |
        v
  rl_env.py          — turns lane counts into a state vector, applies actions to SUMO
        |
        v
  dqn.py             — neural network that learns which phase minimises queues
        ^
        |
  imitation.py       — pre-trains the DQN by imitating an expert controller
        |
  expert.py          — the rule-based expert it imitates
```

Two training scenarios are used:
- `sumo_scenarios/` — ideal balanced 4-way intersection
- `sumo_scenarios1/` — Lagos intersection (asymmetric arterial road)

Both share the same Python code. Only the `.sumocfg` and `.rou.xml` files differ.

---

## Module Breakdown

### `rl_env.py` — The simulation environment

Wraps SUMO via the TraCI Python API into the standard reinforcement learning loop:
**state → action → reward → next state**

**State vector (12 numbers):**
For each of the 4 approaches, 3 features are read live from SUMO:
- `queue / queue_norm` — number of stopped vehicles, normalised
- `wait / wait_norm` — cumulative waiting time, normalised
- `density / 100` — lane occupancy percentage

**Action space:**
2 discrete actions — action 0 activates phase 0 (one green pair), action 1 activates phase 2 (the other green pair). These are discovered automatically by reading the traffic light logic from SUMO and filtering for green-only phases, so the environment works with any junction.

**Reward function:**
```
reward = -(queue/queue_norm + wait_weight x wait/wait_norm)
```
If the agent switches phase it also subtracts `switch_penalty`. The reward is always negative — the agent is trying to make it as close to zero as possible by keeping queues and waiting times low.

**Yellow phase handling:**
Before switching to a new phase, the environment automatically inserts a 3-second yellow phase if one exists in the traffic light logic. This mirrors real traffic light behaviour.

**Decision interval:**
The agent only acts every 5 simulation seconds. Between decisions, SUMO runs 5 steps and vehicles move. Real controllers do not switch every second.

---

### `dqn.py` — The neural network agent

**`QNet` (the network architecture):**
A 3-layer fully-connected network:
- Input: 12 features (state vector)
- Hidden layer 1: 256 neurons, ReLU
- Hidden layer 2: 256 neurons, ReLU
- Output: 2 Q-values (one per phase)

The two output values are Q-values: the estimated total discounted future reward for taking each action. The agent always picks the action with the higher Q-value.

**Three improvements over vanilla DQN:**

**1. Double DQN**
Vanilla DQN overestimates Q-values because the same network both picks and evaluates the best next action. Double DQN splits the roles: the *online network* selects the action, the *target network* evaluates it.
```python
next_actions = self.online_net(next_states).argmax(...)   # online picks
next_q_values = self.target_net(next_states).gather(...)  # target evaluates
```

**2. Target network**
A frozen copy of the online network, updated every 1,000 steps. Without this, training is unstable because the agent would be chasing a moving target — both sides of the loss function would change every step.

**3. Prioritised Experience Replay (PER)**
Instead of sampling past experiences uniformly, transitions with large TD-errors (surprising or poorly-understood outcomes) are sampled more often. A `SumTree` data structure enables O(log n) weighted sampling. Importance-sampling weights (`beta`) correct for the resulting bias, annealed from 0.4 to 1.0 over training.

**Epsilon-greedy exploration:**
Starts at epsilon=0.2 (warm-start) or 1.0 (from scratch) and decays exponentially to 0.05 over 2,000 steps. This balances exploration early in training with exploitation once a policy is learned.

---

### `expert.py` — The rule-based expert controller

`PhasePressureController` selects the green phase serving the highest traffic pressure.

**How it scores each phase:**
```
score = sum of (halting_vehicles + 0.05 x waiting_time) for every lane in that phase
```
It picks whichever phase has the highest total score. `min_hold_decisions=1` means it can switch as soon as a better phase is found.

This is a greedy pressure-based policy — it does not plan ahead, but it is far better than fixed-time on an asymmetric junction because it responds to actual demand.

---

### `imitation.py` — Expert-guided pre-training

**Two stages:**

**Stage 1 — Demonstration collection:**
The expert controller runs 15 full episodes in SUMO. Every (state, expert_action) pair is saved. This produces ~1,920 labelled samples.

**Stage 2 — Supervised pre-training:**
The DQN's neural network is trained for 400 epochs using cross-entropy loss to predict the expert's action from the state. This is a classification problem — given the 12-feature state, predict which of the 2 phases the expert would choose.

Final accuracy reaches 100%, meaning the network has fully encoded the expert's policy as its starting weights.

The checkpoint is saved and used as the initialisation for either direct evaluation or further RL fine-tuning.

---

### Training stage 2 — RL fine-tuning (`training.py`)

After imitation pre-training, the DQN is further trained using actual reinforcement learning:

1. Load the expert-pretrained weights
2. Start epsilon at 0.2 (already has a good policy — no need for full random exploration)
3. Run 200 RL episodes where the agent acts, receives rewards, and updates via Bellman equation
4. Save the best rolling-10 checkpoint

This stage can exceed the expert because:
- The expert is *greedy* — it picks the best action *right now*
- The DQN with gamma=0.99 *plans ahead* — it learns that sometimes holding a phase slightly longer now prevents a larger queue later

---

### `vision.py` — Real-world vehicle detection

Uses YOLOv8 (nano) to detect vehicles in camera frames. Vehicles are counted per approach lane using detection zones — the frame is divided into 4 equal vertical bands (N, E, S, W). A vehicle is assigned to a zone if its bounding-box centre falls inside that zone.

Vehicle classes detected: car (2), motorbike (3), bus (5), truck (7) — COCO dataset class IDs.

`detect_frame()` takes a numpy array (live webcam frame); `detect()` reads from a file path. Both return the same `list[LaneObservation]` used by `build_state_from_observations()` to produce the 12-feature state vector.

---

### `live.py` — Real-time controller loop

Connects the camera, DQN, and (future) hardware:

```
cv2.VideoCapture → frame → YOLOVehicleDetector.detect_frame()
    → build_state_from_observations() → DQNAgent.select_action(greedy=True)
    → Decision(timestamp, observations, state, action)
    → on_decision callback  ← SQLite logger / GPIO bridge hook
```

The `on_decision` callback is intentionally separate from the capture loop so the logger and hardware bridge plug in without changing the core loop.

---

## Scenarios

### Why two scenarios?

The ideal scenario (`sumo_scenarios/`) has balanced, heavy, uniform demand and shows the DQN can outperform fixed-time under congestion.

The Lagos scenario (`sumo_scenarios1/`) models a real intersection layout from Lagos, Nigeria. Two variants were tested:

**Original (`lagos_intersection`):** all four approaches had identical demand (900 veh/h). Fixed-time is near-optimal when demand is symmetric, so the improvement was only ~4%. This is documented as a finding, not a failure — it demonstrates understanding of *when* adaptive control helps.

**Peak (`lagos_peak`):** realistic asymmetric demand. E-W is a heavy arterial (1200 + 700 surge veh/h); N-S is a light cross street (350 veh/h). Fixed-time wastes equal green time on the empty cross street while the arterial backs up. The DQN exploits the asymmetry and achieves 82% queue reduction and 95% wait reduction.

---

## Results Summary

| Scenario | Controller | Avg Queue | Avg Wait | vs Fixed-time |
|---|---|---:|---:|---|
| Ideal | Original DQN (untrained) | 180.35 | 18,209 | −46% queue / −83% wait |
| Ideal | Expert-guided DQN | 64.16 | 1,046 | — |
| Ideal | Fixed-time | 120.57 | 6,071 | baseline |
| Lagos (symmetric) | Expert-guided DQN | 5.57 | 28.23 | −4% queue / −2% wait |
| Lagos (symmetric) | Fixed-time | 5.81 | 28.85 | baseline |
| Lagos Peak | Expert-guided DQN | 7.42 | 51.85 | −82% queue / −95% wait |
| Lagos Peak | Fixed-time | 40.06 | 1,146 | baseline |
| Lagos Peak | RL-refined DQN | 3.18 | 14.14 | −92% queue / −99% wait |

The comparison GIF (`results/comparison.gif`) shows both controllers on the same traffic simultaneously — the queue divergence during rush hour is visible within 60 seconds of the animation.

---

## Expected Examiner Questions and Answers

### On reinforcement learning

**"What is a Q-value?"**
The Q-value Q(s,a) is the estimated total discounted future reward if you take action a in state s and then follow the optimal policy afterwards. The agent picks the action with the highest Q-value.

**"What is the Bellman equation?"**
The update target is: `Q(s,a) = r + gamma x max_a' Q(s', a')` where r is the immediate reward, gamma=0.99 is the discount factor, and s' is the next state. The network is trained to minimise the squared error between its Q-value prediction and this target.

**"What is experience replay and why do you need it?"**
During training, every (state, action, reward, next_state) transition is stored in a buffer (capacity 100,000). Training samples random mini-batches from this buffer rather than using the most recent transition. This breaks the temporal correlation between consecutive transitions, which would otherwise make gradient updates biased and cause training instability.

**"What is the discount factor gamma and why 0.99?"**
Gamma controls how much the agent values future rewards relative to immediate ones. At gamma=0.99, a reward 100 steps in the future is worth 0.99^100 = 0.37 of an immediate reward. This means the agent plans ahead across roughly 100 decision steps, which covers the full episode. A lower gamma would make it too short-sighted for traffic control.

**"Why DQN and not PPO, A3C, or another algorithm?"**
DQN is appropriate because the action space is small (2 discrete phases) and the state is low-dimensional (12 numbers). Policy-gradient methods like PPO are better suited to continuous or very large action spaces. DQN is also well-understood, reproducible, and easier to explain in a project defence, which matters for an academic submission.

---

### On the network design

**"Why a 3-layer network with 256 neurons?"**
The state is 12 numbers and the action space is 2 choices, so a large network is not needed. 256 neurons per layer provides enough capacity to learn non-linear relationships between queue imbalances and the correct phase, without being so large that it overfits to the limited training data.

**"Why ReLU activations?"**
ReLU (Rectified Linear Unit) is the standard default for deep networks. It avoids the vanishing gradient problem of sigmoid/tanh, is computationally simple, and works well for Q-function approximation.

**"What is Double DQN and why does it matter?"**
In vanilla DQN, the same network both selects the best next action and evaluates its Q-value. This leads to systematically optimistic (overestimated) Q-values, which causes the agent to prefer risky actions. Double DQN uses two separate networks for selection and evaluation, eliminating this bias. In practice it makes training more stable and the final policy more reliable.

---

### On the training method

**"Why did you use imitation learning first?"**
A DQN trained from random exploration on a 900-step episode requires hundreds of episodes to discover a useful policy by trial and error. With only two phases, random exploration occasionally hits the right answer but learns very slowly. The expert pre-training gives the DQN a correct starting policy in ~15 minutes, after which RL refines it further. Without pre-training, 200 RL episodes would mostly be wasted on random behaviour.

**"Is 100% imitation accuracy overfitting?"**
The goal of the imitation stage is not generalisation — it is initialisation. The expert is deterministic and consistent (given a state it always picks the same phase), so 100% accuracy means the network has correctly encoded the expert's rule. The subsequent RL training then moves the weights beyond this starting point through actual environmental feedback.

**"This uses supervised learning — is it really RL?"**
The pre-training step uses supervised learning. But the DQN architecture, Bellman update, replay buffer, and epsilon-greedy exploration are all RL components. The pre-training is a weight-initialisation strategy. The RL stage then genuinely updates the policy through environmental rewards and can improve beyond what the expert knows.

**"Why cross-entropy loss for pre-training?"**
Phase selection is a classification problem — given a 12-feature state, predict which of 2 phases to choose. Cross-entropy is the standard loss for multi-class classification. Mean-squared error could also be used (treating Q-values as regression targets) but cross-entropy converges faster and more reliably for this use case.

---

### On the scenario and results

**"Isn't the 82% improvement just because you tuned the scenario to favour your model?"**
No. Symmetric four-way intersections are actually the unrealistic, idealised case. Real Lagos junctions have a major arterial road and lighter cross streets — they are never symmetric. The original symmetric scenario was the modelling error. The peak scenario corrects this by using demand levels typical of a Lagos arterial (1,200 base + 700 peak veh/h on the main road, 350 veh/h on the cross street). Both results are reported and the difference is explained.

**"Why does the symmetric Lagos scenario only show 4% improvement?"**
When demand is equal on all approaches, a fixed equal-split cycle is already near-optimal. There is almost no slack for an adaptive controller to exploit. This is documented and used as evidence that the system is working correctly — the AI correctly produces near-zero improvement when improvement is genuinely not possible, and large improvement when the junction is asymmetric and saturated.

**"How is fixed-time implemented for the baseline?"**
`FixedTimeController` in `evaluation.py` cycles through actions 0, 1, 0, 1 giving each phase equal time. This matches a conventional fixed-time signal running its default equal-split programme.

**"Why does the reward get worse when queue improves?"**
The reward function includes a switch penalty each time the phase changes. The adaptive DQN switches more frequently than fixed-time, accumulating more penalties. But queue and waiting time — the metrics that actually matter for drivers — improve dramatically. The reward is a training signal, not the final evaluation metric.

---

### On SUMO and the simulation

**"Why SUMO?"**
SUMO (Simulation of Urban MObility) is a free, open-source, microscopic traffic simulator widely used in academic research. Each vehicle is simulated individually with car-following and lane-change models. It produces realistic queue and waiting-time data accessible via TraCI (Traffic Control Interface), a Python API that allows external programs to read sensor data and send control commands each simulation step.

**"What is TraCI?"**
Traffic Control Interface — SUMO's Python API. It allows an external script to read per-lane vehicle counts, waiting times, and occupancy, and to set traffic light phases at each simulation step. The `rl_env.py` module wraps TraCI calls into the standard RL environment interface (reset/step/close).

**"How representative is the SUMO model of real Lagos traffic?"**
SUMO uses simplified car-following models (IDM, Krauss) that approximate real driver behaviour. The demand levels and route structure are set to reflect realistic Lagos arterial road loading based on publicly available data. However, the model does not capture pedestrians, motorcycles weaving between lanes, driver aggression, or random incidents. Results should be interpreted as simulated performance under idealised conditions, not field-measured outcomes.

---

### On the real-world component

**"Does this actually work on real traffic?"**
The YOLOv8 vehicle detection pipeline is tested on real images (see `notebooks/detection_tests.ipynb`) and the live controller loop (`src/ai_traffic_light/live.py`) connects webcam → detection → DQN → phase decision in real time. The GPIO bridge for physical traffic lights is designed but not yet wired to hardware. The system produces correct decisions on live video; translating that to physical signal control requires only the hardware connection.

**"Why YOLOv8?"**
YOLO (You Only Look Once) processes the entire image in a single network forward pass, making it fast enough for live video. YOLOv8n (nano) is the smallest variant, runnable on a Raspberry Pi. It is pre-trained on the COCO dataset which includes all common road vehicle classes (car, motorcycle, bus, truck).

---

## Robustness Results

The RL-refined Lagos Peak checkpoint was tested on five additional scenarios without retraining.

| Scenario | vs Fixed-time Queue | vs Fixed-time Wait | Interpretation |
|---|---|---|---|
| Ideal (different junction) | −20% | −29% | Generalises across junction types |
| Lagos symmetric | +3% (worse) | +12% (worse) | Correct — no asymmetry to exploit |
| Low demand | +6% (worse) | +35% (worse) | Correct — no congestion |
| Extreme asymmetry (E-W 2000) | −61% | −10% | Scales to severe overload |
| Reversed (N-S heavy) | −3% | +27% (worse) | E-W bias — trained direction only |

**Key examiner questions on robustness:**

**"Does your model generalise beyond the training scenario?"**
Yes, partially. It beats fixed-time on the ideal scenario (−20% queue) despite never training there. It correctly produces near-zero improvement on symmetric and low-demand scenarios. It struggles when the heavy direction is N-S rather than E-W — a direct consequence of training on a single asymmetric scenario.

**"What happens when the heavy traffic direction is reversed?"**
Queue reduces slightly (−3%) because the model correctly detects N-S pressure in the state, but waiting time increases (+27%) because it does not hold N-S phases long enough. This shows the model is reading state rather than applying a fixed sequence, but has a learned directional bias. Retraining on multiple demand configurations would resolve this.

**"Does the model add value when there is no congestion?"**
No — and that is the correct behaviour. At low demand, fixed-time is near-optimal and the model's extra switching adds overhead. If the model claimed large improvements under light traffic, that would be suspicious. The small degradation is consistent with the switch penalty in the reward function.

---

## Limitations to state proactively

State these yourself before the examiner asks — it shows maturity.

1. **Single intersection.** Coordinating multiple intersections is a much harder problem not addressed here.
2. **Simulation, not field measurement.** SUMO results are valid within the model's assumptions. Real deployment would require calibration against measured traffic counts.
3. **No continual learning.** The deployed model does not update its weights from live experience. If traffic patterns shift significantly, the model would need retraining.
4. **Zone-based counting is approximate.** A vehicle near a zone boundary could be mis-assigned. Proper deployment needs camera calibration and per-lane detection zones mapped to the physical intersection.
5. **No hardware in the loop yet.** The GPIO bridge and physical lights are the next step.

---

## Two-Stage Training Summary (for report writeup)

The training methodology has two distinct stages:

**Stage 1 — Imitation learning (expert pre-training)**
The `PhasePressureController` expert runs 15 simulation episodes and records every (state, action) pair. The DQN is trained on these ~1,920 samples using cross-entropy loss for 400 epochs until it perfectly reproduces the expert's decisions (100% accuracy, loss ~0).

**Stage 2 — Reinforcement learning (RL fine-tuning)**
Starting from the expert-pretrained weights, standard Double DQN training runs for 200 episodes with epsilon starting at 0.2. The agent interacts directly with the SUMO environment, receives reward signals, and updates its policy via the Bellman equation and prioritised experience replay. This stage can exceed the expert because it optimises for long-term cumulative reward rather than myopic greedy decisions.

The best checkpoint across all RL episodes (tracked by rolling-10 reward) is saved as `results/lagos_peak_rl/best_dqn_model.pt`.
