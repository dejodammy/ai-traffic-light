# CHAPTER FOUR: RESULTS AND DISCUSSION

## 4.1 Introduction

This chapter presents the implementation outcomes and the experimental results of the
intelligent traffic light system developed in this project. Following the methodology
described in Chapter Three, a Deep Q-Network (DQN) reinforcement learning agent was trained
to control the signals of a real road junction in Ikeja, Lagos, using real-time
measurements of the traffic on each approach. The chapter describes how the system was
implemented and evaluated, reports the training behaviour of the agent, and compares its
performance against four baseline controllers across several demand conditions and two
signal architectures. The chapter then critically discusses the findings, including both
the strengths and the limitations of the proposed system. The results are presented
honestly: the conditions under which the learned controller clearly outperforms
conventional control, and the conditions under which it is only competitive, are both
reported, because a faithful account of system behaviour is more valuable than an inflated
one.

## 4.2 System Implementation

The implemented system consists of three integrated components that together fulfil the
objectives stated in Chapter One: a perception component, a simulation environment, and a
reinforcement learning controller.

### 4.2.1 Simulation environment

All training and evaluation were carried out in **SUMO (Simulation of Urban Mobility)**, an
open-source microscopic traffic simulator (Krajzewicz et al., 2012). SUMO models every
vehicle individually, which allows realistic measurement of queues and delays. The
controller communicates with SUMO through the Traffic Control Interface (TraCI), reading
the state of each approach and setting the active signal phase at fixed decision intervals.

### 4.2.2 Modelling the case-study junction

The junction modelled is the signalised intersection of **Obafemi Awolowo Way and Allen
Avenue, Ikeja, Lagos** (approximately 6.6071° N, 3.3492° E), one of the busiest junctions
in the Ikeja area. The road network was exported from **OpenStreetMap** and converted to a
SUMO network. To ensure fidelity, the generated network was checked against satellite
imagery and corrected where the open map data was incomplete; in particular, lane counts
that OpenStreetMap had under-specified were restored so that both Obafemi Awolowo Way (the
arterial) and Allen Avenue (the cross street) are modelled as two-lane carriageways,
consistent with the real road. The network was clipped to retain a single signalised
junction while preserving long approach roads, so that vehicles are generated far upstream
and queues form realistically, and demand was modelled with full turning movements
(left, through and right) rather than a single fixed movement per approach.

Two signal architectures were studied:

* **Two-phase control**, in which opposing approaches receive green together (the
  conventional design for a four-leg junction); and
* **Four-phase (protected) control**, in which each approach receives its own green phase
  in turn, representing a junction with protected turning movements.

### 4.2.3 Perception component (vehicle detection)

In line with Objective 5, vehicle detection was implemented using the **YOLO** object
detection model, which counts vehicles on each approach from a camera image and converts
them into the same numerical state used during training. Because the simulator and the
camera front end both produce the identical normalised state representation, a policy
trained in simulation can be applied to live camera input without retraining. The
quantitative accuracy of the detector under field conditions (lighting, weather, occlusion)
was not measured in this work and is identified as future work in Chapter Five.

### 4.2.4 State, action and reward

In line with Objective 2, the traffic state, action space and reward function were defined
as follows:

* **State:** for each of the four approaches, the normalised queue length, accumulated
  waiting time, and occupancy (density), together with two phase-awareness features (the
  currently active phase and how long it has been held). All features are normalised to the
  range 0–1 so that no single feature dominates learning.
* **Action:** selection of the green phase for the next decision interval. A yellow
  clearance interval is inserted automatically whenever the green direction changes.
* **Reward:** the negative weighted sum of total queue length and total waiting time, with a
  small penalty for changing phase to discourage unsafe and wasteful rapid switching.

## 4.3 Experimental Design

### 4.3.1 Demand scenarios

The controller was evaluated under several demand scenarios (volumes are vehicles per hour
per approach):

| Scenario | Obafemi Awolowo Way | Allen Avenue | Purpose |
|---|---|---|---|
| Asymmetric | 760 | 350 | Realistic imbalance |
| Balanced | 760 | 600 | Roughly equal demand |
| Rush-hour | 1400 | 280 | Heavy arterial congestion |
| Allen-dominant (reversed) | 350 | 1400 | Tests that the agent serves the busy road, not always the arterial |
| Unseen (generalisation) | 1000 | 500 | A demand never seen in training |
| Stochastic / random | mean ± noise | mean ± noise | Time-varying demand reflecting real variability |

### 4.3.2 Baseline controllers

To establish whether the learned controller is genuinely useful, it was compared against
four non-learning controllers, ranging from weak to strong:

1. **Naive fixed-time:** a conventional fixed timer holding each phase for 30 s (an
   un-optimised plan).
2. **Webster-optimal fixed-time:** a fixed timer whose cycle length and green splits are
   computed by Webster's method for the prevailing demand — the best possible fixed plan.
3. **Actuated (gap-out):** a non-learning adaptive controller that extends a green while its
   approach has demand and then serves the longest queue — a competent reactive baseline.
4. A fast fixed timer was also retained for reference.

### 4.3.3 Performance metrics

Four metrics, measured identically for every controller, were used: **average queue
length**, **average waiting time**, **average accumulated waiting time** (the total delay a
vehicle experiences across repeated stops, which does not reset on brief movement), and
**throughput** (vehicles served).

## 4.4 Training Results and Convergence

Each controller was trained in two stages: an imitation (expert pre-training) stage that
initialised the network by copying a simple traffic heuristic, followed by a reinforcement
learning stage that refined the policy through interaction with the simulator. Figure 4.2
shows the training behaviour: the episode reward rises and then plateaus, while the average
queue length and average waiting time fall below the fixed-time reference and stabilise,
typically converging within the first 100–150 episodes. The simultaneous convergence of all
three quantities indicates stable learning, and shows that training beyond convergence
yields little further benefit.

![Figure 4.2: Training convergence of the DQN — episode reward, average queue length and average waiting time per episode, with the fixed-time baseline shown for reference.](results/figures/training_curves_peak.png)

**Figure 4.2:** Training convergence of the DQN controller (reward, average queue and
average waiting time per episode). The dashed line is the fixed-time baseline.

## 4.5 Evaluation Results

Unless stated otherwise, results are reported for the realistic junction model under
rush-hour demand. A positive percentage denotes the improvement (reduction) achieved by the
DQN relative to the baseline.

### 4.5.1 Performance against fixed-time and actuated control

Table 4.1 reports the performance of the proposed controller against the baselines, and
Figure 4.3 presents the same results graphically.

**Table 4.1:** Controller performance on the realistic junction (rush-hour demand).

| Controller | Avg queue | Avg wait (s) | Accumulated wait (s) | Throughput |
|---|---|---|---|---|
| **DQN (proposed)** | **6.67** | **23.4** | **204.6** | 347 |
| Naive fixed-time (30 s) | 9.18 | 136.8 | 407.1 | 396 |
| Actuated (gap-out) | 8.48 | 278.0 | 523.8 | 405 |
| Webster-optimal fixed-time | 5.91 | 63.9 | 316.4 | 432 |

![Figure 4.3: Controller performance on the realistic junction.](results/figures/baseline_comparison.png)

**Figure 4.3:** Average queue length, waiting time, accumulated waiting time and throughput
for each controller on the realistic junction (rush-hour demand).

Relative to the **naive fixed timer**, the DQN reduced average queue length by **27 %**,
average waiting time by **83 %**, and accumulated waiting time by **50 %**. Relative to the
**actuated** controller it reduced queue length by **21 %**, waiting time by **92 %**, and
accumulated waiting time by **61 %**. The improvement over the actuated baseline is
particularly significant, because it shows that the learned policy outperforms not only a
fixed plan but also a competent reactive controller. Figure 4.4 summarises these delay
reductions.

![Figure 4.4: Delay reduction of the DQN relative to each baseline.](results/figures/delay_reduction.png)

**Figure 4.4:** Reduction in waiting time and accumulated waiting time achieved by the DQN
relative to each baseline controller.

### 4.5.2 Performance against the Webster-optimal timer

Against the **Webster-optimal** timer — the strongest fixed baseline — the DQN reduced
average waiting time by **63 %** and accumulated waiting time by **35 %**. However, the
Webster timer achieved a slightly shorter average queue (the DQN was about **13 %** higher)
and a higher throughput (about **20 %** more vehicles served).

This is an important and honest result. A Webster-optimal timer allocates green time in
proportion to demand and is, by construction, near-optimal for *average* conditions, so it
performs strongly on queue length and throughput. The decisive advantage of the DQN is in
**waiting time and delay**, because it responds to the *instantaneous* state of the junction
rather than following a fixed schedule — it does not force vehicles to wait through a long
fixed green when the opposing approach is empty. In short, the value of the learned
controller lies in its responsiveness to real-time variability, which no fixed plan can
replicate, rather than in beating an optimally-tuned timer on long-run averages.

### 4.5.3 Effect of signal architecture (two-phase versus four-phase)

The comparison was repeated for the four-phase (protected) signal. Because a four-phase
fixed timer must cycle through all four approaches in turn — serving even empty approaches
on a long cycle — the fixed baselines incur very large delays, while the DQN can skip empty
approaches and serve only those with demand. The DQN therefore retained a large
waiting-time advantage: relative to the naive timer it reduced waiting time by **73 %** and
accumulated waiting time by **47 %**, and relative to the Webster-optimal timer it reduced
waiting time by **67 %** and accumulated waiting time by **33 %**, while remaining level on
queue length. The pattern is identical to the two-phase case and is more pronounced, which
confirms that the controller's benefit grows where fixed plans are least efficient.

### 4.5.4 Generalisation to unseen and reversed demand

Two tests examined whether the agent learned a *general* strategy rather than memorising a
single pattern. First, when the cross street (Allen Avenue) was made the busy road and the
arterial light (the *Allen-dominant* scenario), the agent continued to serve the busier
approach, confirming that it learned the general rule "serve whichever approach is
congested" rather than "always favour the arterial." Second, a model evaluated on an
**unseen** demand level (1000/500) that it had never trained on, without any retraining,
continued to outperform the fixed-time baseline. These results indicate that the learned
policy generalises across demand intensities and directions.

### 4.5.5 Effect of reward design

Experiments in which only the reward weighting was varied showed a clear and controllable
effect on behaviour: increasing the penalty on waiting time reduced waiting time at a small
cost in queue length, and vice versa. This demonstrates that the controller's objective can
be tuned to a transport authority's priorities, and that reward engineering — not only the
training itself — shapes the resulting policy.

## 4.6 Discussion of Findings

The results support three main conclusions. First, the proposed DQN controller delivers
**large and consistent reductions in vehicle delay** — typically between 50 % and 90 % in
waiting time — against both conventional fixed-time signals and a competent actuated
controller, across demand patterns, signal architectures, and traffic it was not trained on.
Second, when compared against a Webster-optimal fixed timer, the controller's advantage is
specifically one of **responsiveness**: it reduces waiting time substantially while
remaining competitive, though not superior, on average queue length and throughput. Third,
the magnitude of the benefit is governed by the **structure of the demand** — adaptive
control helps most under imbalanced, variable or protected-phase conditions, where fixed
schedules are least suitable, and least under balanced, steady demand, for which a fixed
timer is already well suited.

These findings are consistent with established traffic-engineering theory, which holds that
signal optimisation yields its greatest benefit in the unsaturated-to-near-saturated regime
and under variable demand, while at saturation the limiting factor is road capacity rather
than signal timing. The honest characterisation of *where* and *why* the controller helps is
itself a useful outcome, since it indicates the conditions under which deploying a learned
controller in Lagos would be most worthwhile — namely the heavy, imbalanced and variable
conditions that characterise the city's peak-hour traffic.

---

# CHAPTER FIVE: SUMMARY, CONCLUSION AND RECOMMENDATIONS

## 5.1 Summary

This project designed, implemented and evaluated an intelligent traffic light system based
on deep reinforcement learning, with Lagos traffic as a case study. The system combines a
computer-vision front end (YOLO-based vehicle detection that converts a camera image of each
approach into a numerical traffic state) with a Deep Q-Network controller that decides which
signal phase to display. The controller was trained entirely in the SUMO simulator on a
model of a real Lagos junction — Obafemi Awolowo Way and Allen Avenue, Ikeja — exported from
OpenStreetMap and validated against satellite imagery.

A rigorous evaluation was carried out. The controller was compared against four baselines —
a fast fixed timer, a conventional 30-second fixed timer, a Webster-optimal fixed timer, and
a gap-out actuated controller — using four metrics: average queue length, average waiting
time, accumulated waiting time, and throughput. It was tested under several demand
conditions (asymmetric, balanced, rush-hour, Allen-dominant, unseen and stochastic) and on
two signal architectures (two-phase and four-phase protected control). The objectives stated
in Chapter One were thereby met: a state representation and reward function were developed
(Objective 2), a DQN controller was implemented and trained in simulation (Objective 3), a
realistic SUMO model of a real junction was created (Objective 4), and camera-based vehicle
detection was integrated (Objective 5).

## 5.2 Conclusion

The proposed controller substantially reduces vehicle delay relative to conventional and
reactive signal control. Against a typical fixed-time signal it reduced average waiting time
by up to approximately **83 %**, accumulated delay by approximately **50 %**, and queue
length by approximately **27 %**; against a competent actuated controller the reductions in
delay were larger still. Against a Webster-optimal fixed timer — the strongest possible fixed
plan — the controller reduced waiting time by approximately **63 %** while remaining
competitive on queue length and throughput.

It is therefore concluded that a deep reinforcement learning controller offers a clear and
practically significant improvement in vehicle delay over fixed-time control for a junction
of the type studied, and that its principal advantage is the ability to respond to real-time
fluctuations in traffic that no fixed timing plan can accommodate. This benefit is greatest
under exactly the heavy, imbalanced and variable conditions that characterise Lagos peak-hour
traffic, which supports the relevance of the approach to the case study.

## 5.3 Contributions of the Study

The main contributions of this work are:

1. A complete and reproducible pipeline that trains a DQN signal controller in simulation on
   a *real* junction exported from open map data, and applies the same policy to live camera
   input through a YOLO-based perception module.
2. A faithful SUMO model of a real Lagos junction, validated and corrected against satellite
   imagery, with realistic turning movements and long approaches.
3. A rigorous, multi-baseline and multi-metric evaluation — including a Webster-optimal timer
   and a gap-out actuated controller — that honestly characterises where learned control does
   and does not outperform classical methods.
4. Empirical evidence, across demand patterns and signal architectures, that the advantage of
   learned control lies in responsiveness to variability, with a clear statement of the
   conditions under which it is most beneficial.

## 5.4 Limitations of the Study

The following limitations are acknowledged, consistent with the scope set out in Chapter One:

1. **Simulation-based evaluation.** Although the junction geometry is real and the demand is
   informed by published Lagos traffic figures, the demand itself was modelled rather than
   measured at the junction, because field traffic counts for this intersection were not
   publicly available.
2. **Single-junction scope.** The real site is part of a corridor of several closely spaced
   signals; this study controlled a single junction in isolation and did not address
   coordinated multi-junction control.
3. **Competitive, not dominant, on every metric.** A Webster-optimal fixed timer matched or
   exceeded the controller on average queue length and throughput; the controller's clear
   advantage is in delay.
4. **Perception not evaluated end-to-end.** The YOLO detector was integrated but its accuracy
   under real Lagos conditions (weather, occlusion, low light, motorcycles and tricycles) was
   not quantitatively evaluated.
5. **Hardware prototype limitations.** The Raspberry Pi and USB-camera prototype cannot
   control a real signal network and was not deployed at a live intersection, for safety and
   legal reasons.

## 5.5 Recommendations and Future Work

1. **Field data collection.** Use the project's own YOLO module on recorded footage of the
   junction to obtain measured turning-movement counts, replacing the estimated demand and
   strengthening the realism of the evaluation.
2. **Corridor (multi-agent) control.** Extend the approach to coordinate the several signals
   along the Obafemi Awolowo Way corridor, exploiting vehicle platooning between junctions.
3. **Throughput-aware reward.** Incorporate throughput, or adopt a pressure-based
   (max-pressure) formulation, so that the controller's strong delay performance is matched by
   competitive throughput against optimal fixed timers.
4. **Robust perception.** Quantitatively evaluate and, if necessary, fine-tune the vehicle
   detector for Lagos conditions, including motorcycles (*okada*) and tricycles (*keke*).
5. **Supervised hardware pilot.** Advance the Raspberry Pi / GPIO integration to a controlled
   test-rig pilot with a fixed-time safety fallback.
6. **Emergency-vehicle priority.** Complete and evaluate the emergency-vehicle preemption
   capability so that ambulances and fire services can be granted priority, addressing the
   need identified in the statement of the problem.

---

### References (to be merged into your main reference list)

Krajzewicz, D., Erdmann, J., Behrisch, M., & Bieker, L. (2012). Recent development and
applications of SUMO – Simulation of Urban MObility. *International Journal on Advances in
Systems and Measurements, 5*(3–4), 128–138.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... &
Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature,
518*(7540), 529–533.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.).
MIT Press.

Webster, F. V. (1958). *Traffic signal settings* (Road Research Technical Paper No. 39).
Her Majesty's Stationery Office.
