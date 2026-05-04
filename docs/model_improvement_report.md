# Model Improvement Report

## Purpose

This report explains the change made to improve the traffic-light controller and shows the difference between the original DQN model and the improved expert-guided DQN model.

The goal was to make the intelligent traffic-light controller perform better than a regular fixed-time traffic-light system.

## Before the Change

The first model used a normal DQN training process. The agent learned mainly through trial and error in the SUMO simulation.

This approach worked technically, but the model did not learn a strong traffic-control policy within the available number of training episodes. In the first evaluation, the DQN performed worse than the fixed-time baseline.

### Before Change: Ideal Scenario

| Controller | Reward | Average Queue | Average Wait |
|---|---:|---:|---:|
| Original DQN | -13437.70 | 180.35 | 18209.52 |
| Fixed-time traffic light | -6356.80 | 120.57 | 6070.96 |

In this result, the original DQN had higher queue length and higher waiting time than the fixed-time traffic light.

## Change Made

The model was improved by adding expert-guided DQN pretraining.

Instead of allowing the DQN to learn only by random trial and error, a simple adaptive expert controller was created first. This expert controller checks the traffic state and chooses the signal phase that serves the direction with the highest queue and waiting demand.

The expert controller was then used to generate demonstration data. The DQN was trained to imitate these expert decisions before evaluation.

In simple terms:

1. The expert controller observed the traffic situation.
2. It selected a good signal phase based on queue and waiting time.
3. The state and selected action were saved as training examples.
4. The DQN was pretrained using those examples.
5. The improved DQN was evaluated against the fixed-time controller.

The main files added or changed were:

| File | Purpose |
|---|---|
| `src/ai_traffic_light/expert.py` | Defines the adaptive expert controller. |
| `src/ai_traffic_light/imitation.py` | Collects expert demonstrations and pretrains the DQN. |
| `src/main.py` | Adds the `pretrain-expert` command. |

## Training Setup

| Scenario | Demonstration Episodes | DQN Pretraining Epochs | Evaluation Episodes |
|---|---:|---:|---:|
| Ideal scenario | 12 | 350 | 5 |
| Lagos intersection | 12 | 350 | 5 |

## After the Change

### After Change: Ideal Scenario

| Controller | Reward | Average Queue | Average Wait |
|---|---:|---:|---:|
| Improved expert-guided DQN | -859.30 | 64.16 | 1046.82 |
| Fixed-time traffic light | -6356.80 | 120.57 | 6070.96 |

The improved DQN performed better than the fixed-time traffic light in the ideal scenario. It reduced average queue length and average waiting time.

Compared with the original DQN:

| Metric | Original DQN | Improved DQN | Reduction |
|---|---:|---:|---:|
| Average Queue | 180.35 | 64.16 | 64.42% |
| Average Wait | 18209.52 | 1046.82 | 94.25% |

Compared with the fixed-time traffic light:

| Metric | Fixed-time | Improved DQN | Reduction |
|---|---:|---:|---:|
| Average Queue | 120.57 | 64.16 | 46.78% |
| Average Wait | 6070.96 | 1046.82 | 82.76% |

### After Change: Lagos Scenario

| Controller | Reward | Average Queue | Average Wait |
|---|---:|---:|---:|
| Improved expert-guided DQN | -30.30 | 5.57 | 28.23 |
| Fixed-time traffic light | -30.00 | 5.81 | 28.85 |

In the Lagos scenario, the improved DQN achieved slightly lower average queue and waiting time than the fixed-time controller.

| Metric | Fixed-time | Improved DQN | Reduction |
|---|---:|---:|---:|
| Average Queue | 5.81 | 5.57 | 4.03% |
| Average Wait | 28.85 | 28.23 | 2.14% |

The reward was slightly lower because the reward function includes penalties such as switching behavior. However, the direct traffic metrics, queue length and waiting time, improved.

## Result Charts

### Ideal Scenario: Before and After Metrics

![Ideal before and after metrics](../results/report_figures/ideal_before_after_metrics.png)

### Ideal Scenario: Reward Comparison

![Ideal reward comparison](../results/report_figures/ideal_reward_comparison.png)

### Lagos Scenario: Improved DQN vs Fixed-Time

![Lagos after vs fixed-time metrics](../results/report_figures/lagos_after_vs_fixed_metrics.png)

### Percentage Improvement Summary

![Model improvement summary](../results/report_figures/model_improvement_summary.png)

## Commands Used

Generate the improved ideal-scenario model:

```powershell
python src/main.py pretrain-expert --scenario ideal --episodes 12 --epochs 350 --episode-steps 600 --decision-interval 10 --output-dir results\expert_pretrain
```

Evaluate the improved ideal-scenario model:

```powershell
python src/main.py evaluate --scenario ideal --checkpoint results\expert_pretrain\expert_pretrained_dqn_model.pt --episodes 5 --episode-steps 600 --decision-interval 10 --output-dir results\evaluation_expert_pretrain
```

Generate the improved Lagos-scenario model:

```powershell
python src/main.py pretrain-expert --scenario lagos_intersection --episodes 12 --epochs 350 --episode-steps 600 --decision-interval 10 --expert-min-hold-decisions 1 --output-dir results\lagos_expert_pretrain_hold1
```

Evaluate the improved Lagos-scenario model:

```powershell
python src/main.py evaluate --scenario lagos_intersection --checkpoint results\lagos_expert_pretrain_hold1\expert_pretrained_dqn_model.pt --episodes 5 --episode-steps 600 --decision-interval 10 --output-dir results\lagos_evaluation_expert_pretrain_hold1
```

Run automated tests:

```powershell
python -m pytest
```

## Conclusion

The original DQN model did not outperform the fixed-time traffic-light baseline. After expert-guided DQN pretraining was introduced, the improved model performed better on the main traffic metrics.

The ideal scenario showed a strong improvement, while the Lagos scenario showed a smaller but positive reduction in queue length and waiting time.

For the project report, the improvement should be described as an expert-guided DQN approach rather than a DQN trained only from scratch.
