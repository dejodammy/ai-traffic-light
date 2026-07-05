"""Full evaluation matrix: all 8 realistic scenarios x {DQN, naive, Webster, actuated},
reporting QUEUE, ACCUMULATED WAIT and THROUGHPUT (no resetting wait). Writes markdown."""
from __future__ import annotations
import torch
from ai_traffic_light.config import build_scenario_config
from ai_traffic_light.dqn import DQNAgent
from ai_traffic_light.rl_env import SumoRLEnv
from ai_traffic_light.evaluation import FixedTimeController, LearnedController
from baseline_compare import (HoldFixedController, WebsterFixedController, ActuatedController,
                              webster_green, _run_episode_acc)

EPISODES = 2
ARTERIAL = {"610011092#2", "134404792#3"}

SCEN = [  # scenario, run dir, arterial vph, cross vph, signal, demand label
    ("lagos_big_peak_aw",     "lagos_big_peak_aw_rl",     1400, 280, "2-phase", "Rush-hour"),
    ("lagos_big_asym_aw",     "lagos_big_asym_aw_rl",      760, 350, "2-phase", "Asymmetric"),
    ("lagos_big_bal_aw",      "lagos_big_bal_aw_rl",       760, 600, "2-phase", "Balanced"),
    ("lagos_big_reversed_aw", "lagos_big_reversed_aw_rl",  350,1400, "2-phase", "Allen-dominant"),
    ("lagos_big4_peak_aw",    "lagos_big4_peak_aw_rl",    1400, 280, "4-phase", "Rush-hour"),
    ("lagos_big4_asym_aw",    "lagos_big4_asym_aw_rl",     760, 350, "4-phase", "Asymmetric"),
    ("lagos_big4_bal_aw",     "lagos_big4_bal_aw_rl",      760, 600, "4-phase", "Balanced"),
    ("lagos_big4_reversed_aw","lagos_big4_reversed_aw_rl", 350,1400, "4-phase", "Allen-dominant"),
]


def avg(rows, k):
    return sum(r[k] for r in rows) / len(rows)


def run(cfg, make_ctrl):
    env = SumoRLEnv(cfg)
    return [_run_episode_acc(env, make_ctrl) for _ in range(EPISODES)]


def pct(base, dqn):
    return (base - dqn) / base * 100 if base else 0.0


lines = ["# Full Evaluation Matrix (realistic junction)\n",
         "Metrics: average queue length, accumulated waiting time (s), throughput (vehicles served). "
         "Positive % = reduction achieved by the DQN versus that baseline.\n"]

summary = []
for scen, run_dir, art, cross, signal, demand in SCEN:
    cfg = build_scenario_config(scen, episode_steps=600)
    agent = DQNAgent.from_checkpoint(torch.load(f"../results/{run_dir}/best_dqn_model.pt", map_location="cpu"), device="cpu")
    hold = max(1, round(30 / cfg.decision_interval))
    probe = SumoRLEnv(cfg); probe.reset()
    pdem = [max((art if ap in ARTERIAL else cross) for ap in probe.served_approaches(a)) if probe.served_approaches(a) else 1.0
            for a in range(probe.action_size)]
    probe.close()
    _, _, wholds = webster_green(pdem, [2]*len(pdem), cfg.decision_interval)

    res = {
        "DQN": run(cfg, lambda e: LearnedController(agent)),
        "Naive (30s)": run(cfg, lambda e: HoldFixedController(e.action_size, hold)),
        "Webster": run(cfg, lambda e: WebsterFixedController(e.action_size, wholds)),
        "Actuated": run(cfg, lambda e: ActuatedController(e)),
    }
    lines.append(f"\n## {signal} — {demand} (Awolowo {art} / Allen {cross} veh/h)\n")
    lines.append("| Controller | Avg queue | Accumulated wait (s) | Throughput |")
    lines.append("|---|---|---|---|")
    for name, rows in res.items():
        lines.append(f"| {name} | {avg(rows,'avg_queue'):.2f} | {avg(rows,'avg_acc_wait'):.1f} | {avg(rows,'throughput'):.0f} |")
    dq, da = avg(res["DQN"], "avg_queue"), avg(res["DQN"], "avg_acc_wait")
    imp = []
    for b in ("Naive (30s)", "Webster", "Actuated"):
        bq, ba = avg(res[b], "avg_queue"), avg(res[b], "avg_acc_wait")
        imp.append(f"vs {b}: queue {pct(bq,dq):+.0f}%, acc-wait {pct(ba,da):+.0f}%")
    lines.append("\n*DQN improvement:* " + "; ".join(imp))
    summary.append((signal, demand, dq, da,
                    pct(avg(res['Naive (30s)'],'avg_acc_wait'), da),
                    pct(avg(res['Webster'],'avg_acc_wait'), da),
                    pct(avg(res['Actuated'],'avg_acc_wait'), da)))
    print(f"done: {scen}")

lines.append("\n\n## Summary — DQN accumulated-wait reduction (%)\n")
lines.append("| Signal | Demand | DQN queue | DQN acc-wait | vs Naive | vs Webster | vs Actuated |")
lines.append("|---|---|---|---|---|---|---|")
for s, d, q, a, n, w, ac in summary:
    lines.append(f"| {s} | {d} | {q:.2f} | {a:.1f} | {n:+.0f}% | {w:+.0f}% | {ac:+.0f}% |")

open("../results/full_eval_matrix.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\nsaved ../results/full_eval_matrix.md")
