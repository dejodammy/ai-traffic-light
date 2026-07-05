"""Generate the 3-panel training chart (reward / queue / wait) for any run from its
training_logs.npz, with a fixed-time baseline line. Mirrors the project's standard plot.

Usage (from src/):
    ..\\venv\\Scripts\\python.exe plot_training_curves.py --run ..\\results\\lagos_single_rl \
        --scenario lagos_single --out ..\\results\\lagos_single_rl\\training_curves.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rolling(x, w=10):
    if len(x) < 1:
        return x
    w = min(w, len(x))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def fixed_time_baseline(scenario):
    """Run one fixed-time episode to get baseline avg queue & wait (same metric as training)."""
    from ai_traffic_light.config import build_scenario_config
    from ai_traffic_light.rl_env import SumoRLEnv
    from ai_traffic_light.evaluation import FixedTimeController, _run_episode
    env = SumoRLEnv(build_scenario_config(scenario, episode_steps=600))
    env.reset()
    m = _run_episode(env, FixedTimeController(env.action_size))
    return m["avg_queue"], m["avg_wait"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="results dir containing training_logs.npz")
    p.add_argument("--scenario", default=None, help="scenario name for the fixed-time baseline line")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    d = np.load(Path(args.run) / "training_logs.npz")
    rewards, q, w = d["rewards"], d["avg_queue"], d["avg_wait"]
    ep = np.arange(1, len(rewards) + 1)
    ep_r = np.arange(10, len(rewards) + 1)

    bq = bw = None
    if args.scenario:
        try:
            bq, bw = fixed_time_baseline(args.scenario)
        except Exception as e:
            print(f"  (baseline skipped: {e})")

    title = args.title or f"DQN Training — {args.scenario or Path(args.run).name}"
    fig, ax = plt.subplots(3, 1, figsize=(11, 13))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    ax[0].plot(ep, rewards, color="#9fb4e0", lw=1, label="Episode reward")
    ax[0].plot(ep_r, rolling(rewards), color="#10256b", lw=2.2, label="Rolling avg (10)")
    ax[0].set(title="Episode Reward over Training", xlabel="Episode", ylabel="Total Reward")
    ax[0].legend(loc="upper left"); ax[0].grid(alpha=0.3)

    ax[1].plot(ep, q, color="#f0a6a0", lw=1, label="Avg queue per episode")
    ax[1].plot(ep_r, rolling(q), color="#a3120a", lw=2.2, label="Rolling avg (10)")
    if bq is not None:
        ax[1].axhline(bq, color="gray", ls="--", lw=1.6, label=f"Fixed-time baseline ({bq:.1f})")
    ax[1].set(title="Average Queue Length per Episode", xlabel="Episode", ylabel="Vehicles halting")
    ax[1].legend(loc="upper right"); ax[1].grid(alpha=0.3)

    ax[2].plot(ep, w, color="#9bd6a3", lw=1, label="Avg wait per episode")
    ax[2].plot(ep_r, rolling(w), color="#1a7a32", lw=2.2, label="Rolling avg (10)")
    if bw is not None:
        ax[2].axhline(bw, color="gray", ls="--", lw=1.6, label=f"Fixed-time baseline ({bw:,.0f})")
    ax[2].set(title="Average Waiting Time per Episode", xlabel="Episode", ylabel="Cumulative wait (seconds)")
    ax[2].legend(loc="upper right"); ax[2].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = args.out or str(Path(args.run) / "training_curves.png")
    plt.savefig(out, dpi=120)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
