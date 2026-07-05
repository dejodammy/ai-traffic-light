"""Generate an animated 1-v-1 comparison GIF: DQN vs naive, Webster and actuated controllers
running on the SAME junction and SAME traffic. Shows live per-approach queues + cumulative delay."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from ai_traffic_light.config import build_scenario_config
from ai_traffic_light.dqn import DQNAgent
from ai_traffic_light.rl_env import SumoRLEnv
from ai_traffic_light.evaluation import LearnedController
from baseline_compare import HoldFixedController, WebsterFixedController, ActuatedController, webster_green

SCEN = "lagos_big_peak"          # 2-phase rush-hour: DQN wins clearly
ART, CROSS = 1400.0, 280.0
ARTERIAL = {"610011092#2", "134404792#3"}
STEPS = 110


def record(make_ctrl):
    cfg = build_scenario_config(SCEN, episode_steps=600)
    env = SumoRLEnv(cfg)
    state = env.reset()
    ctrl = make_ctrl(env)
    approaches = env.approach_ids
    frames, cum, done = [], 0.0, False
    while not done and len(frames) < STEPS:
        a = ctrl.select_action(state)
        green = set(env.served_approaches(a))
        state, r, done, info = env.step(a)
        cum += info["total_queue"]
        frames.append((env.approach_queue_map(), green, info["total_queue"], cum))
    env.close()
    return approaches, frames


print("running controllers on the same traffic...")
agent = DQNAgent.from_checkpoint(
    torch.load(f"../results/{SCEN}_aw_rl/best_dqn_model.pt", map_location="cpu"), device="cpu")
probe = SumoRLEnv(build_scenario_config(SCEN)); probe.reset()
pdem = [max((ART if ap in ARTERIAL else CROSS) for ap in probe.served_approaches(a)) or 1.0
        for a in range(probe.action_size)]
probe.close()
_, _, wholds = webster_green(pdem, [2] * len(pdem), 5)

runs = {
    "DQN (proposed)":      record(lambda e: LearnedController(agent)),
    "Fixed-time (30s)":    record(lambda e: HoldFixedController(e.action_size, 6)),
    "Webster-optimal":     record(lambda e: WebsterFixedController(e.action_size, wholds)),
    "Actuated (gap-out)":  record(lambda e: ActuatedController(e)),
}
approaches = runs["DQN (proposed)"][0]
labels = ["Awolowo →" if a in ARTERIAL else "Allen ↑" for a in approaches]
labels = [f"{l}{i}" for i, l in enumerate(labels)]
n = min(len(v[1]) for v in runs.values())
qmax = max(max(f[2] for f in v[1][:n]) for v in runs.values()) * 1.15 + 1
print("frames:", n)

NAVY, GREEN, RED, GREY = "#0e1f56", "#1f9d55", "#d63b3b", "#9aa3b2"
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.patch.set_facecolor("white")
axes = axes.ravel()
order = list(runs.keys())


def draw(fi):
    for ax, name in zip(axes, order):
        ax.clear()
        _, frames = runs[name]
        qmap, green, totq, cum = frames[fi]
        vals = [qmap[a] for a in approaches]
        colors = [GREEN if a in green else RED for a in approaches]
        ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_ylim(0, qmax); ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8)
        win = (name == "DQN (proposed)")
        ax.set_title(f"{name}", fontsize=13, fontweight="bold", color=(NAVY if win else "#333"))
        ax.text(0.5, 0.93, f"queue now: {totq:.0f}", transform=ax.transAxes, ha="center",
                fontsize=11, color=NAVY, fontweight="bold")
        ax.text(0.5, 0.83, f"total delay so far: {cum:,.0f}", transform=ax.transAxes, ha="center",
                fontsize=9.5, color=GREY)
        for sp in ax.spines.values():
            sp.set_color(GREEN if win else "#dddddd"); sp.set_linewidth(2.5 if win else 1)
        ax.set_ylabel("vehicles queued", fontsize=9)
    fig.suptitle(f"Same junction · same traffic · live queue comparison      (t = {fi*5}s)",
                 fontsize=14, fontweight="bold", color=NAVY)
    fig.legend(handles=[plt.Rectangle((0,0),1,1,color=GREEN), plt.Rectangle((0,0),1,1,color=RED)],
               labels=["approach has GREEN", "approach waiting (red)"],
               loc="lower center", ncol=2, fontsize=9, frameon=False)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])


anim = FuncAnimation(fig, draw, frames=n, interval=140)
out = "../results/controller_comparison.gif"
anim.save(out, writer=PillowWriter(fps=7))
print("saved", os.path.abspath(out))
