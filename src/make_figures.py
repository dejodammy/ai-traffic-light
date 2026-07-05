"""Generate figures for Chapter 4 (accumulated-wait metric only)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("../results/figures", exist_ok=True)

# Measured results on the realistic junction, rush-hour demand
controllers = ["DQN\n(proposed)", "Naive\nfixed (30s)", "Actuated\n(gap-out)", "Webster\noptimal"]
queue   = [4.45, 9.18, 8.48, 5.91]
accwait = [125.5, 407.1, 523.8, 316.4]
through = [433, 396, 405, 432]
colors  = ["#1a7a32", "#9aa0a6", "#c98a1a", "#a3120a"]

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
fig.suptitle("Figure 4.3: Controller performance on the realistic junction (rush-hour demand)",
             fontsize=12, fontweight="bold")
for (title, vals), a in zip(
        [("Average queue length (vehicles)", queue),
         ("Accumulated waiting time (s)", accwait),
         ("Throughput (vehicles served)", through)], ax):
    bars = a.bar(controllers, vals, color=colors, edgecolor="black", linewidth=0.5)
    a.set_title(title, fontsize=10); a.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width()/2, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("../results/figures/baseline_comparison.png", dpi=130)
print("saved baseline_comparison.png")

# Accumulated-wait reduction of the DQN vs each baseline
plt.figure(figsize=(7, 4.2))
base = ["vs Naive\nfixed", "vs Actuated", "vs Webster\noptimal"]
acc_imp = [69, 76, 60]
bars = plt.bar(base, acc_imp, color="#1a7a32", edgecolor="black", lw=0.5, width=0.55)
for b, v in zip(bars, acc_imp):
    plt.text(b.get_x() + b.get_width()/2, v, f"{v}%", ha="center", va="bottom", fontsize=10)
plt.ylabel("Accumulated-wait reduction by DQN (%)")
plt.title("Figure 4.4: Reduction in accumulated waiting time achieved by the DQN",
          fontsize=11, fontweight="bold")
plt.grid(axis="y", alpha=0.3); plt.ylim(0, 75); plt.tight_layout()
plt.savefig("../results/figures/delay_reduction.png", dpi=130)
print("saved delay_reduction.png")

# Training convergence: reward + queue only (no resetting-wait panel)
d = np.load("../results/lagos_big_peak_aw_rl/training_logs.npz")
rew, q = d["rewards"], d["avg_queue"]
ep = np.arange(1, len(rew) + 1)
def roll(x, w=10):
    w = min(w, len(x)); return np.convolve(x, np.ones(w)/w, mode="valid")
epr = np.arange(10, len(rew) + 1)
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("Figure 4.2: DQN training convergence", fontsize=13, fontweight="bold")
ax[0].plot(ep, rew, color="#9fb4e0", lw=1, label="Episode reward")
ax[0].plot(epr, roll(rew), color="#10256b", lw=2.2, label="Rolling avg (10)")
ax[0].set(title="Episode Reward over Training", xlabel="Episode", ylabel="Total Reward")
ax[0].legend(loc="lower right"); ax[0].grid(alpha=0.3)
ax[1].plot(ep, q, color="#f0a6a0", lw=1, label="Avg queue per episode")
ax[1].plot(epr, roll(q), color="#a3120a", lw=2.2, label="Rolling avg (10)")
ax[1].axhline(9.18, color="gray", ls="--", lw=1.6, label="Fixed-time baseline (9.18)")
ax[1].set(title="Average Queue Length per Episode", xlabel="Episode", ylabel="Vehicles halting")
ax[1].legend(loc="upper right"); ax[1].grid(alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("../results/figures/training_curves_peak.png", dpi=130)
print("saved training_curves_peak.png")

# Figure: DQN acc-wait reduction vs actuated, all 8 scenarios (2-phase vs 4-phase)
import numpy as np
scen = ["Rush-hour", "Asymmetric", "Balanced", "Allen-dom."]
two = [76, 19, 61, 52]
four = [-24, -84, -30, -29]
x = np.arange(len(scen)); w = 0.38
plt.figure(figsize=(9, 4.6))
b1 = plt.bar(x - w/2, two, w, label="Two-phase", color="#1a7a32", edgecolor="black", lw=0.5)
b2 = plt.bar(x + w/2, four, w, label="Four-phase", color="#c98a1a", edgecolor="black", lw=0.5)
plt.axhline(0, color="black", lw=0.8)
for bars in (b1, b2):
    for b in bars:
        v = b.get_height()
        plt.text(b.get_x()+b.get_width()/2, v + (2 if v >= 0 else -6), f"{v:+d}%", ha="center", fontsize=8)
plt.xticks(x, scen); plt.ylabel("Acc-wait reduction vs actuated (%)")
plt.title("Figure 4.4: DQN vs actuated control by signal architecture", fontsize=11, fontweight="bold")
plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig("../results/figures/dqn_vs_actuated.png", dpi=130)
print("saved dqn_vs_actuated.png")
