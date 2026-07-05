"""Generate clean diagrams for the defense slides: system architecture and the RL loop."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("../results/figures", exist_ok=True)
NAVY, GREEN, AMBER, GREY = "#10256b", "#1a7a32", "#c98a1a", "#5f6b7a"


def box(ax, x, y, w, h, text, color, fc="white", tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                ec=color, fc=fc, lw=2.2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10.5,
            color=tc or color, weight="bold", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22,
                                 lw=2.2, color=color))


# ---------- System architecture ----------
fig, ax = plt.subplots(figsize=(12, 5.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")
fig.suptitle("System Architecture", fontsize=15, weight="bold", color=NAVY)
boxes = [
    (0.2, "Camera /\nVideo feed", GREY),
    (2.4, "YOLO\nvehicle detection", GREEN),
    (4.7, "State vector\n(queue, wait,\ndensity / approach)", NAVY),
    (7.2, "DQN agent\n(Double DQN)", NAVY),
    (9.6, "Signal phase\ndecision", AMBER),
]
w, h, y = 1.9, 1.6, 2.4
for x, t, c in boxes:
    box(ax, x, y, w, h, t, c, fc=(c == NAVY and "#eef1fb" or "white"))
for i in range(len(boxes) - 1):
    arrow(ax, boxes[i][0] + w, y + h/2, boxes[i+1][0], y + h/2)
# environment box below
box(ax, 4.7, 0.3, 4.4, 1.1, "SUMO simulation  /  Real traffic lights (Raspberry Pi GPIO)", GREEN, fc="#eaf4ec")
arrow(ax, 10.55, y, 9.0, 1.4, AMBER)                       # decision -> lights
arrow(ax, 4.7, 0.85, 1.15, y, GREY)                        # environment -> camera (feedback)
ax.text(2.5, 1.5, "feedback loop", fontsize=9, color=GREY, style="italic")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("../results/figures/architecture.png", dpi=140, bbox_inches="tight")
print("saved architecture.png")

# ---------- RL loop ----------
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.set_xlim(0, 8.5); ax.set_ylim(0, 4.6); ax.axis("off")
fig.suptitle("Reinforcement Learning Loop", fontsize=15, weight="bold", color=NAVY)
box(ax, 0.6, 1.7, 2.6, 1.4, "AGENT\n(DQN controller)", NAVY, fc="#eef1fb")
box(ax, 5.3, 1.7, 2.6, 1.4, "ENVIRONMENT\n(Junction in SUMO)", GREEN, fc="#eaf4ec")
arrow(ax, 3.2, 2.7, 5.3, 2.7, NAVY)
ax.text(4.25, 2.95, "action\n(green phase)", ha="center", fontsize=9.5, color=NAVY)
arrow(ax, 5.3, 2.0, 3.2, 2.0, GREEN)
ax.text(4.25, 1.45, "state + reward\n(queues, delay)", ha="center", fontsize=9.5, color=GREEN)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("../results/figures/rl_loop.png", dpi=140, bbox_inches="tight")
print("saved rl_loop.png")
