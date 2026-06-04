from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .dqn import DQNAgent
from .evaluation import FixedTimeController, LearnedController
from .rl_env import EnvConfig, SumoRLEnv


# ──────────────────────────────────────────────────────────────────────────────
# Intersection layout constants (axes span 0..10, junction box at 4..6)
# ──────────────────────────────────────────────────────────────────────────────

_CAR_S = 0.22          # car block size (square)
_CAR_STEP = 0.40       # spacing between successive queued cars
_MAX_CARS = 9          # max cars to draw before showing "+N" overflow

# Per-direction: position of first queued car (cx,cy) and step direction (dx,dy)
_LAYOUT: dict[str, dict] = {
    "N": dict(cx=5.0, cy=6.30, dx=0.0, dy=0.40),
    "S": dict(cx=5.0, cy=3.70, dx=0.0, dy=-0.40),
    "E": dict(cx=6.30, cy=5.0, dx=0.40, dy=0.0),
    "W": dict(cx=3.70, cy=5.0, dx=-0.40, dy=0.0),
}
_LIGHT_POS = {"N": (4.65, 6.05), "S": (5.35, 3.95), "E": (5.95, 5.35), "W": (4.05, 4.65)}
_LABEL_POS = {"N": (5.0, 9.3),  "S": (5.0, 0.7),  "E": (9.3, 5.0),  "W": (0.7, 5.0)}
_OVERFLOW_POS = {"N": (5.0, 9.75), "S": (5.0, 0.25), "E": (9.75, 5.0), "W": (0.25, 5.0)}


def _dir_of(approach: str) -> str:
    return {"n": "N", "s": "S", "e": "E", "w": "W"}.get(approach[0].lower(), "?")


# ──────────────────────────────────────────────────────────────────────────────
# Intersection drawing
# ──────────────────────────────────────────────────────────────────────────────

def _draw_intersection(
    ax,
    queues: dict[str, float],
    served: set[str],
    approach_ids: list[str],
) -> None:
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#12122a")

    # Roads (grey strips)
    ax.add_patch(mpatches.Rectangle((4, 0), 2, 10, color="#3a3a4a", zorder=0))   # N-S road
    ax.add_patch(mpatches.Rectangle((0, 4), 10, 2, color="#3a3a4a", zorder=0))   # E-W road
    ax.add_patch(mpatches.Rectangle((4, 4), 2, 2,  color="#4a4a5a", zorder=1))   # junction box

    # Centre dot
    ax.add_patch(plt.Circle((5, 5), 0.18, color="#666677", zorder=2))

    for approach in approach_ids:
        d = _dir_of(approach)
        lay = _LAYOUT.get(d)
        if lay is None:
            continue

        is_green = approach in served
        queue = int(queues.get(approach, 0))

        # Traffic-light indicator
        lx, ly = _LIGHT_POS[d]
        light_color = "#33ee55" if is_green else "#ee3333"
        ax.add_patch(plt.Circle((lx, ly), 0.19, color=light_color, zorder=5))
        ax.add_patch(plt.Circle((lx, ly), 0.19, fill=False, edgecolor="white", linewidth=0.6, zorder=6))

        # Queued-car blocks
        show = min(queue, _MAX_CARS)
        cx0, cy0 = lay["cx"], lay["cy"]
        dx, dy = lay["dx"], lay["dy"]
        s = _CAR_S
        for i in range(show):
            bx = cx0 + i * dx - s / 2
            by = cy0 + i * dy - s / 2
            ax.add_patch(mpatches.FancyBboxPatch(
                (bx, by), s, s,
                boxstyle="round,pad=0.03",
                facecolor="#3a7bd5",
                edgecolor="#88aaff",
                linewidth=0.4,
                zorder=4,
            ))

        # Overflow badge
        if queue > _MAX_CARS:
            ox, oy = _OVERFLOW_POS[d]
            ax.text(ox, oy, f"+{queue - _MAX_CARS}", fontsize=6, ha="center", va="center",
                    color="#ffaa33", fontweight="bold", zorder=7)

        # Direction label + queue count
        lbx, lby = _LABEL_POS[d]
        ax.text(lbx, lby, f"{d}  {queue}", fontsize=7.5, ha="center", va="center",
                color="white", fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#22223a", alpha=0.85, edgecolor="none"))


# ──────────────────────────────────────────────────────────────────────────────
# Episode recorder
# ──────────────────────────────────────────────────────────────────────────────

def record_run(env_config: EnvConfig, controller, label: str) -> dict:
    """Run one episode and capture per-decision snapshots."""
    env = SumoRLEnv(env_config)
    state = env.reset()
    frames: list[dict] = []
    done = False
    while not done:
        action = controller.select_action(state)
        state, _reward, done, info = env.step(action)
        frames.append({
            "time": info["step"],
            "queues": env.approach_queue_map(),
            "served": set(env.served_approaches(action)),
            "total_queue": info["total_queue"],
        })
    approach_ids = env.approach_ids
    env.close()
    return {"label": label, "frames": frames, "approach_ids": approach_ids}


def _frame_at(frames: list[dict], t: float) -> dict | None:
    """Latest frame with time <= t (step function)."""
    result = None
    for f in frames:
        if f["time"] <= t:
            result = f
        else:
            break
    return result


# ──────────────────────────────────────────────────────────────────────────────
# GIF renderer
# ──────────────────────────────────────────────────────────────────────────────

def _make_chart_data(frames: list[dict]) -> tuple[list, list]:
    ts = [f["time"] for f in frames]
    qs = [f["total_queue"] for f in frames]
    return ts, qs


def make_comparison_gif(
    dqn_run: dict,
    fixed_run: dict,
    output_path: str = "results/comparison.gif",
    fps: int = 10,
) -> None:
    approach_ids = dqn_run["approach_ids"]

    # Build a common time grid from the union of both runs
    all_times = sorted(set(f["time"] for f in dqn_run["frames"]) |
                       set(f["time"] for f in fixed_run["frames"]))
    # Keep at most 120 frames to limit file size (~12s at 10 fps)
    step = max(1, len(all_times) // 120)
    times = all_times[::step]
    max_time = all_times[-1]

    dqn_ts, dqn_qs = _make_chart_data(dqn_run["frames"])
    fix_ts, fix_qs = _make_chart_data(fixed_run["frames"])
    chart_ymax = max(fix_qs + [1]) * 1.15

    fig = plt.figure(figsize=(13, 7.5), facecolor="#12122a")
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.38, wspace=0.08,
                          left=0.04, right=0.97, top=0.93, bottom=0.08)
    ax_dqn   = fig.add_subplot(gs[0, 0])
    ax_fix   = fig.add_subplot(gs[0, 1])
    ax_chart = fig.add_subplot(gs[1, :])
    for ax in (ax_dqn, ax_fix, ax_chart):
        ax.set_facecolor("#12122a")

    # Static supertitle
    fig.text(0.5, 0.975, "AI Traffic Light — DQN vs Fixed-Time",
             ha="center", va="top", color="white", fontsize=13, fontweight="bold")

    def update(fi: int) -> None:
        t = times[fi]

        df = _frame_at(dqn_run["frames"], t)
        ff = _frame_at(fixed_run["frames"], t)

        if df:
            _draw_intersection(ax_dqn, df["queues"], df["served"], approach_ids)
            ax_dqn.set_title(f"DQN controller   (queue: {int(df['total_queue'])})",
                             color="#55ee77", fontsize=10, fontweight="bold", pad=5)
        if ff:
            _draw_intersection(ax_fix, ff["queues"], ff["served"], approach_ids)
            ax_fix.set_title(f"Fixed-time   (queue: {int(ff['total_queue'])})",
                             color="#ff6b6b", fontsize=10, fontweight="bold", pad=5)

        ax_chart.clear()
        ax_chart.set_facecolor("#12122a")
        # DQN line
        d_idx = [i for i, tv in enumerate(dqn_ts) if tv <= t]
        f_idx = [i for i, tv in enumerate(fix_ts) if tv <= t]
        if d_idx:
            ax_chart.plot([dqn_ts[i] for i in d_idx], [dqn_qs[i] for i in d_idx],
                          color="#55ee77", linewidth=2.0, label="DQN")
        if f_idx:
            ax_chart.plot([fix_ts[i] for i in f_idx], [fix_qs[i] for i in f_idx],
                          color="#ff6b6b", linewidth=2.0, label="Fixed-time")
        # Faint full series for context
        ax_chart.plot(dqn_ts, dqn_qs, color="#55ee77", linewidth=0.5, alpha=0.18)
        ax_chart.plot(fix_ts, fix_qs, color="#ff6b6b", linewidth=0.5, alpha=0.18)
        ax_chart.axvline(t, color="white", linewidth=0.9, alpha=0.4)
        ax_chart.set_xlim(0, max_time)
        ax_chart.set_ylim(0, chart_ymax)
        ax_chart.set_xlabel("Simulation time (s)", color="#aaaaaa", fontsize=8)
        ax_chart.set_ylabel("Total queue", color="#aaaaaa", fontsize=8)
        ax_chart.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax_chart.spines.values():
            spine.set_color("#444455")
        leg = ax_chart.legend(loc="upper right", fontsize=8,
                               facecolor="#22223a", edgecolor="#555566", labelcolor="white")
        ax_chart.text(0.01, 0.92, f"t = {int(t)} s", transform=ax_chart.transAxes,
                      color="#aaaaaa", fontsize=7)

    ani = FuncAnimation(fig, update, frames=len(times), interval=1000 // fps, blit=False)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────────────

def make_comparison(
    checkpoint_path: str,
    env_config: EnvConfig,
    output_path: str = "results/comparison.gif",
    fps: int = 10,
) -> dict:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    agent = DQNAgent.from_checkpoint(checkpoint, device="cpu")

    print("Recording DQN episode…")
    dqn_run = record_run(env_config, LearnedController(agent), "DQN")
    print(f"  {len(dqn_run['frames'])} decisions, avg queue {np.mean([f['total_queue'] for f in dqn_run['frames']]):.1f}")

    print("Recording Fixed-time episode…")
    fixed_run = record_run(env_config, FixedTimeController(agent.action_dim), "Fixed-time")
    print(f"  {len(fixed_run['frames'])} decisions, avg queue {np.mean([f['total_queue'] for f in fixed_run['frames']]):.1f}")

    print("Rendering GIF…")
    make_comparison_gif(dqn_run, fixed_run, output_path=output_path, fps=fps)

    avg_dqn = float(np.mean([f["total_queue"] for f in dqn_run["frames"]]))
    avg_fix = float(np.mean([f["total_queue"] for f in fixed_run["frames"]]))
    pct = (avg_fix - avg_dqn) / max(avg_fix, 1) * 100
    return {
        "output": str(Path(output_path).resolve()),
        "dqn_avg_queue": round(avg_dqn, 2),
        "fixed_avg_queue": round(avg_fix, 2),
        "queue_reduction_pct": round(pct, 1),
    }
