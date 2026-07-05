"""Compare the trained DQN against TWO fixed-time baselines:
  1. The project's fast baseline (switches every decision = every 5s)
  2. A REALISTIC fixed timer (holds each phase ~30s, like a real signal)

Usage (from src/):
    ..\\venv\\Scripts\\python.exe baseline_compare.py --scenario lagos_single \
        --checkpoint ..\\results\\lagos_single_rl\\best_dqn_model.pt
"""
from __future__ import annotations

import argparse
import torch

from ai_traffic_light.config import build_scenario_config
from ai_traffic_light.dqn import DQNAgent
from ai_traffic_light.rl_env import SumoRLEnv
from ai_traffic_light.evaluation import FixedTimeController, LearnedController


def _run_episode_acc(env, make_ctrl):
    """Run one episode, returning avg queue, avg (resetting) wait, and avg ACCUMULATED wait.
    `make_ctrl(env)` is called AFTER reset so the controller sees a valid action_size."""
    state = env.reset()
    controller = make_ctrl(env)
    done = False
    q = w = aw = 0.0
    n = 0
    arrived = 0
    while not done:
        state, _, done, info = env.step(controller.select_action(state))
        q += info["total_queue"]
        w += info["total_wait"]
        aw += info["total_acc_wait"]
        arrived = info["arrived_total"]
        n += 1
    env.close()
    return {"avg_queue": q / max(n, 1), "avg_wait": w / max(n, 1),
            "avg_acc_wait": aw / max(n, 1), "throughput": arrived}


class HoldFixedController:
    """Realistic fixed timer: holds each phase for `hold` decisions before switching."""
    def __init__(self, action_size: int, hold: int):
        self.action_size = action_size
        self.hold = hold
        self._action = 0
        self._count = 0

    def select_action(self, _state):
        if self._count >= self.hold:
            self._action = (self._action + 1) % self.action_size
            self._count = 0
        self._count += 1
        return self._action


class WebsterFixedController:
    """Webster-optimal fixed timer: each phase held for a per-phase green proportional
    to its demand (the textbook best fixed-time plan)."""
    def __init__(self, action_size: int, holds: list[int]):
        self.action_size = action_size
        self.holds = holds          # decisions to hold each phase
        self._action = 0
        self._count = 0

    def select_action(self, _state):
        if self._count >= self.holds[self._action % len(self.holds)]:
            self._action = (self._action + 1) % self.action_size
            self._count = 0
        self._count += 1
        return self._action


class ActuatedController:
    """Gap-out ACTUATED control (a fair non-RL adaptive baseline): holds the current
    green for at least `min_green` decisions, extends it while its approaches still
    have queue (up to `max_green`), then switches to the highest-queue phase. The
    min/max green prevents the thrashing a naive longest-queue policy suffers."""
    def __init__(self, env, min_green=2, max_green=8, gap_thresh=0.03):
        self.approach_ids = env.approach_ids
        self.action_size = env.action_size
        self.served = [env.served_approaches(a) for a in range(env.action_size)]
        self.min_green, self.max_green, self.gap_thresh = min_green, max_green, gap_thresh
        self._action = 0
        self._count = 0

    def _phase_queue(self, q, a):
        return sum(q.get(ap, 0.0) for ap in self.served[a])

    def select_action(self, state):
        q = {ap: state[i * 3] for i, ap in enumerate(self.approach_ids)}  # normalized queue per approach
        self._count += 1
        if self._count < self.min_green:
            return self._action                                  # enforce minimum green
        cur = self._phase_queue(q, self._action)
        if cur < self.gap_thresh or self._count >= self.max_green:  # gap-out or max green -> reconsider
            best_a = max(range(self.action_size), key=lambda a: self._phase_queue(q, a))
            if best_a != self._action:
                self._action, self._count = best_a, 0
        return self._action


def webster_green(phase_demands_vph, lanes_per_phase, decision_interval,
                  sat_per_lane=1800.0, lost_per_phase=4.0, max_cycle=120.0):
    """Return (cycle_s, [green_s per phase], [hold decisions per phase]) by Webster's method.
    phase_demands_vph: critical approach demand (veh/h) serving each phase.
    """
    ys = [d / (ln * sat_per_lane) for d, ln in zip(phase_demands_vph, lanes_per_phase)]
    Y = sum(ys)
    n = len(phase_demands_vph)
    L = n * lost_per_phase
    Y = min(Y, 0.92)                                  # guard against >=1
    C = min(max_cycle, (1.5 * L + 5.0) / (1.0 - Y))   # cap at a practical cycle length
    eff = C - L
    greens = [max(decision_interval, (y / Y) * eff) for y in ys]
    holds = [max(1, round(g / decision_interval)) for g in greens]
    return C, greens, holds


def avg(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="lagos_single")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--green-seconds", type=int, default=30, help="Naive fixed-timer green per phase")
    p.add_argument("--arterial-vph", type=float, default=1400.0, help="Arterial peak demand veh/h/dir (phase 0)")
    p.add_argument("--cross-vph", type=float, default=280.0, help="Cross-street demand veh/h/dir (phase 1)")
    p.add_argument("--lanes", type=int, default=2, help="Lanes per approach")
    args = p.parse_args()

    cfg = build_scenario_config(args.scenario, episode_steps=600)
    hold = max(1, round(args.green_seconds / cfg.decision_interval))

    # Probe the junction to map each green phase to the demand of the approach(es) it serves,
    # so Webster works for any phase count (2-phase opposing pairs OR 4-phase per-approach).
    ARTERIAL = {"610011092#2", "134404792#3"}
    probe = SumoRLEnv(cfg)
    probe.reset()
    phase_demands = []
    for a in range(probe.action_size):
        served = probe.served_approaches(a)
        d = max((args.arterial_vph if ap in ARTERIAL else args.cross_vph) for ap in served) if served else 1.0
        phase_demands.append(d)
    probe.close()
    wcycle, wgreens, wholds = webster_green(
        phase_demands, [args.lanes] * len(phase_demands), cfg.decision_interval)

    agent = DQNAgent.from_checkpoint(torch.load(args.checkpoint, map_location="cpu"), device="cpu")

    def run(make_ctrl):
        env = SumoRLEnv(cfg)
        rows = []
        for _ in range(args.episodes):
            rows.append(_run_episode_acc(env, make_ctrl))
        return rows

    dqn = run(lambda env: LearnedController(agent))
    fast = run(lambda env: FixedTimeController(env.action_size))
    realistic = run(lambda env: HoldFixedController(env.action_size, hold))
    webster = run(lambda env: WebsterFixedController(env.action_size, wholds))
    actuated = run(lambda env: ActuatedController(env))

    print(f"\nScenario: {args.scenario}")
    print(f"Webster-optimal timing: cycle={wcycle:.0f}s  green per phase={[round(g) for g in wgreens]}s  "
          f"(arterial {args.arterial_vph:.0f} / cross {args.cross_vph:.0f} veh/h)\n")
    print(f"{'controller':<34}{'queue':>9}{'wait':>9}{'acc_wait':>10}{'throughput':>12}")
    for name, rows in [("DQN (learned)", dqn),
                       ("fixed FAST (5s switch)", fast),
                       (f"fixed NAIVE ({args.green_seconds}s equal)", realistic),
                       ("fixed WEBSTER-OPTIMAL", webster),
                       ("ACTUATED (longest-queue)", actuated)]:
        print(f"{name:<34}{avg(rows,'avg_queue'):>9.2f}{avg(rows,'avg_wait'):>9.1f}"
              f"{avg(rows,'avg_acc_wait'):>10.1f}{avg(rows,'throughput'):>12.0f}")

    dq, dw, da = avg(dqn, "avg_queue"), avg(dqn, "avg_wait"), avg(dqn, "avg_acc_wait")
    for label, base in [("NAIVE", realistic), ("WEBSTER", webster), ("ACTUATED", actuated)]:
        bq, bw, ba = avg(base, "avg_queue"), avg(base, "avg_wait"), avg(base, "avg_acc_wait")
        print(f"\nDQN vs {label}:  queue {(bq-dq)/bq*100:+.0f}%   "
              f"wait {(bw-dw)/bw*100:+.0f}%   acc_wait {(ba-da)/ba*100:+.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
