"""Record the trained DQN driving the REAL Ikeja junction (lagos_calib_phase) in SUMO,
rendered top-down from actual vehicle positions (red = stopped/queued, green = moving).
Produces the defense fallback GIF. Adapted from make_sumo_demo.py."""
import os
import torch
import traci
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from ai_traffic_light.config import build_scenario_config
from ai_traffic_light.dqn import DQNAgent
from ai_traffic_light.rl_env import SumoRLEnv

SCEN = "lagos_4ph_random"
CKPT = "../results/lagos_4ph_random_rl/best_dqn_model.pt"
MAXSTEPS = 160

agent = DQNAgent.from_checkpoint(torch.load(CKPT, map_location="cpu"), device="cpu")
env = SumoRLEnv(build_scenario_config(SCEN, episode_steps=900))
state = env.reset()
tls = env.cfg.tls_id

segs, internal = [], []
for lid in traci.lane.getIDList():
    shp = traci.lane.getShape(lid)
    if len(shp) < 2:
        continue
    tgt = internal if lid.startswith(":") else segs
    for i in range(len(shp) - 1):
        tgt.append([shp[i], shp[i + 1]])

ends = []
for l in dict.fromkeys(traci.trafficlight.getControlledLanes(tls)):
    sh = traci.lane.getShape(l)
    if sh:
        ends.append(sh[-1])
cx = sum(p[0] for p in ends) / len(ends); cy = sum(p[1] for p in ends) / len(ends)
R = 110
xmin, xmax, ymin, ymax = cx - R, cx + R, cy - R, cy + R

frames = []
step = 0
print("simulating DQN control on real Ikeja junction...")
while step < MAXSTEPS and traci.simulation.getMinExpectedNumber() >= 0:
    action = agent.select_action(state, greedy=True)
    target = env._action_phases[action]
    cur = traci.trafficlight.getPhase(tls)
    env._run_yellow_if_needed(cur, target)
    traci.trafficlight.setPhase(tls, target)
    for _ in range(env.cfg.decision_interval):
        traci.simulationStep(); step += 1
        veh = traci.vehicle.getIDList()
        pts = [(traci.vehicle.getPosition(v), traci.vehicle.getSpeed(v)) for v in veh]
        frames.append((pts, env._total_queue()))
        if step >= MAXSTEPS:
            break
    state = env._get_state()
env.close()
print("frames:", len(frames))

NAVY = "#0e1f56"
fig, ax = plt.subplots(figsize=(9, 9)); fig.patch.set_facecolor("white")

def setup():
    ax.clear()
    if internal:
        ax.add_collection(LineCollection(internal, colors="#cfd6e4", linewidths=7, zorder=1))
    ax.add_collection(LineCollection(segs, colors="#cfd6e4", linewidths=10, zorder=1))
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.axis("off")

def draw(fi):
    setup()
    pts, totq = frames[fi]
    if pts:
        mx = [p[0][0] for p in pts]; my = [p[0][1] for p in pts]
        col = ["#d63b3b" if p[1] < 0.3 else "#1f9d55" for p in pts]
        ax.scatter(mx, my, c=col, s=26, zorder=4, edgecolors="black", linewidths=0.3)
    ax.set_title(f"DQN controller on the real Ikeja junction  ·  t = {fi}s",
                 fontsize=14, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.02, f"vehicles queued: {totq:.0f}      red = stopped   ·   green = moving",
            transform=ax.transAxes, ha="center", fontsize=11, color=NAVY)

anim = FuncAnimation(fig, draw, frames=len(frames), interval=120)
out = "../results/defense_real_junction_demo.gif"
anim.save(out, writer=PillowWriter(fps=9))
print("saved", os.path.abspath(out))
