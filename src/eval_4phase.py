"""Compare PressLight (_pl) and PDLight (_pd) 4-phase models against the actuated baseline.
The _pd models use a dynamic action space, so the actuated baseline runs on the plain
(non-dynamic) scenario with identical demand for a fair comparison."""
import torch
from ai_traffic_light.config import build_scenario_config
from ai_traffic_light.dqn import DQNAgent
from ai_traffic_light.rl_env import SumoRLEnv
from ai_traffic_light.evaluation import LearnedController
from baseline_compare import ActuatedController, _run_episode_acc

EPISODES = 2


def avg(rows, k):
    return sum(r[k] for r in rows) / len(rows)


def run(scenario, make_ctrl):
    env = SumoRLEnv(build_scenario_config(scenario, episode_steps=600))
    return [_run_episode_acc(env, make_ctrl) for _ in range(EPISODES)]


def load(run_dir):
    return DQNAgent.from_checkpoint(
        torch.load(f"../results/{run_dir}/best_dqn_model.pt", map_location="cpu"), device="cpu")


for d in ("peak", "asym", "bal", "reversed"):
    act = run(f"lagos_big4_{d}", lambda e: ActuatedController(e))          # plain net, 4 actions
    pl = run(f"lagos_big4_{d}_pl", (lambda ag: (lambda e: LearnedController(ag)))(load(f"lagos_big4_{d}_pl_rl")))
    pd = run(f"lagos_big4_{d}_pd", (lambda ag: (lambda e: LearnedController(ag)))(load(f"lagos_big4_{d}_pd_rl")))
    aw = avg(act, "avg_acc_wait")
    print(f"\n=== 4-phase {d.upper()} ===")
    print(f"{'controller':<22}{'queue':>8}{'acc_wait':>10}{'thru':>7}")
    for name, r in [("Actuated", act), ("PressLight (_pl)", pl), ("PDLight (_pd)", pd)]:
        print(f"{name:<22}{avg(r,'avg_queue'):>8.2f}{avg(r,'avg_acc_wait'):>10.1f}{avg(r,'throughput'):>7.0f}")
    print(f"  PressLight vs Actuated acc-wait: {(aw-avg(pl,'avg_acc_wait'))/aw*100:+.0f}%")
    print(f"  PDLight    vs Actuated acc-wait: {(aw-avg(pd,'avg_acc_wait'))/aw*100:+.0f}%")
