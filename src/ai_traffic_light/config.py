from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class ScenarioPreset:
    name: str
    sumocfg: Path
    tls_id: str
    action_phases: tuple[int, ...] | None = None
    decision_interval: int = 10
    episode_steps: int = 600
    use_gui: bool = False
    seed: int | None = 42
    use_yellow: bool = True
    yellow_duration: int = 3
    queue_weight: float = 1.0
    wait_weight: float = 0.2
    switch_penalty: float = 0.5
    queue_norm: float = 20.0
    wait_norm: float = 120.0
    max_approaches: int = 4
    phase_features: bool = False
    use_acc_wait: bool = False
    acc_wait_norm: float = 600.0
    use_pressure: bool = False
    use_pressure_reward: bool = False
    pressure_norm: float = 20.0
    dynamic_duration: bool = False
    duration_options: tuple = (5, 10, 20)


def _scenario_presets() -> dict[str, ScenarioPreset]:
    return {
        "ideal": ScenarioPreset(
            name="ideal",
            sumocfg=PROJECT_ROOT / "sumo_scenarios" / "ideal.sumocfg",
            tls_id="c",
        ),
        "lagos_intersection": ScenarioPreset(
            name="lagos_intersection",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "intersection.sumocfg",
            tls_id="center",
        ),
        "lagos_peak": ScenarioPreset(
            name="lagos_peak",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "intersection_peak.sumocfg",
            tls_id="center",
            decision_interval=5,
            switch_penalty=0.1,
        ),
        "lagos_low": ScenarioPreset(
            name="lagos_low",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "intersection_low.sumocfg",
            tls_id="center",
            decision_interval=5,
            switch_penalty=0.1,
        ),
        "lagos_extreme": ScenarioPreset(
            name="lagos_extreme",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "intersection_extreme.sumocfg",
            tls_id="center",
            decision_interval=5,
            switch_penalty=0.1,
        ),
        "lagos_reversed": ScenarioPreset(
            name="lagos_reversed",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "intersection_reversed.sumocfg",
            tls_id="center",
            decision_interval=5,
            switch_penalty=0.1,
        ),
        "lagos_chaos": ScenarioPreset(
            name="lagos_chaos",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "intersection_chaos.sumocfg",
            tls_id="center",
            decision_interval=5,
            switch_penalty=0.1,
        ),
        "lagos_real": ScenarioPreset(
            name="lagos_real",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5,
            switch_penalty=0.1,
        ),
        "lagos_calib": ScenarioPreset(
            name="lagos_calib",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_calib.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5,
            switch_penalty=0.1,
        ),
        "lagos_calib_phase": ScenarioPreset(
            name="lagos_calib_phase",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_calib.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5,
            switch_penalty=0.1,
            phase_features=True,
        ),
        "lagos_estimated": ScenarioPreset(
            name="lagos_estimated",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_estimated.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5,
            switch_penalty=0.1,
            phase_features=True,
        ),
        "lagos_estimated_tuned": ScenarioPreset(
            name="lagos_estimated_tuned",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_estimated.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5,
            switch_penalty=0.1,
            wait_weight=0.7,   # was 0.2 — make waiting time matter as much as queue
            phase_features=True,
        ),
        "lagos_single": ScenarioPreset(
            name="lagos_single",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_single.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5,
            switch_penalty=0.1,
            wait_weight=0.7,
            phase_features=True,
        ),
        "lagos_single_bal": ScenarioPreset(
            name="lagos_single_bal",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_single_bal.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5,
            switch_penalty=0.1,
            wait_weight=0.7,
            phase_features=True,
        ),
        "lagos_single_bal_t1": ScenarioPreset(
            name="lagos_single_bal_t1",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_single_bal.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5,
            switch_penalty=0.05,
            wait_weight=1.5,
            phase_features=True,
        ),
        "lagos_single_peak": ScenarioPreset(
            name="lagos_single_peak",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_single_peak.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5,
            switch_penalty=0.05,
            wait_weight=1.0,
            phase_features=True,
        ),
        "lagos_single_peak_q": ScenarioPreset(
            name="lagos_single_peak_q",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_single_peak.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5,
            switch_penalty=0.05,
            wait_weight=0.2,   # queue-emphasis: push queue toward 50% (wait already wins big)
            phase_features=True,
        ),
        "lagos_single_peak2_q": ScenarioPreset(
            name="lagos_single_peak2_q",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_single_peak2.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5,
            switch_penalty=0.05,
            wait_weight=0.2,
            phase_features=True,
        ),
        "lagos_4ph": ScenarioPreset(
            name="lagos_4ph",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_single_4ph.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5,
            switch_penalty=0.05,
            wait_weight=0.5,
            phase_features=True,
        ),
        "lagos_4ph_asym": ScenarioPreset(
            name="lagos_4ph_asym",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_4ph_asym.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5, switch_penalty=0.05, wait_weight=0.5, phase_features=True,
        ),
        "lagos_4ph_bal": ScenarioPreset(
            name="lagos_4ph_bal",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_4ph_bal.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5, switch_penalty=0.05, wait_weight=0.5, phase_features=True,
        ),
        "lagos_4ph_heavy": ScenarioPreset(
            name="lagos_4ph_heavy",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_4ph_heavy.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5, switch_penalty=0.05, wait_weight=0.5, phase_features=True,
        ),
        "lagos_4ph_random": ScenarioPreset(
            name="lagos_4ph_random",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_4ph_random.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5, switch_penalty=0.05, wait_weight=0.5, phase_features=True,
        ),
        # --- Allen-dominant (reversed), unseen generalization, and stochastic demands ---
        "lagos_single_reversed": ScenarioPreset(
            name="lagos_single_reversed",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_single_reversed.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5, switch_penalty=0.05, wait_weight=0.5, phase_features=True,
        ),
        "lagos_single_gen": ScenarioPreset(
            name="lagos_single_gen",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_single_gen.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5, switch_penalty=0.05, wait_weight=0.5, phase_features=True,
        ),
        "lagos_single_stoch": ScenarioPreset(
            name="lagos_single_stoch",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_single_stoch.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5, switch_penalty=0.05, wait_weight=0.5, phase_features=True,
        ),
        "lagos_4ph_reversed": ScenarioPreset(
            name="lagos_4ph_reversed",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_4ph_reversed.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5, switch_penalty=0.05, wait_weight=0.5, phase_features=True,
        ),
        "lagos_4ph_gen": ScenarioPreset(
            name="lagos_4ph_gen",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_4ph_gen.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5, switch_penalty=0.05, wait_weight=0.5, phase_features=True,
        ),
        "lagos_4ph_stoch": ScenarioPreset(
            name="lagos_4ph_stoch",
            sumocfg=PROJECT_ROOT / "sumo_scenarios1" / "lagos_4ph_stoch.sumocfg",
            tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
            decision_interval=5, switch_penalty=0.05, wait_weight=0.5, phase_features=True,
        ),
        # === REALISTIC bigger network: long approaches (far spawning) + turning movements ===
        **{
            f"lagos_{pref}_{d}": ScenarioPreset(
                name=f"lagos_{pref}_{d}",
                sumocfg=PROJECT_ROOT / "sumo_scenarios1" / f"lagos_{cfg}_{d}.sumocfg",
                tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
                decision_interval=5, switch_penalty=0.05, wait_weight=0.5, phase_features=True,
            )
            for pref, cfg in [("big", "big"), ("big4", "big4")]
            for d in ("peak", "asym", "bal", "reversed")
        },
        # accumulated-wait reward variants (anti-starvation) of the realistic scenarios
        **{
            f"lagos_{pref}_{d}_aw": ScenarioPreset(
                name=f"lagos_{pref}_{d}_aw",
                sumocfg=PROJECT_ROOT / "sumo_scenarios1" / f"lagos_{cfg}_{d}.sumocfg",
                tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
                decision_interval=5, switch_penalty=0.05, wait_weight=1.0, phase_features=True,
                use_acc_wait=True, acc_wait_norm=600.0,
            )
            for pref, cfg in [("big", "big"), ("big4", "big4")]
            for d in ("peak", "asym", "bal", "reversed")
        },
        # max-pressure 4-phase variants: acc-wait reward + downstream-queue (pressure) state
        **{
            f"lagos_big4_{d}_mp": ScenarioPreset(
                name=f"lagos_big4_{d}_mp",
                sumocfg=PROJECT_ROOT / "sumo_scenarios1" / f"lagos_big4_{d}.sumocfg",
                tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
                decision_interval=5, switch_penalty=0.05, wait_weight=1.0, phase_features=True,
                use_acc_wait=True, acc_wait_norm=600.0, use_pressure=True,
            )
            for d in ("peak", "asym", "bal", "reversed")
        },
        # PressLight-style: pressure REWARD + pressure state (the technique that beats actuated)
        **{
            f"lagos_big4_{d}_pl": ScenarioPreset(
                name=f"lagos_big4_{d}_pl",
                sumocfg=PROJECT_ROOT / "sumo_scenarios1" / f"lagos_big4_{d}.sumocfg",
                tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
                decision_interval=5, switch_penalty=0.05, phase_features=True,
                use_pressure=True, use_pressure_reward=True, pressure_norm=20.0, acc_wait_norm=600.0,
            )
            for d in ("peak", "asym", "bal", "reversed")
        },
        # PDLight-style: pressure reward + pressure state + DYNAMIC phase duration (agent picks length)
        **{
            f"lagos_big4_{d}_pd": ScenarioPreset(
                name=f"lagos_big4_{d}_pd",
                sumocfg=PROJECT_ROOT / "sumo_scenarios1" / f"lagos_big4_{d}.sumocfg",
                tls_id="cluster_12245849475_12245849476_12245849480_12245849481_#9more",
                decision_interval=5, switch_penalty=0.05, phase_features=True,
                use_pressure=True, use_pressure_reward=True, pressure_norm=20.0, acc_wait_norm=600.0,
                dynamic_duration=True, duration_options=(5, 10, 20),
            )
            for d in ("peak", "asym", "bal", "reversed")
        },
    }


def list_scenario_presets() -> tuple[str, ...]:
    return tuple(sorted(_scenario_presets()))


def build_scenario_config(name: str = "ideal", **overrides):
    from .rl_env import EnvConfig

    presets = _scenario_presets()
    if name not in presets:
        options = ", ".join(sorted(presets))
        raise ValueError(f"Unknown scenario preset '{name}'. Available presets: {options}")

    preset = presets[name]
    env_cfg = EnvConfig(
        name=preset.name,
        sumocfg=str(preset.sumocfg),
        tls_id=preset.tls_id,
        action_phases=preset.action_phases,
        decision_interval=preset.decision_interval,
        episode_steps=preset.episode_steps,
        use_gui=preset.use_gui,
        seed=preset.seed,
        use_yellow=preset.use_yellow,
        yellow_duration=preset.yellow_duration,
        queue_weight=preset.queue_weight,
        wait_weight=preset.wait_weight,
        switch_penalty=preset.switch_penalty,
        queue_norm=preset.queue_norm,
        wait_norm=preset.wait_norm,
        max_approaches=preset.max_approaches,
        phase_features=preset.phase_features,
        use_acc_wait=preset.use_acc_wait,
        acc_wait_norm=preset.acc_wait_norm,
        use_pressure=preset.use_pressure,
        use_pressure_reward=preset.use_pressure_reward,
        pressure_norm=preset.pressure_norm,
        dynamic_duration=preset.dynamic_duration,
        duration_options=preset.duration_options,
    )
    return replace(env_cfg, **overrides)
