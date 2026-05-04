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
    )
    return replace(env_cfg, **overrides)
