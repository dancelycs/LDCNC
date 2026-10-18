"""Sensitivity study for independently reimplemented published adaptations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = ROOT
LEGACY_ROOT = ROOT
GOVERNOR_ROOT = ROOT
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from public_comparator_predictors import (  # noqa: E402
    DynamicRBFModeComparator,
    FixedTimeASMOComparator,
    LearningEstimatorComparator,
    ParameterTriggeredComparator,
    SynchronousESOComparator,
)
from screen_certificate_governor import run_certificate_method  # noqa: E402
from screen_learned_effectiveness_control import certificate_parameters  # noqa: E402


def predictor(name: str, multiplier: float) -> object:
    if name == "sftsmc_eso":
        return SynchronousESOComparator(bandwidth=24.0 * multiplier)
    if name == "dl_rbf":
        return DynamicRBFModeComparator(width=0.055 * multiplier)
    if name == "pet_aftc":
        return ParameterTriggeredComparator(threshold=0.012 * multiplier)
    if name == "fasmo_cftc":
        return FixedTimeASMOComparator(
            low_gain=7.0 * multiplier, high_gain=5.0 * multiplier)
    if name == "le_ftc":
        return LearningEstimatorComparator(
            gain=18.0 * multiplier, sign_gain=0.018 * multiplier)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19301)
    parser.add_argument("--scenarios", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    namespace = argparse.Namespace(
        decision_interval=20, acceleration_bound=2.0,
        simulation_timestep=0.005)
    common = {
        "model_path": ROOT / "assets" / "seven_link_arm.xml",
        "seed": args.seed,
        "scenarios": args.scenarios,
        "stress_level": 0.27,
        "nominal_horizon": 1000,
        "maximum_horizon": 1800,
        "fault_profile": "pulse",
        "estimator_torque_mode": "external_effectiveness",
        **certificate_parameters(namespace),
    }
    methods = ("sftsmc_eso", "dl_rbf", "pet_aftc", "fasmo_cftc", "le_ftc")
    result: dict[str, object] = {
        "protocol": {"seed": args.seed, "scenarios": args.scenarios,
                     "multipliers": [0.8, 1.0, 1.2]},
        "methods": {},
    }
    for method in methods:
        result["methods"][method] = {}
        for multiplier in (0.8, 1.0, 1.2):
            payload = run_certificate_method(
                **common, effectiveness_predictor=predictor(method, multiplier))
            result["methods"][method][f"{multiplier:.1f}"] = payload
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        method: {
            setting: {"rmse": values["tracking_rmse"],
                      "failure": values["failure_rate"]}
            for setting, values in settings.items()
        }
        for method, settings in result["methods"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
