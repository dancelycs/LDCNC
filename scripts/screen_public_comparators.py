"""Evaluate recent independently adapted public comparators."""

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

from public_comparator_predictors import COMPARATORS  # noqa: E402
from screen_certificate_governor import run_certificate_method  # noqa: E402
from screen_learned_effectiveness_control import certificate_parameters  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("all", *COMPARATORS), default="all")
    parser.add_argument("--seed", type=int, default=18401)
    parser.add_argument("--scenarios", type=int, default=8)
    parser.add_argument("--stress-level", type=float, default=0.27)
    parser.add_argument("--simulation-timestep", type=float, default=0.005)
    parser.add_argument("--nominal-horizon", type=int, default=1000)
    parser.add_argument("--maximum-horizon", type=int, default=1800)
    parser.add_argument("--decision-interval", type=int, default=20)
    parser.add_argument("--acceleration-bound", type=float, default=2.0)
    parser.add_argument(
        "--fault-profile",
        choices=("pulse", "ramp", "intermittent", "persistent"),
        default="pulse",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "assets" / "seven_link_arm.xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "public_comparators_smoke.json",
    )
    args = parser.parse_args()

    selected = list(COMPARATORS) if args.method == "all" else [args.method]
    common = {
        "model_path": args.model,
        "seed": args.seed,
        "scenarios": args.scenarios,
        "stress_level": args.stress_level,
        "nominal_horizon": args.nominal_horizon,
        "maximum_horizon": args.maximum_horizon,
        "fault_profile": args.fault_profile,
        "estimator_torque_mode": "external_effectiveness",
        **certificate_parameters(args),
    }
    result: dict[str, object] = {
        "protocol": {
            "seed": args.seed,
            "scenarios": args.scenarios,
            "stress_level": args.stress_level,
            "fault_profile": args.fault_profile,
            "comparison_level": "matched estimator interface",
            "provenance": "independent engineering adaptations; not official code",
        },
        "methods": {},
    }
    methods = result["methods"]
    assert isinstance(methods, dict)
    for name in selected:
        predictor = COMPARATORS[name]()
        payload = run_certificate_method(
            **common,
            effectiveness_predictor=predictor,
        )
        methods[name] = {
            "metadata": predictor.metadata.__dict__,
            "parameters": {
                key: value
                for key, value in predictor.__dict__.items()
                if isinstance(value, (int, float, str, bool))
            },
            "result": payload,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

