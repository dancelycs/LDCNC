"""Evaluate the point and proposed interfaces under deployment perturbations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = ROOT
LEGACY_ROOT = ROOT
GOVERNOR_ROOT = ROOT
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from screen_certificate_governor import run_certificate_method  # noqa: E402
from screen_learned_effectiveness_control import (  # noqa: E402
    OnlineEffectivenessPredictor,
    certificate_parameters,
)


class DelayedPredictor:
    def __init__(self, predictor: OnlineEffectivenessPredictor, delay: int) -> None:
        self.predictor = predictor
        self.delay = delay
        self.queue: deque[np.ndarray] = deque()

    def __call__(self, env: object) -> np.ndarray:
        current = self.predictor(env)  # type: ignore[arg-type]
        self.queue.append(np.asarray(current, dtype=float).copy())
        if len(self.queue) <= self.delay:
            return np.ones_like(current)
        return self.queue.popleft()


def paired_summary(point: dict, proposed: dict) -> dict[str, float]:
    left = point["scenario_details"]
    right = proposed["scenario_details"]
    dr = np.asarray([b["tracking_rmse"] - a["tracking_rmse"]
                     for a, b in zip(left, right)])
    dc = np.asarray([b["mean_stage_cost"] - a["mean_stage_cost"]
                     for a, b in zip(left, right)])
    return {
        "delta_rmse": float(dr.mean()),
        "delta_cost": float(dc.mean()),
        "joint_improvement_rate": float(np.mean((dr < 0) & (dc < 0))),
        "point_failure_rate": float(point["failure_rate"]),
        "proposed_failure_rate": float(proposed["failure_rate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=19201)
    parser.add_argument("--scenarios", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = {
        "nominal": (0.03, 0, 0.0, 0, "simulator"),
        "acceleration_noise": (0.08, 0, 0.0, 0, "simulator"),
        "causal_acceleration": (0.03, 0, 0.0, 0, "causal_difference"),
        "estimator_delay": (0.03, 2, 0.0, 0, "simulator"),
        "torque_quantization": (0.03, 0, 0.5, 0, "simulator"),
        "asynchronous_change": (0.03, 0, 0.0, 50, "simulator"),
        "combined": (0.08, 2, 0.5, 50, "simulator"),
    }
    namespace = argparse.Namespace(
        decision_interval=20, acceleration_bound=2.0,
        simulation_timestep=0.005)
    result: dict[str, object] = {
        "seed": args.seed, "scenarios": args.scenarios,
        "conditions": {},
    }
    for name, (noise, delay, quantization, offset, acceleration_source) in protocol.items():
        common = {
            "model_path": ROOT / "assets" / "seven_link_arm.xml",
            "seed": args.seed,
            "scenarios": args.scenarios,
            "stress_level": 0.27,
            "nominal_horizon": 1000,
            "maximum_horizon": 1800,
            "fault_profile": "pulse",
            "estimator_torque_mode": "external_effectiveness",
            "acceleration_noise_std": noise,
            "actuator_event_offset_steps": offset,
            "torque_quantization": quantization,
            **certificate_parameters(namespace),
        }
        point_predictor = DelayedPredictor(
            OnlineEffectivenessPredictor(args.checkpoint, "history", "point",
                                         "dual", acceleration_source=acceleration_source), delay)
        proposed_predictor = DelayedPredictor(
            OnlineEffectivenessPredictor(
                args.checkpoint, "history", "latched_nominal_correction", "dual",
                acceleration_source=acceleration_source),
            delay)
        point = run_certificate_method(
            **common, effectiveness_predictor=point_predictor)
        proposed = run_certificate_method(
            **common, effectiveness_predictor=proposed_predictor)
        result["conditions"][name] = {
            "acceleration_noise_std": noise,
            "estimator_delay_samples": delay,
            "torque_quantization_nm": quantization,
            "actuator_drag_offset_samples": offset,
            "acceleration_source": acceleration_source,
            "point": point,
            "proposed": proposed,
            "paired": paired_summary(point, proposed),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value["paired"] for name, value in
                      result["conditions"].items()}, indent=2))


if __name__ == "__main__":
    main()
