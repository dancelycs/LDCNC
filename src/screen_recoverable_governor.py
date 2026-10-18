"""Screen exact- and elastic-deadline nonlinear reference governors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = ROOT
LEGACY_ROOT = ROOT
sys.path.insert(0, str(ROOT / "src"))

from screen_tracking_task import reference  # noqa: E402
from simulation_environment import (  # noqa: E402
    MAX_SPEED,
    MIN_SPEED,
    ProgressConstrainedEnvironment,
)


def risk_and_demand(
    env: ProgressConstrainedEnvironment,
    step: int,
    error_threshold: float,
    safety_gain: float,
    backlog_gain: float,
    minimum_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    errors = np.zeros(env.count)
    velocity_errors = np.zeros(env.count)
    for index, sample in enumerate(env.data):
        q_ref, dq_base, _ = reference(env.phase[index])
        errors[index] = np.linalg.norm(sample.qpos - q_ref)
        velocity_errors[index] = np.linalg.norm(
            sample.qvel - env.last_speed[index] * dq_base
        )
    error_risk = np.maximum(errors - error_threshold, 0.0) / 0.35
    velocity_risk = np.maximum(velocity_errors - 0.35, 0.0) / 0.80
    innovation_risk = np.maximum(env.innovation - 0.08, 0.0) / 0.20
    risk = np.clip(
        np.maximum.reduce((error_risk, velocity_risk, innovation_risk)),
        0.0,
        1.0,
    )
    safe_upper = np.clip(MAX_SPEED - safety_gain * risk, minimum_speed, MAX_SPEED)
    scheduled_progress = min((step + 1) * env.dt, env.target_progress)
    backlog = np.maximum(scheduled_progress - env.phase, 0.0)
    demand = np.clip(1.0 + backlog_gain * backlog, minimum_speed, MAX_SPEED)
    return safe_upper, demand, risk


def run_method(
    model_path: Path,
    seed: int,
    scenarios: int,
    stress_level: float,
    method: str,
    decision_interval: int,
    nominal_horizon: int,
    maximum_horizon: int,
    error_threshold: float,
    safety_gain: float,
    backlog_gain: float,
    recovery_minimum_speed: float,
    maximum_progress_acceleration: float,
    simulation_timestep: float | None = None,
) -> dict[str, object]:
    exact_deadline = method == "exact_governor"
    minimum_speed = MIN_SPEED if method != "elastic_governor" else recovery_minimum_speed
    env = ProgressConstrainedEnvironment(
        model_path,
        scenarios,
        seed,
        nominal_horizon,
        strong_switch=True,
        stress_level=stress_level,
        enforce_terminal_progress=exact_deadline,
        minimum_speed=minimum_speed,
        maximum_speed=MAX_SPEED,
        maximum_progress_acceleration=(
            maximum_progress_acceleration if method == "elastic_governor" else None
        ),
        include_progress_acceleration=True,
        simulation_timestep=simulation_timestep,
    )
    command = np.ones(scenarios)
    active = np.ones(scenarios, dtype=bool)
    completion_step = np.full(scenarios, maximum_horizon, dtype=np.int64)
    squared_error = np.zeros(scenarios)
    accumulated_cost = np.zeros(scenarios)
    sample_count = np.zeros(scenarios)
    maximum_error = np.zeros(scenarios)
    risk_sum = np.zeros(scenarios)
    decision_count = 0

    for step in range(maximum_horizon):
        if step % decision_interval == 0:
            if method == "fixed_speed":
                command[:] = 1.0
                risk = np.zeros(scenarios)
            else:
                safe_upper, demand, risk = risk_and_demand(
                    env,
                    step,
                    error_threshold,
                    safety_gain,
                    backlog_gain,
                    minimum_speed,
                )
                command = np.minimum(demand, safe_upper)
            risk_sum += risk
            decision_count += 1
        midpoint = 0.5 * (minimum_speed + MAX_SPEED)
        half_range = 0.5 * (MAX_SPEED - minimum_speed)
        raw = np.arctanh(
            np.clip((command - midpoint) / half_range, -0.999, 0.999)
        )[:, None]
        _, stage_cost, _ = env.step(raw)
        finite_cost = np.nan_to_num(
            stage_cost.astype(np.float64), nan=1.0e6, posinf=1.0e6, neginf=1.0e6
        )
        for index, sample in enumerate(env.data):
            if not active[index]:
                continue
            q_ref, _, _ = reference(env.phase[index])
            error = float(np.linalg.norm(sample.qpos - q_ref))
            squared_error[index] += error**2
            accumulated_cost[index] += min(float(finite_cost[index]), 1.0e6)
            sample_count[index] += 1
            maximum_error[index] = max(maximum_error[index], error)
        newly_complete = active & (env.phase >= env.target_progress - 1.0e-10)
        completion_step[newly_complete] = step + 1
        active[newly_complete] = False
        if not np.any(active):
            break

    failure = maximum_error > 0.75
    divergence = maximum_error > 10.0
    scenario_cost = accumulated_cost / np.maximum(sample_count, 1.0)
    tail_count = max(1, int(np.ceil(0.10 * scenarios)))
    successful = ~failure
    successful_tail_count = max(1, int(np.ceil(0.10 * successful.sum())))
    successful_cost = scenario_cost[successful]
    return {
        "tracking_rmse": float(
            np.sqrt(squared_error.sum() / np.maximum(sample_count.sum(), 1.0))
        ),
        "mean_stage_cost": float(scenario_cost.mean()),
        "stage_cost_cvar_90": float(np.sort(scenario_cost)[-tail_count:].mean()),
        "failure_rate": float(failure.mean()),
        "divergence_rate": float(divergence.mean()),
        "successful_tracking_rmse": float(
            np.sqrt(
                squared_error[successful].sum()
                / np.maximum(sample_count[successful].sum(), 1.0)
            )
        ),
        "successful_mean_stage_cost": float(
            successful_cost.mean() if successful_cost.size else np.nan
        ),
        "successful_stage_cost_cvar_90": float(
            np.sort(successful_cost)[-successful_tail_count:].mean()
            if successful_cost.size
            else np.nan
        ),
        "moderate_error_rate": float((maximum_error > 0.25).mean()),
        "maximum_tracking_error": float(maximum_error.max()),
        "mean_completion_time": float(completion_step.mean() * env.dt),
        "completion_time_cvar_90": float(
            np.sort(completion_step)[-tail_count:].mean() * env.dt
        ),
        "incomplete_rate": float(active.mean()),
        "mean_risk": float(risk_sum.mean() / max(decision_count, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7101)
    parser.add_argument("--scenarios", type=int, default=64)
    parser.add_argument("--stress-levels", type=float, nargs="+", default=[0.2, 0.25, 0.27])
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--nominal-horizon", type=int, default=500)
    parser.add_argument("--maximum-horizon", type=int, default=700)
    parser.add_argument("--error-threshold", type=float, default=0.18)
    parser.add_argument("--safety-gain", type=float, default=0.8)
    parser.add_argument("--backlog-gain", type=float, default=1.5)
    parser.add_argument("--recovery-minimum-speed", type=float, default=0.05)
    parser.add_argument("--maximum-progress-acceleration", type=float, default=2.0)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "assets" / "seven_link_arm.xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "recoverable_governor_screen.json",
    )
    args = parser.parse_args()
    result: dict[str, object] = {
        "seed": args.seed,
        "configuration": {
            "decision_interval": args.decision_interval,
            "nominal_horizon": args.nominal_horizon,
            "maximum_horizon": args.maximum_horizon,
            "error_threshold": args.error_threshold,
            "safety_gain": args.safety_gain,
            "backlog_gain": args.backlog_gain,
            "recovery_minimum_speed": args.recovery_minimum_speed,
            "maximum_progress_acceleration": args.maximum_progress_acceleration,
        },
        "stress_levels": {},
    }
    for level in args.stress_levels:
        result["stress_levels"][str(level)] = {  # type: ignore[index]
            method: run_method(
                args.model,
                args.seed,
                args.scenarios,
                level,
                method,
                args.decision_interval,
                args.nominal_horizon,
                args.maximum_horizon,
                args.error_threshold,
                args.safety_gain,
                args.backlog_gain,
                args.recovery_minimum_speed,
                args.maximum_progress_acceleration,
            )
            for method in ("fixed_speed", "exact_governor", "elastic_governor")
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
