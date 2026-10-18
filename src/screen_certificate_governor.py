"""Screen an analytic torque-certificate recoverable progress governor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = ROOT
LEGACY_ROOT = ROOT
sys.path.insert(0, str(ROOT / "src"))

from recoverable_governor import (  # noqa: E402
    affine_box_interval,
    affine_upper_interval,
    governor_decision,
    nominal_progress_acceleration,
    sampled_error_barrier_terms,
    sampled_clf_terms,
)
from predictive_recovery import (  # noqa: E402
    backup_tube_peak_error,
    feasible_backup_accelerations,
)
from screen_recoverable_governor import risk_and_demand, run_method  # noqa: E402
from screen_tracking_task import DRAG_PROFILE, FRICTION_PROFILE, reference  # noqa: E402
from simulation_environment import (  # noqa: E402
    CALIBRATED_BETA,
    MAX_SPEED,
    ProgressConstrainedEnvironment,
)
from validate_seven_link_mujoco import inverse_torque, set_payload  # noqa: E402


def torque_certificate(
    env: ProgressConstrainedEnvironment,
    index: int,
    acceleration_bound: float,
    actuator_lower_bound: float,
    torque_reserve: float,
    confidence_scale: float,
    clf_rate: float,
    clf_level: float,
    clf_remainder_bound: float,
    barrier_error_limit: float | None = None,
    barrier_frequency: float = 4.0,
    barrier_damping: float = 1.0,
    barrier_remainder_bound: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return the robust path-acceleration interval for one plant state."""
    sample = env.data[index]
    estimator = env.estimators[index]
    q_ref, path_tangent, path_curvature = reference(env.phase[index])
    speed = env.last_speed[index]
    q = sample.qpos.copy()
    dq = sample.qvel.copy()
    error = q - q_ref
    velocity_error = dq - speed * path_tangent
    acceleration_zero = (
        speed**2 * path_curvature
        - env.kp * error
        - env.kd * velocity_error
    )

    set_payload(
        env.inverse_model,
        env.inverse_payload,
        env.inverse_mass,
        env.inverse_inertia,
        0.05 + estimator.value[0],
    )
    env.inverse_model.dof_frictionloss[:] = FRICTION_PROFILE * estimator.value[1]
    covariance = np.linalg.inv(estimator.information + 1.0e-10 * np.eye(3))
    drag_standard_error = np.sqrt(max(covariance[2, 2], 0.0))
    lower_drag = float(
        np.clip(
            estimator.value[2] - CALIBRATED_BETA * drag_standard_error,
            0.0,
            estimator.value[2],
        )
    )
    drag_compensation = DRAG_PROFILE * lower_drag * dq * np.abs(dq)
    torque_zero = inverse_torque(
        env.inverse_model, env.inverse_data, q, dq, acceleration_zero
    ) + drag_compensation
    torque_unit = inverse_torque(
        env.inverse_model,
        env.inverse_data,
        q,
        dq,
        acceleration_zero + path_tangent,
    ) + drag_compensation
    torque_slope = torque_unit - torque_zero

    set_payload(
        env.inverse_model,
        env.inverse_payload,
        env.inverse_mass,
        env.inverse_inertia,
        0.05,
    )
    env.inverse_model.dof_frictionloss[:] = 0.0
    nominal_zero = inverse_torque(
        env.inverse_model, env.inverse_data, q, dq, acceleration_zero
    )
    set_payload(
        env.inverse_model,
        env.inverse_payload,
        env.inverse_mass,
        env.inverse_inertia,
        1.05,
    )
    payload_unit = inverse_torque(
        env.inverse_model, env.inverse_data, q, dq, acceleration_zero
    )
    set_payload(
        env.inverse_model,
        env.inverse_payload,
        env.inverse_mass,
        env.inverse_inertia,
        0.05,
    )
    env.inverse_model.dof_frictionloss[:] = FRICTION_PROFILE
    friction_unit = inverse_torque(
        env.inverse_model, env.inverse_data, q, dq, acceleration_zero
    )
    regressor = np.stack(
        (
            payload_unit - nominal_zero,
            friction_unit - nominal_zero,
            DRAG_PROFILE * dq * np.abs(dq),
        ),
        axis=-1,
    )
    parameter_margin = confidence_scale * np.sqrt(
        np.maximum(np.einsum("ij,jk,ik->i", regressor, covariance, regressor), 0.0)
    )
    disturbance_margin = env.innovation[index] / np.sqrt(regressor.shape[0])
    available_torque = (
        45.0 * actuator_lower_bound
        - torque_reserve
        - parameter_margin
        - disturbance_margin
    )
    lower, upper, torque_violation = affine_box_interval(
        torque_zero,
        torque_slope,
        available_torque,
        -acceleration_bound,
        acceleration_bound,
    )
    lyapunov, derivative_offset, derivative_slope = sampled_clf_terms(
        error,
        velocity_error,
        sample.qacc,
        path_tangent,
        path_curvature,
        float(speed),
        env.kp,
    )
    derivative_offset += clf_remainder_bound
    derivative_upper = -clf_rate * (lyapunov - clf_level)
    lower, upper, clf_violation = affine_upper_interval(
        derivative_offset,
        derivative_slope,
        derivative_upper,
        lower,
        upper,
    )
    barrier_violation = 0.0
    if barrier_error_limit is not None:
        h, hdot, hddot_offset, hddot_slope = sampled_error_barrier_terms(
            error,
            velocity_error,
            sample.qacc,
            path_tangent,
            path_curvature,
            float(speed),
            barrier_error_limit,
        )
        barrier_offset = -(
            hddot_offset
            + 2.0 * barrier_damping * barrier_frequency * hdot
            + barrier_frequency**2 * h
        )
        lower, upper, barrier_violation = affine_upper_interval(
            barrier_offset,
            -hddot_slope,
            -barrier_remainder_bound,
            lower,
            upper,
        )
    slope_energy = float(np.dot(torque_slope, torque_slope))
    relief_acceleration = (
        -float(np.dot(torque_slope, torque_zero)) / slope_energy
        if slope_energy > 1.0e-12
        else 0.0
    )
    relief_acceleration = float(
        np.clip(relief_acceleration, -acceleration_bound, acceleration_bound)
    )
    return (
        lower,
        upper,
        max(torque_violation, clf_violation, barrier_violation),
        relief_acceleration,
    )


def run_certificate_method(
    *,
    model_path: Path,
    seed: int,
    scenarios: int,
    stress_level: float,
    nominal_horizon: int,
    maximum_horizon: int,
    decision_interval: int,
    acceleration_bound: float,
    torque_reserve: float,
    confidence_scale: float,
    clf_rate: float,
    clf_level: float,
    clf_remainder_bound: float,
    maximum_recovery_jerk: float,
    recovery_mode: str,
    recovery_dwell_decisions: int = 0,
    effectiveness_compensation: str = "none",
    estimator_torque_mode: str = "commanded",
    simulation_timestep: float | None = None,
    risk_speed_constraint: bool = False,
    risk_error_threshold: float = 0.12,
    risk_safety_gain: float = 1.3,
    predictive_backup_constraint: bool = False,
    predictive_horizon_steps: int = 8,
    predictive_error_limit: float = 0.5,
    predictive_grid_size: int = 21,
    predictive_disturbance_scale: float = 0.0,
    barrier_error_limit: float | None = None,
    barrier_frequency: float = 4.0,
    barrier_damping: float = 1.0,
    barrier_remainder_bound: float = 0.0,
    active_identification_probe: bool = False,
    event_triggered_identification_probe: bool = False,
    effectiveness_predictor: Callable[[ProgressConstrainedEnvironment], np.ndarray]
    | None = None,
    dynamic_effectiveness_bound: bool = False,
    fault_profile: str = "pulse",
    fault_speed_derating: bool = False,
    freeze_physical_on_external_fault: bool = False,
    acceleration_noise_std: float = 0.03,
    actuator_event_offset_steps: int = 0,
    torque_quantization: float = 0.0,
    strong_switch: bool = True,
    disable_drag_switch: bool = False,
    return_details: bool = False,
) -> dict[str, object]:
    env = ProgressConstrainedEnvironment(
        model_path,
        scenarios,
        seed,
        nominal_horizon,
        strong_switch=strong_switch,
        stress_level=stress_level,
        enforce_terminal_progress=False,
        minimum_speed=0.0,
        maximum_speed=MAX_SPEED,
        include_progress_acceleration=True,
        effectiveness_compensation=effectiveness_compensation,
        estimator_torque_mode=estimator_torque_mode,
        effectiveness_window_steps=decision_interval,
        simulation_timestep=simulation_timestep,
        fault_profile=fault_profile,
        freeze_physical_on_external_fault=freeze_physical_on_external_fault,
        acceleration_noise_std=acceleration_noise_std,
        actuator_event_offset_steps=actuator_event_offset_steps,
        torque_quantization=torque_quantization,
    )
    if disable_drag_switch:
        env.drag_after[:] = env.drag_before
    actuator_lower_bound = max(0.25, 1.0 - 0.60 * stress_level)
    command = np.ones(scenarios)
    acceleration = np.zeros(scenarios)
    deadline = np.full(scenarios, nominal_horizon * env.dt)
    initial_deadline = deadline.copy()
    active = np.ones(scenarios, dtype=bool)
    completion_step = np.full(scenarios, maximum_horizon, dtype=np.int64)
    squared_error = np.zeros(scenarios)
    accumulated_cost = np.zeros(scenarios)
    sample_count = np.zeros(scenarios)
    maximum_error = np.zeros(scenarios)
    maximum_error_step = np.full(scenarios, -1, dtype=np.int64)
    first_moderate_step = np.full(scenarios, -1, dtype=np.int64)
    first_failure_step = np.full(scenarios, -1, dtype=np.int64)
    failure_during_event = np.zeros(scenarios, dtype=bool)
    violation_sum = np.zeros(scenarios)
    decision_count = np.zeros(scenarios)
    in_recovery = np.zeros(scenarios, dtype=bool)
    feasible_dwell = np.zeros(scenarios, dtype=np.int64)
    recovery_decision_count = np.zeros(scenarios)
    risk_sum = np.zeros(scenarios)
    predictive_filter_count = np.zeros(scenarios)
    predictive_empty_count = np.zeros(scenarios)
    effectiveness_error_event = np.zeros(scenarios)
    effectiveness_error_nominal = np.zeros(scenarios)
    effectiveness_count_event = np.zeros(scenarios)
    effectiveness_count_nominal = np.zeros(scenarios)
    nominal_selection_count_event = np.zeros(scenarios)
    nonnominal_selection_count_nominal = np.zeros(scenarios)
    innovation_sum_event = np.zeros(scenarios)
    innovation_sum_nominal = np.zeros(scenarios)
    maximum_innovation = np.zeros(scenarios)
    decision_period = decision_interval * env.dt

    for step in range(maximum_horizon):
        if step % decision_interval == 0:
            time = step * env.dt
            if risk_speed_constraint:
                risk_upper, _, risk = risk_and_demand(
                    env,
                    step,
                    error_threshold=risk_error_threshold,
                    safety_gain=risk_safety_gain,
                    backlog_gain=1.5,
                    minimum_speed=0.0,
                )
                risk_sum += risk
            else:
                risk_upper = np.full(scenarios, MAX_SPEED)
            if (
                fault_speed_derating
                and effectiveness_predictor is not None
                and hasattr(effectiveness_predictor, "latest_selected")
            ):
                selected_effectiveness = getattr(
                    effectiveness_predictor, "latest_selected"
                )
                if selected_effectiveness is not None:
                    risk_upper = np.minimum(
                        risk_upper,
                        np.clip(selected_effectiveness, 0.0, 1.0),
                    )
            for index in np.flatnonzero(active):
                certificate_effectiveness = actuator_lower_bound
                if (
                    dynamic_effectiveness_bound
                    and effectiveness_predictor is not None
                    and hasattr(effectiveness_predictor, "latest_lower")
                ):
                    learned_lower = getattr(
                        effectiveness_predictor, "latest_lower"
                    )
                    if learned_lower is not None:
                        certificate_effectiveness = max(
                            actuator_lower_bound,
                            float(learned_lower[int(index)]),
                        )
                lower, upper, violation, relief_acceleration = torque_certificate(
                    env,
                    int(index),
                    acceleration_bound,
                    certificate_effectiveness,
                    torque_reserve,
                    confidence_scale,
                    clf_rate,
                    clf_level,
                    clf_remainder_bound,
                    barrier_error_limit,
                    barrier_frequency,
                    barrier_damping,
                    barrier_remainder_bound,
                )
                risk_acceleration_upper = (
                    risk_upper[index] - env.last_speed[index]
                ) / decision_period
                upper = min(upper, float(risk_acceleration_upper))
                violation = max(violation, lower - upper, 0.0)
                if predictive_backup_constraint and lower <= upper:
                    sample = env.data[int(index)]
                    q_ref, path_tangent, _ = reference(float(env.phase[index]))
                    error = sample.qpos - q_ref
                    velocity_error = (
                        sample.qvel - env.last_speed[index] * path_tangent
                    )
                    predictor = lambda candidate: backup_tube_peak_error(
                        error=error,
                        velocity_error=velocity_error,
                        progress=float(env.phase[index]),
                        speed=float(env.last_speed[index]),
                        first_acceleration=candidate,
                        maximum_acceleration=acceleration_bound,
                        decision_period=decision_period,
                        horizon_steps=predictive_horizon_steps,
                        actuator_lower_bound=certificate_effectiveness,
                        proportional_gain=env.kp,
                        derivative_gain=env.kd,
                        reference=reference,
                        disturbance_acceleration_bound=(
                            predictive_disturbance_scale * env.innovation[index]
                        ),
                    )
                    remaining_time = max(
                        float(deadline[index]) - time, decision_period
                    )
                    nominal_acceleration = nominal_progress_acceleration(
                        max(env.target_progress - env.phase[index], 0.0),
                        float(env.last_speed[index]),
                        remaining_time,
                        acceleration_bound,
                    )
                    nominal_acceleration = float(
                        np.clip(nominal_acceleration, lower, upper)
                    )
                    if predictor(nominal_acceleration) <= predictive_error_limit:
                        selected = nominal_acceleration
                        candidates = np.array([selected])
                    else:
                        candidates, _ = feasible_backup_accelerations(
                            lower=lower,
                            upper=upper,
                            grid_size=predictive_grid_size,
                            error_limit=predictive_error_limit,
                            predictor=predictor,
                        )
                    if candidates.size:
                        selected = float(
                            candidates[np.argmin(np.abs(candidates - nominal_acceleration))]
                        )
                        if abs(selected - nominal_acceleration) > 1.0e-9:
                            predictive_filter_count[index] += 1
                        lower = selected
                        upper = selected
                    else:
                        violation = max(violation, 1.0e-12)
                        predictive_empty_count[index] += 1
                recovery_acceleration = (
                    -acceleration_bound
                    if recovery_mode == "hard_brake"
                    else relief_acceleration
                )
                decision_violation = violation
                if recovery_dwell_decisions > 0:
                    if violation > 0.0:
                        in_recovery[index] = True
                        feasible_dwell[index] = 0
                    if in_recovery[index]:
                        if violation <= 0.0 and env.last_speed[index] <= 0.02:
                            feasible_dwell[index] += 1
                        elif violation > 0.0:
                            feasible_dwell[index] = 0
                        if feasible_dwell[index] >= recovery_dwell_decisions:
                            in_recovery[index] = False
                            feasible_dwell[index] = 0
                        else:
                            decision_violation = max(violation, 1.0e-12)
                            recovery_acceleration = (
                                -acceleration_bound
                                if env.last_speed[index] > 0.02
                                else 0.0
                            )
                            recovery_decision_count[index] += 1
                decision = governor_decision(
                    time=time,
                    deadline=float(deadline[index]),
                    distance=max(env.target_progress - env.phase[index], 0.0),
                    speed=float(env.last_speed[index]),
                    decision_period=decision_period,
                    maximum_speed=MAX_SPEED,
                    maximum_acceleration=acceleration_bound,
                    certificate_lower=lower,
                    certificate_upper=upper,
                    certificate_violation=decision_violation,
                    recovery_acceleration=recovery_acceleration,
                    previous_acceleration=float(acceleration[index]),
                    maximum_recovery_jerk=maximum_recovery_jerk,
                )
                acceleration[index] = decision.acceleration
                deadline[index] = decision.deadline
                violation_sum[index] += decision.certificate_violation
                decision_count[index] += 1
        command[active] = np.clip(
            command[active] + acceleration[active] * env.dt, 0.0, MAX_SPEED
        )
        command[~active] = 0.0
        midpoint = 0.5 * MAX_SPEED
        half_range = 0.5 * MAX_SPEED
        raw = np.arctanh(
            np.clip((command - midpoint) / half_range, -0.999, 0.999)
        )[:, None]
        identification_probe = None
        if active_identification_probe:
            # The 2 N m amplitude exactly uses the authority already removed
            # from the robust certificate by ``torque_reserve``.
            sign = 1.0 if step % 2 == 0 else -1.0
            identification_probe = (
                sign * torque_reserve * env.identification_null_direction
            )
        elif event_triggered_identification_probe:
            sign = 1.0 if step % 2 == 0 else -1.0
            trigger = np.array(
                [
                    env.innovation[index]
                    > float(env.estimators[index].huber_delta)
                    for index in range(scenarios)
                ],
                dtype=float,
            )
            identification_probe = (
                sign
                * torque_reserve
                * trigger[:, None]
                * env.identification_null_direction
            )
        predicted_effectiveness = (
            effectiveness_predictor(env)
            if effectiveness_predictor is not None
            else None
        )
        _, stage_cost, _ = env.step(
            raw,
            identification_probe=identification_probe,
            estimator_effectiveness_override=predicted_effectiveness,
        )
        finite_cost = np.nan_to_num(
            stage_cost.astype(np.float64), nan=1.0e6, posinf=1.0e6, neginf=1.0e6
        )
        for index, sample in enumerate(env.data):
            if not active[index]:
                continue
            true_effectiveness = float(env.true_effectiveness[index])
            in_actuator_event = true_effectiveness < 0.999
            effectiveness_error = abs(
                env.estimated_effectiveness[index] - true_effectiveness
            )
            if in_actuator_event:
                effectiveness_error_event[index] += effectiveness_error
                effectiveness_count_event[index] += 1
                nominal_selection_count_event[index] += float(
                    env.estimated_effectiveness[index] >= 1.0 - 1.0e-9
                )
                innovation_sum_event[index] += env.innovation[index]
            else:
                effectiveness_error_nominal[index] += effectiveness_error
                effectiveness_count_nominal[index] += 1
                nonnominal_selection_count_nominal[index] += float(
                    env.estimated_effectiveness[index] < 1.0 - 1.0e-9
                )
                innovation_sum_nominal[index] += env.innovation[index]
            maximum_innovation[index] = max(
                maximum_innovation[index], float(env.innovation[index])
            )
            q_ref, _, _ = reference(env.phase[index])
            error = float(np.linalg.norm(sample.qpos - q_ref))
            squared_error[index] += error**2
            accumulated_cost[index] += min(float(finite_cost[index]), 1.0e6)
            sample_count[index] += 1
            if error > maximum_error[index]:
                maximum_error[index] = error
                maximum_error_step[index] = step + 1
            if error > 0.25 and first_moderate_step[index] < 0:
                first_moderate_step[index] = step + 1
            if error > 0.75 and first_failure_step[index] < 0:
                first_failure_step[index] = step + 1
                failure_during_event[index] = in_actuator_event
        newly_complete = active & (env.phase >= env.target_progress - 1.0e-10)
        completion_step[newly_complete] = step + 1
        active[newly_complete] = False
        if not np.any(active):
            break

    scenario_cost = accumulated_cost / np.maximum(sample_count, 1.0)
    tail_count = max(1, int(np.ceil(0.10 * scenarios)))
    failure = maximum_error > 0.75
    divergence = maximum_error > 10.0
    successful = ~failure
    successful_cost = scenario_cost[successful]
    successful_tail_count = max(1, int(np.ceil(0.10 * successful.sum())))
    result: dict[str, object] = {
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
        "mean_deadline_extension": float((deadline - initial_deadline).mean()),
        "mean_certificate_violation": float(
            (violation_sum / np.maximum(decision_count, 1.0)).mean()
        ),
        "mean_recovery_fraction": float(
            (recovery_decision_count / np.maximum(decision_count, 1.0)).mean()
        ),
        "mean_risk": float(
            np.mean(risk_sum / np.maximum(decision_count, 1.0))
        ),
        "mean_predictive_filter_fraction": float(
            (predictive_filter_count / np.maximum(decision_count, 1.0)).mean()
        ),
        "mean_predictive_empty_fraction": float(
            (predictive_empty_count / np.maximum(decision_count, 1.0)).mean()
        ),
        "effectiveness_mae_during_event": float(
            (
                effectiveness_error_event
                / np.maximum(effectiveness_count_event, 1.0)
            ).mean()
        ),
        "effectiveness_mae_outside_event": float(
            (
                effectiveness_error_nominal
                / np.maximum(effectiveness_count_nominal, 1.0)
            ).mean()
        ),
    }
    if return_details:
        details = []
        for index in range(scenarios):
            failure_step = int(first_failure_step[index])
            switch_step = int(env.switch_step[index])
            details.append(
                {
                    "scenario": index,
                    "tracking_rmse": float(
                        np.sqrt(
                            squared_error[index]
                            / max(sample_count[index], 1.0)
                        )
                    ),
                    "mean_stage_cost": float(scenario_cost[index]),
                    "maximum_error": float(maximum_error[index]),
                    "maximum_error_step": int(maximum_error_step[index]),
                    "first_moderate_step": int(first_moderate_step[index]),
                    "first_failure_step": failure_step,
                    "failure_during_actuator_event": bool(
                        failure_step >= 0 and failure_during_event[index]
                    ),
                    "switch_step": switch_step,
                    "actuator_effectiveness": float(env.actuator_effectiveness[index]),
                    "payload": float(env.payload[index]),
                    "friction_scale": float(env.friction[index]),
                    "drag_before": float(env.drag_before[index]),
                    "drag_after": float(env.drag_after[index]),
                    "completion_step": int(completion_step[index]),
                    "mean_certificate_violation": float(
                        violation_sum[index] / max(decision_count[index], 1.0)
                    ),
                    "nominal_selection_fraction_during_event": float(
                        nominal_selection_count_event[index]
                        / max(effectiveness_count_event[index], 1.0)
                    ),
                    "nonnominal_selection_fraction_outside_event": float(
                        nonnominal_selection_count_nominal[index]
                        / max(effectiveness_count_nominal[index], 1.0)
                    ),
                    "mean_innovation_during_event": float(
                        innovation_sum_event[index]
                        / max(effectiveness_count_event[index], 1.0)
                    ),
                    "mean_innovation_outside_event": float(
                        innovation_sum_nominal[index]
                        / max(effectiveness_count_nominal[index], 1.0)
                    ),
                    "maximum_innovation": float(maximum_innovation[index]),
                }
            )
        result["scenario_details"] = details
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--scenarios", type=int, default=48)
    parser.add_argument("--stress-levels", type=float, nargs="+", default=[0.25, 0.27])
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--nominal-horizon", type=int, default=500)
    parser.add_argument("--maximum-horizon", type=int, default=900)
    parser.add_argument("--acceleration-bound", type=float, default=2.0)
    parser.add_argument("--torque-reserve", type=float, default=2.0)
    parser.add_argument("--confidence-scale", type=float, default=0.75)
    parser.add_argument("--clf-rate", type=float, default=1.0)
    parser.add_argument("--clf-level", type=float, default=0.5)
    parser.add_argument("--clf-remainder-bound", type=float, default=0.08037916898668823)
    parser.add_argument("--maximum-recovery-jerk", type=float, default=1.0e6)
    parser.add_argument("--recovery-dwell-decisions", type=int, default=0)
    parser.add_argument(
        "--effectiveness-compensation", choices=("none", "oracle"), default="none"
    )
    parser.add_argument("--simulation-timestep", type=float, default=None)
    parser.add_argument("--risk-speed-constraint", action="store_true")
    parser.add_argument("--risk-error-threshold", type=float, default=0.12)
    parser.add_argument("--risk-safety-gain", type=float, default=1.3)
    parser.add_argument("--predictive-backup-constraint", action="store_true")
    parser.add_argument("--predictive-horizon-steps", type=int, default=8)
    parser.add_argument("--predictive-error-limit", type=float, default=0.5)
    parser.add_argument("--predictive-grid-size", type=int, default=21)
    parser.add_argument("--predictive-disturbance-scale", type=float, default=0.0)
    parser.add_argument("--barrier-error-limit", type=float, default=None)
    parser.add_argument("--barrier-frequency", type=float, default=4.0)
    parser.add_argument("--barrier-damping", type=float, default=1.0)
    parser.add_argument("--barrier-remainder-bound", type=float, default=0.0)
    parser.add_argument(
        "--recovery-mode",
        choices=("hard_brake", "torque_relief"),
        default="hard_brake",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "assets" / "seven_link_arm.xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "certificate_governor_screen.json",
    )
    args = parser.parse_args()
    results: dict[str, object] = {
        "seed": args.seed,
        "configuration": vars(args) | {"model": str(args.model), "output": str(args.output)},
        "stress_levels": {},
    }
    for stress in args.stress_levels:
        common = dict(
            model_path=args.model,
            seed=args.seed,
            scenarios=args.scenarios,
            stress_level=stress,
            nominal_horizon=args.nominal_horizon,
            maximum_horizon=args.maximum_horizon,
            simulation_timestep=args.simulation_timestep,
        )
        fixed = run_method(
            **common,
            method="fixed_speed",
            decision_interval=args.decision_interval,
            error_threshold=0.12,
            safety_gain=1.3,
            backlog_gain=1.5,
            recovery_minimum_speed=0.05,
            maximum_progress_acceleration=args.acceleration_bound,
        )
        heuristic = run_method(
            **common,
            method="elastic_governor",
            decision_interval=args.decision_interval,
            error_threshold=0.12,
            safety_gain=1.3,
            backlog_gain=1.5,
            recovery_minimum_speed=0.05,
            maximum_progress_acceleration=args.acceleration_bound,
        )
        certificate = run_certificate_method(
            **common,
            decision_interval=args.decision_interval,
            acceleration_bound=args.acceleration_bound,
            torque_reserve=args.torque_reserve,
            confidence_scale=args.confidence_scale,
            clf_rate=args.clf_rate,
            clf_level=args.clf_level,
            clf_remainder_bound=args.clf_remainder_bound,
            maximum_recovery_jerk=args.maximum_recovery_jerk,
            recovery_mode=args.recovery_mode,
            recovery_dwell_decisions=args.recovery_dwell_decisions,
            effectiveness_compensation=args.effectiveness_compensation,
            risk_speed_constraint=args.risk_speed_constraint,
            risk_error_threshold=args.risk_error_threshold,
            risk_safety_gain=args.risk_safety_gain,
            predictive_backup_constraint=args.predictive_backup_constraint,
            predictive_horizon_steps=args.predictive_horizon_steps,
            predictive_error_limit=args.predictive_error_limit,
            predictive_grid_size=args.predictive_grid_size,
            predictive_disturbance_scale=args.predictive_disturbance_scale,
            barrier_error_limit=args.barrier_error_limit,
            barrier_frequency=args.barrier_frequency,
            barrier_damping=args.barrier_damping,
            barrier_remainder_bound=args.barrier_remainder_bound,
        )
        results["stress_levels"][str(stress)] = {  # type: ignore[index]
            "fixed_speed": fixed,
            "heuristic_second_order": heuristic,
            "certificate_second_order": certificate,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
