"""Seven-link tracking task with unknown payload and Coulomb friction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from adaptive_identification import RobustLinearRLS, RobustScalarRLS

JOINTS = 7
from validate_seven_link_mujoco import inverse_torque, set_payload


FRICTION_PROFILE = np.linspace(0.55, 0.20, JOINTS)
DRAG_PROFILE = np.linspace(5.0, 2.0, JOINTS)


def reference(time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amplitude = np.linspace(0.28, 0.16, JOINTS)
    frequency = np.linspace(0.55, 1.15, JOINTS)
    phase = np.linspace(0.0, np.pi, JOINTS)
    angle = frequency * time + phase
    q = amplitude * np.sin(angle)
    dq = amplitude * frequency * np.cos(angle)
    ddq = -amplitude * np.square(frequency) * np.sin(angle)
    return q, dq, ddq


def evaluate(
    path: Path,
    payloads: np.ndarray,
    friction_scales: np.ndarray,
    drag_scales: np.ndarray,
    initial_offsets: np.ndarray,
    method: str,
    horizon: int,
    seed: int,
    disturbance_scale: float,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    inverse_model = mujoco.MjModel.from_xml_path(str(path))
    inverse_data = mujoco.MjData(inverse_model)
    inverse_payload = mujoco.mj_name2id(
        inverse_model, mujoco.mjtObj.mjOBJ_BODY, "payload"
    )
    inverse_mass = float(inverse_model.body_mass[inverse_payload])
    inverse_inertia = inverse_model.body_inertia[inverse_payload].copy()
    plants: list[mujoco.MjModel] = []
    samples: list[mujoco.MjData] = []
    estimators: list[RobustScalarRLS | RobustLinearRLS] = []
    q0, dq0, _ = reference(0.0)
    for payload, friction, offset in zip(
        payloads, friction_scales, initial_offsets
    ):
        plant = mujoco.MjModel.from_xml_path(str(path))
        sample = mujoco.MjData(plant)
        body = mujoco.mj_name2id(
            plant, mujoco.mjtObj.mjOBJ_BODY, "payload"
        )
        set_payload(
            plant,
            body,
            float(plant.body_mass[body]),
            plant.body_inertia[body].copy(),
            0.05 + float(payload),
        )
        plant.dof_frictionloss[:] = FRICTION_PROFILE * friction
        sample.qpos[:] = q0 + offset
        sample.qvel[:] = dq0
        mujoco.mj_forward(plant, sample)
        plants.append(plant)
        samples.append(sample)
        estimators.append(
            RobustLinearRLS(
                initial=np.array([1.0, 1.0]),
                lower=np.array([0.4, 0.5]),
                upper=np.array([2.0, 1.8]),
                prior_information=0.5,
                forgetting=0.995,
                huber_delta=1.0,
            )
            if method == "joint_rls"
            else RobustScalarRLS(
                initial=1.0,
                prior_information=0.5,
                forgetting=0.995,
                huber_delta=1.0,
            )
        )
    squared_error = np.zeros(payloads.size)
    maximum_error = np.zeros(payloads.size)
    energy = np.zeros(payloads.size)
    estimate_error = np.zeros(payloads.size)
    friction_error = np.zeros(payloads.size)
    kp = np.linspace(18.0, 10.0, JOINTS)
    kd = np.linspace(6.0, 3.5, JOINTS)
    dt = float(inverse_model.opt.timestep)
    for step in range(horizon):
        q_ref, dq_ref, ddq_ref = reference(step * dt)
        for index, (plant, sample, estimator) in enumerate(
            zip(plants, samples, estimators)
        ):
            q = sample.qpos.copy()
            dq = sample.qvel.copy()
            desired_acceleration = (
                ddq_ref - kp * (q - q_ref) - kd * (dq - dq_ref)
            )
            if method == "nominal":
                control_payload = 1.0
            elif method in (
                "oracle_payload", "oracle_manual", "oracle_full",
                "oracle_augmented",
            ):
                control_payload = float(payloads[index])
            elif method == "joint_rls":
                control_payload = float(estimator.value[0])
            else:
                control_payload = estimator.value
            set_payload(
                inverse_model,
                inverse_payload,
                inverse_mass,
                inverse_inertia,
                0.05 + control_payload,
            )
            if method in ("oracle_full", "oracle_augmented"):
                control_friction = float(friction_scales[index])
            elif method == "joint_rls":
                control_friction = float(estimator.value[1])
            else:
                control_friction = 0.0
            inverse_model.dof_frictionloss[:] = (
                FRICTION_PROFILE * control_friction
            )
            torque = inverse_torque(
                    inverse_model,
                    inverse_data,
                    q,
                    dq,
                    desired_acceleration,
            )
            if method == "oracle_manual":
                torque = torque + (
                    FRICTION_PROFILE
                    * friction_scales[index]
                    * np.sign(dq)
                )
            if method == "oracle_augmented":
                torque = torque + (
                    DRAG_PROFILE * drag_scales[index] * dq * np.abs(dq)
                )
            torque = np.clip(torque, -45.0, 45.0)
            sample.ctrl[:] = torque
            sample.qfrc_applied[:] = (
                -DRAG_PROFILE * drag_scales[index] * dq * np.abs(dq)
            )
            if 200 <= step < 250 and disturbance_scale > 0.0:
                sample.qfrc_applied[:3] = disturbance_scale * np.array(
                    [3.0, -2.0, 1.5]
                )
            mujoco.mj_forward(plant, sample)
            if method == "robust_rls":
                measured_acceleration = sample.qacc + rng.normal(
                    scale=0.03, size=JOINTS
                )
                inverse_model.dof_frictionloss[:] = 0.0
                set_payload(
                    inverse_model,
                    inverse_payload,
                    inverse_mass,
                    inverse_inertia,
                    0.05,
                )
                tau_zero = inverse_torque(
                    inverse_model,
                    inverse_data,
                    q,
                    dq,
                    measured_acceleration,
                )
                set_payload(
                    inverse_model,
                    inverse_payload,
                    inverse_mass,
                    inverse_inertia,
                    1.05,
                )
                tau_unit = inverse_torque(
                    inverse_model,
                    inverse_data,
                    q,
                    dq,
                    measured_acceleration,
                )
                estimator.update(tau_unit - tau_zero, torque - tau_zero)
            elif method == "joint_rls":
                measured_acceleration = sample.qacc + rng.normal(
                    scale=0.03, size=JOINTS
                )
                inverse_model.dof_frictionloss[:] = 0.0
                set_payload(
                    inverse_model, inverse_payload, inverse_mass,
                    inverse_inertia, 0.05
                )
                tau_zero = inverse_torque(
                    inverse_model, inverse_data, q, dq,
                    measured_acceleration
                )
                set_payload(
                    inverse_model, inverse_payload, inverse_mass,
                    inverse_inertia, 1.05
                )
                tau_payload = inverse_torque(
                    inverse_model, inverse_data, q, dq,
                    measured_acceleration
                )
                set_payload(
                    inverse_model, inverse_payload, inverse_mass,
                    inverse_inertia, 0.05
                )
                inverse_model.dof_frictionloss[:] = FRICTION_PROFILE
                tau_friction = inverse_torque(
                    inverse_model, inverse_data, q, dq,
                    measured_acceleration
                )
                regressor = np.stack(
                    (tau_payload - tau_zero, tau_friction - tau_zero),
                    axis=-1,
                )
                estimator.update(regressor, torque - tau_zero)
            error = q - q_ref
            norm = np.linalg.norm(error)
            squared_error[index] += np.dot(error, error)
            maximum_error[index] = max(maximum_error[index], norm)
            energy[index] += np.dot(torque, torque)
            mujoco.mj_step(plant, sample)
    for index, estimator in enumerate(estimators):
        if method == "joint_rls":
            estimate_error[index] = abs(
                estimator.value[0] - payloads[index]
            )
            friction_error[index] = abs(
                estimator.value[1] - friction_scales[index]
            )
        else:
            estimate_error[index] = abs(estimator.value - payloads[index])
    failure = maximum_error > 0.75
    tail = (payloads <= 0.65) | (payloads >= 1.75)
    return {
        "tracking_rmse": float(np.sqrt(squared_error.mean() / horizon)),
        "median_max_error": float(np.median(maximum_error)),
        "failure_rate": float(failure.mean()),
        "tail_failure_rate": float(failure[tail].mean()),
        "mean_torque_energy": float(energy.mean()),
        "final_payload_mae": float(estimate_error.mean()),
        "final_friction_mae": float(friction_error.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--scenarios", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--disturbance-scale", type=float, default=0.0)
    parser.add_argument(
        "--model", type=Path, default=Path("assets/seven_link_arm.xml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tracking_task_seed131.json"),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    payloads = rng.uniform(0.4, 2.0, args.scenarios)
    friction_scales = rng.uniform(0.5, 1.8, args.scenarios)
    drag_scales = rng.uniform(0.5, 1.8, args.scenarios)
    initial_offsets = rng.uniform(-0.12, 0.12, (args.scenarios, JOINTS))
    result = {
        "seed": args.seed,
        "scenarios": args.scenarios,
        "horizon": args.horizon,
        "disturbance_scale": args.disturbance_scale,
        "methods": {},
    }
    for method in (
        "nominal",
        "robust_rls",
        "joint_rls",
        "oracle_payload",
        "oracle_manual",
        "oracle_full",
        "oracle_augmented",
    ):
        result["methods"][method] = evaluate(
            args.model,
            payloads,
            friction_scales,
            drag_scales,
            initial_offsets,
            method,
            args.horizon,
            args.seed + 2000,
            args.disturbance_scale,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
