"""Validate the seven-DoF MuJoCo plant and payload inverse dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def set_payload(
    model: mujoco.MjModel,
    payload_id: int,
    nominal_mass: float,
    nominal_inertia: np.ndarray,
    mass: float,
) -> None:
    scale = mass / nominal_mass
    model.body_mass[payload_id] = mass
    model.body_inertia[payload_id] = nominal_inertia * scale


def inverse_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
) -> np.ndarray:
    data.qpos[:] = q
    data.qvel[:] = dq
    data.qacc[:] = ddq
    mujoco.mj_inverse(model, data)
    return data.qfrc_inverse.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("assets/seven_link_arm.xml"),
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/seven_link_validation.json"),
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    payload_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "payload"
    )
    nominal_mass = float(model.body_mass[payload_id])
    nominal_inertia = model.body_inertia[payload_id].copy()
    rng = np.random.default_rng(args.seed)
    errors = []
    affine_errors = []
    saturated = []
    for _ in range(args.samples):
        q = rng.uniform(-0.6, 0.6, model.nq)
        dq = rng.uniform(-0.8, 0.8, model.nv)
        ddq = rng.uniform(-1.5, 1.5, model.nv)
        payload = rng.uniform(0.4, 2.0)
        set_payload(
            model,
            payload_id,
            nominal_mass,
            nominal_inertia,
            0.05,
        )
        tau_zero = inverse_torque(model, data, q, dq, ddq)
        set_payload(
            model,
            payload_id,
            nominal_mass,
            nominal_inertia,
            1.05,
        )
        tau_unit = inverse_torque(model, data, q, dq, ddq)
        predicted = tau_zero + payload * (tau_unit - tau_zero)
        set_payload(
            model,
            payload_id,
            nominal_mass,
            nominal_inertia,
            0.05 + payload,
        )
        actual = inverse_torque(model, data, q, dq, ddq)
        affine_errors.append(np.max(np.abs(predicted - actual)))

        data.qpos[:] = q
        data.qvel[:] = dq
        saturated.append(np.any(np.abs(actual) > 45.0))
        data.ctrl[:] = actual
        mujoco.mj_forward(model, data)
        errors.append(np.linalg.norm(data.qacc - ddq))

    result = {
        "degrees_of_freedom": int(model.nv),
        "actuators": int(model.nu),
        "timestep": float(model.opt.timestep),
        "payload_range": [0.4, 2.0],
        "inverse_dynamics_affine_max_error": float(
            np.max(affine_errors)
        ),
        "inverse_forward_acceleration_rmse": float(
            np.sqrt(np.mean(np.square(errors)))
        ),
        "requested_torque_saturation_fraction": float(
            np.mean(saturated)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
