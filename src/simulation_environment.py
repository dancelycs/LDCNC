"""Seven-link MuJoCo environment for the reported control experiments."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from adaptive_identification import RobustLinearRLS
from screen_tracking_task import DRAG_PROFILE, FRICTION_PROFILE, reference
from validate_seven_link_mujoco import inverse_torque, set_payload

MIN_SPEED = 0.6
MAX_SPEED = 1.4
TARGET_SPEED = 1.0
CALIBRATED_BETA = 0.3074275769025972
JOINTS = 7

class ProgressConstrainedEnvironment:
    """Seven-link tracking with random drag changes and fixed terminal progress."""

    def __init__(
        self,
        path: Path,
        count: int,
        seed: int,
        episode_length: int,
        strong_switch: bool = False,
        stress_level: float = 0.0,
        enforce_terminal_progress: bool = True,
        minimum_speed: float = MIN_SPEED,
        maximum_speed: float = MAX_SPEED,
        maximum_progress_acceleration: float | None = None,
        include_progress_acceleration: bool = False,
        effectiveness_compensation: str = "none",
        estimator_torque_mode: str = "commanded",
        effectiveness_window_steps: int | None = None,
        simulation_timestep: float | None = None,
        fault_profile: str = "pulse",
        freeze_physical_on_external_fault: bool = False,
        acceleration_noise_std: float = 0.03,
        actuator_event_offset_steps: int = 0,
        torque_quantization: float = 0.0,
    ) -> None:
        self.count = count
        self.episode_length = episode_length
        self.strong_switch = strong_switch
        self.stress_level = stress_level
        self.enforce_terminal_progress = enforce_terminal_progress
        self.minimum_speed = minimum_speed
        self.maximum_speed = maximum_speed
        self.maximum_progress_acceleration = maximum_progress_acceleration
        self.include_progress_acceleration = include_progress_acceleration
        if fault_profile not in ("pulse", "ramp", "intermittent", "persistent"):
            raise ValueError(
                "fault_profile must be 'pulse', 'ramp', 'intermittent', or "
                "'persistent'"
            )
        self.fault_profile = fault_profile
        self.freeze_physical_on_external_fault = (
            freeze_physical_on_external_fault
        )
        if acceleration_noise_std < 0.0 or torque_quantization < 0.0:
            raise ValueError("noise and quantization parameters must be nonnegative")
        self.acceleration_noise_std = float(acceleration_noise_std)
        self.actuator_event_offset_steps = int(actuator_event_offset_steps)
        self.torque_quantization = float(torque_quantization)
        if effectiveness_compensation not in ("none", "oracle", "external"):
            raise ValueError(
                "effectiveness_compensation must be 'none', 'oracle', or "
                "'external'"
            )
        self.effectiveness_compensation = effectiveness_compensation
        if estimator_torque_mode not in (
            "commanded",
            "oracle_applied",
            "causal_effectiveness",
            "orthogonal_effectiveness",
            "orthogonal_isolation",
            "orthogonal_unbiased",
            "orthogonal_block",
            "orthogonal_block_unbiased",
            "oracle_freeze",
            "augmented_joint",
            "dual_timescale",
            "event_active_orthogonal",
            "external_effectiveness",
            "external_effectiveness_unbiased",
        ):
            raise ValueError(
                "estimator_torque_mode must be 'commanded', 'oracle_applied', "
                "'causal_effectiveness', 'orthogonal_effectiveness', or "
                "'orthogonal_isolation', 'orthogonal_unbiased', or "
                "'orthogonal_block', 'orthogonal_block_unbiased', or "
                "'oracle_freeze', 'augmented_joint', 'dual_timescale', or "
                "'event_active_orthogonal', 'external_effectiveness', or "
                "'external_effectiveness_unbiased'"
            )
        self.estimator_torque_mode = estimator_torque_mode
        self.rng = np.random.default_rng(seed)
        self.models = [mujoco.MjModel.from_xml_path(str(path)) for _ in range(count)]
        if simulation_timestep is not None:
            for model in self.models:
                model.opt.timestep = simulation_timestep
        self.data = [mujoco.MjData(model) for model in self.models]
        self.payload_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
            for model in self.models
        ]
        self.base_mass = [float(m.body_mass[b]) for m, b in zip(self.models, self.payload_ids)]
        self.base_inertia = [m.body_inertia[b].copy() for m, b in zip(self.models, self.payload_ids)]
        self.inverse_model = mujoco.MjModel.from_xml_path(str(path))
        if simulation_timestep is not None:
            self.inverse_model.opt.timestep = simulation_timestep
        self.inverse_data = mujoco.MjData(self.inverse_model)
        self.inverse_payload = mujoco.mj_name2id(
            self.inverse_model, mujoco.mjtObj.mjOBJ_BODY, "payload"
        )
        self.inverse_mass = float(self.inverse_model.body_mass[self.inverse_payload])
        self.inverse_inertia = self.inverse_model.body_inertia[self.inverse_payload].copy()
        self.kp = np.linspace(18.0, 10.0, JOINTS)
        self.kd = np.linspace(6.0, 3.5, JOINTS)
        self.dt = float(self.inverse_model.opt.timestep)
        self.effectiveness_window_steps = (
            max(1, int(round(0.10 / self.dt)))
            if effectiveness_window_steps is None
            else int(effectiveness_window_steps)
        )
        if self.effectiveness_window_steps <= 0:
            raise ValueError("effectiveness_window_steps must be positive")
        self.estimator_forgetting = 0.98 ** (self.dt / 0.01)
        self.actuator_event_steps = max(1, int(round(0.75 / self.dt)))
        self.target_progress = TARGET_SPEED * episode_length * self.dt
        self.payload = np.ones(count)
        self.friction = np.ones(count)
        self.drag_before = np.ones(count)
        self.drag_after = np.ones(count)
        self.switch_step = np.zeros(count, dtype=np.int64)
        self.actuator_effectiveness = np.ones(count)
        self.estimated_effectiveness = np.ones(count)
        self.true_effectiveness = np.ones(count)
        self.identification_null_direction = np.zeros((count, JOINTS))
        self.effectiveness_numerator = np.zeros(count)
        self.effectiveness_denominator = np.zeros(count)
        self.effectiveness_block_count = np.zeros(count, dtype=np.int64)
        self.effectiveness_regressor_blocks: list[list[np.ndarray]] = [
            [] for _ in range(count)
        ]
        self.effectiveness_torque_blocks: list[list[np.ndarray]] = [
            [] for _ in range(count)
        ]
        self.effectiveness_zero_blocks: list[list[np.ndarray]] = [
            [] for _ in range(count)
        ]
        self.age = np.zeros(count, dtype=np.int64)
        self.phase = np.zeros(count)
        self.innovation = np.zeros(count)
        self.maximum_error = np.zeros(count)
        self.last_speed = np.ones(count)
        self.last_energy = np.zeros(count)
        self.estimators: list[RobustLinearRLS] = []
        self.augmented_estimators: list[RobustLinearRLS] = []
        self.slow_physical_estimators: list[RobustLinearRLS] = []
        self.fast_fault_estimators: list[RobustLinearRLS] = []
        self.reset(np.ones(count, dtype=bool))

    def reset(self, mask: np.ndarray) -> None:
        q_ref, dq_ref, _ = reference(0.0)
        if not self.estimators:
            self.estimators = [None] * self.count  # type: ignore[list-item]
        if not self.augmented_estimators:
            self.augmented_estimators = [None] * self.count  # type: ignore[list-item]
        if not self.slow_physical_estimators:
            self.slow_physical_estimators = [None] * self.count  # type: ignore[list-item]
        if not self.fast_fault_estimators:
            self.fast_fault_estimators = [None] * self.count  # type: ignore[list-item]
        for index in np.flatnonzero(mask):
            self.payload[index] = self.rng.uniform(0.4, 2.0)
            self.friction[index] = self.rng.uniform(0.5, 1.8)
            self.drag_before[index] = self.rng.uniform(0.5, 1.8)
            if self.strong_switch:
                self.drag_after[index] = (
                    self.rng.uniform(1.55, 2.0)
                    if self.drag_before[index] <= 1.15
                    else self.rng.uniform(0.10, 0.45)
                )
            else:
                self.drag_after[index] = self.rng.uniform(0.5, 1.8)
            self.switch_step[index] = self.rng.integers(
                int(0.35 * self.episode_length),
                int(0.65 * self.episode_length) + 1,
            )
            self.actuator_effectiveness[index] = self.rng.uniform(
                max(0.25, 1.0 - 0.60 * self.stress_level),
                max(0.30, 1.0 - 0.30 * self.stress_level),
            )
            self.true_effectiveness[index] = 1.0
            model = self.models[index]
            set_payload(
                model,
                self.payload_ids[index],
                self.base_mass[index],
                self.base_inertia[index],
                0.05 + self.payload[index],
            )
            model.dof_frictionloss[:] = FRICTION_PROFILE * self.friction[index]
            sample = self.data[index]
            sample.qpos[:] = q_ref + self.rng.uniform(-0.12, 0.12, JOINTS)
            sample.qvel[:] = dq_ref
            sample.time = 0.0
            sample.ctrl[:] = 0.0
            sample.qfrc_applied[:] = 0.0
            mujoco.mj_forward(model, sample)
            self.estimators[index] = RobustLinearRLS(
                initial=np.array([1.0, 1.0, 1.0]),
                lower=np.array([0.4, 0.5, 0.0]),
                upper=np.array([2.0, 1.8, 2.0]),
                prior_information=0.5,
                forgetting=self.estimator_forgetting,
                huber_delta=1.0,
            )
            self.augmented_estimators[index] = RobustLinearRLS(
                initial=np.array([1.0, 1.0, 1.0, 1.0]),
                lower=np.array([0.4, 0.5, 0.0, 0.25]),
                upper=np.array([2.0, 1.8, 2.0, 1.0]),
                prior_information=0.5,
                forgetting=self.estimator_forgetting,
                huber_delta=1.0,
            )
            self.slow_physical_estimators[index] = RobustLinearRLS(
                initial=np.array([1.0, 1.0]),
                lower=np.array([0.4, 0.5]),
                upper=np.array([2.0, 1.8]),
                prior_information=0.5,
                forgetting=1.0,
                huber_delta=1.0,
            )
            self.fast_fault_estimators[index] = RobustLinearRLS(
                initial=np.array([1.0, 1.0]),
                lower=np.array([0.0, 0.25]),
                upper=np.array([2.0, 1.0]),
                prior_information=0.5,
                forgetting=self.estimator_forgetting,
                huber_delta=1.0,
            )
            self.age[index] = 0
            self.phase[index] = 0.0
            self.innovation[index] = 0.0
            self.estimated_effectiveness[index] = 1.0
            self.identification_null_direction[index] = 0.0
            self.effectiveness_numerator[index] = 0.0
            self.effectiveness_denominator[index] = 0.0
            self.effectiveness_block_count[index] = 0
            self.effectiveness_regressor_blocks[index].clear()
            self.effectiveness_torque_blocks[index].clear()
            self.effectiveness_zero_blocks[index].clear()
            self.maximum_error[index] = 0.0
            self.last_speed[index] = TARGET_SPEED
            self.last_energy[index] = 0.0

    def state(self) -> np.ndarray:
        context = np.empty((self.count, 5), dtype=np.float32)
        for index, (sample, estimator) in enumerate(zip(self.data, self.estimators)):
            q_ref, dq_base, _ = reference(self.phase[index])
            error = np.linalg.norm(sample.qpos - q_ref)
            velocity_error = np.linalg.norm(
                sample.qvel - self.last_speed[index] * dq_base
            )
            remaining = max(self.episode_length - self.age[index], 1)
            required_speed = (
                self.target_progress - self.phase[index]
            ) / (remaining * self.dt)
            context[index] = np.array(
                [
                    np.clip(error / 0.4, 0.0, 2.0) - 1.0,
                    np.clip(velocity_error / 0.8, 0.0, 2.0) - 1.0,
                    np.clip(self.innovation[index] / 0.10 - 1.0, -1.0, 1.0),
                    2.0 * self.age[index] / self.episode_length - 1.0,
                    np.clip((required_speed - TARGET_SPEED) / 0.4, -1.0, 1.0),
                ]
            )
        return context

    def feasible_speed(self, command: np.ndarray) -> np.ndarray:
        """Project commands onto the viability interval for terminal progress."""
        if not self.enforce_terminal_progress:
            speed = np.clip(command, self.minimum_speed, self.maximum_speed)
        else:
            remaining = self.episode_length - self.age
            later = np.maximum(remaining - 1, 0)
            distance = self.target_progress - self.phase
            lower = (distance - self.maximum_speed * later * self.dt) / self.dt
            upper = (distance - self.minimum_speed * later * self.dt) / self.dt
            lower = np.maximum(lower, self.minimum_speed)
            upper = np.minimum(upper, self.maximum_speed)
            speed = np.clip(command, lower, upper)
        if self.maximum_progress_acceleration is not None:
            increment = self.maximum_progress_acceleration * self.dt
            speed = np.clip(
                speed,
                self.last_speed - increment,
                self.last_speed + increment,
            )
        return speed

    def step(
        self,
        raw_action: np.ndarray,
        torque_override: np.ndarray | None = None,
        reference_override: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        identification_probe: np.ndarray | None = None,
        estimator_effectiveness_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if torque_override is not None and torque_override.shape != (self.count, JOINTS):
            raise ValueError("torque_override must have shape (count, joints)")
        if identification_probe is not None and identification_probe.shape != (
            self.count,
            JOINTS,
        ):
            raise ValueError("identification_probe must have shape (count, joints)")
        if (
            estimator_effectiveness_override is not None
            and estimator_effectiveness_override.shape != (self.count,)
        ):
            raise ValueError(
                "estimator_effectiveness_override must have shape (count,)"
            )
        if reference_override is not None:
            if len(reference_override) != 3 or any(
                value.shape != (self.count, JOINTS)
                for value in reference_override
            ):
                raise ValueError(
                    "reference_override must contain position, velocity, and "
                    "acceleration arrays with shape (count, joints)"
                )
        midpoint = 0.5 * (self.minimum_speed + self.maximum_speed)
        half_range = 0.5 * (self.maximum_speed - self.minimum_speed)
        command = midpoint + half_range * np.tanh(raw_action[:, 0])
        speed = self.feasible_speed(command)
        progress_acceleration = (speed - self.last_speed) / self.dt
        cost = np.empty(self.count, dtype=np.float64)
        for index, (model, sample, estimator) in enumerate(
            zip(self.models, self.data, self.estimators)
        ):
            if reference_override is None:
                q_ref, dq_base, ddq_base = reference(self.phase[index])
                dq_ref = speed[index] * dq_base
                ddq_ref = speed[index] ** 2 * ddq_base
                if self.include_progress_acceleration:
                    ddq_ref = ddq_ref + progress_acceleration[index] * dq_base
            else:
                q_ref = reference_override[0][index]
                dq_ref = reference_override[1][index]
                ddq_ref = reference_override[2][index]
            q = sample.qpos.copy()
            dq = sample.qvel.copy()
            true_drag = (
                self.drag_before[index]
                if self.age[index] < self.switch_step[index]
                else self.drag_after[index]
            )
            covariance = np.linalg.inv(estimator.information + 1.0e-10 * np.eye(3))
            standard_error = np.sqrt(max(covariance[2, 2], 0.0))
            lower_drag = float(
                np.clip(
                    estimator.value[2] - CALIBRATED_BETA * standard_error,
                    0.0,
                    estimator.value[2],
                )
            )
            error = q - q_ref
            velocity_error = dq - dq_ref
            desired_acceleration = ddq_ref - self.kp * error - self.kd * velocity_error
            set_payload(
                self.inverse_model,
                self.inverse_payload,
                self.inverse_mass,
                self.inverse_inertia,
                0.05 + estimator.value[0],
            )
            self.inverse_model.dof_frictionloss[:] = FRICTION_PROFILE * estimator.value[1]
            drag_compensation = DRAG_PROFILE * lower_drag * dq * np.abs(dq)
            requested_torque = (
                inverse_torque(
                    self.inverse_model, self.inverse_data, q, dq, desired_acceleration
                )
                + drag_compensation
            )
            event_offset = int(
                self.age[index] - self.switch_step[index]
                - self.actuator_event_offset_steps
            )
            in_actuator_event = event_offset >= 0
            if self.fault_profile == "pulse":
                in_actuator_event = in_actuator_event and (
                    event_offset < self.actuator_event_steps
                )
                effectiveness = (
                    self.actuator_effectiveness[index]
                    if in_actuator_event else 1.0
                )
            elif self.fault_profile == "ramp":
                if in_actuator_event and event_offset < self.actuator_event_steps:
                    fraction = (event_offset + 1) / self.actuator_event_steps
                    effectiveness = 1.0 - fraction * (
                        1.0 - self.actuator_effectiveness[index]
                    )
                elif in_actuator_event and event_offset < 2 * self.actuator_event_steps:
                    fraction = (
                        event_offset - self.actuator_event_steps + 1
                    ) / self.actuator_event_steps
                    effectiveness = self.actuator_effectiveness[index] + fraction * (
                        1.0 - self.actuator_effectiveness[index]
                    )
                else:
                    effectiveness = 1.0
                    in_actuator_event = False
            elif self.fault_profile == "intermittent":
                intermittent_duration = 2 * self.actuator_event_steps
                in_actuator_event = in_actuator_event and (
                    event_offset < intermittent_duration
                )
                half_period = max(1, int(round(0.075 / self.dt)))
                fault_on = (event_offset // half_period) % 2 == 0
                effectiveness = (
                    self.actuator_effectiveness[index]
                    if in_actuator_event and fault_on else 1.0
                )
                in_actuator_event = in_actuator_event and fault_on
            else:
                effectiveness = (
                    self.actuator_effectiveness[index]
                    if in_actuator_event else 1.0
                )
            self.true_effectiveness[index] = effectiveness
            if self.effectiveness_compensation == "oracle":
                compensation = effectiveness
            elif (
                self.effectiveness_compensation == "external"
                and estimator_effectiveness_override is not None
            ):
                compensation = float(
                    np.clip(estimator_effectiveness_override[index], 0.25, 1.0)
                )
            else:
                compensation = 1.0
            base_torque = (
                requested_torque / compensation
                if torque_override is None
                else torque_override[index]
            )
            if identification_probe is not None:
                base_torque = base_torque + identification_probe[index]
            torque = np.clip(
                base_torque,
                -45.0,
                45.0,
            )
            if self.torque_quantization > 0.0:
                torque = self.torque_quantization * np.round(
                    torque / self.torque_quantization
                )
                torque = np.clip(torque, -45.0, 45.0)
            sample.ctrl[:] = effectiveness * torque
            sample.qfrc_applied[:] = -DRAG_PROFILE * true_drag * dq * np.abs(dq)
            mujoco.mj_forward(model, sample)
            measured_acceleration = sample.qacc + self.rng.normal(
                scale=self.acceleration_noise_std, size=JOINTS
            )
            set_payload(
                self.inverse_model,
                self.inverse_payload,
                self.inverse_mass,
                self.inverse_inertia,
                0.05,
            )
            self.inverse_model.dof_frictionloss[:] = 0.0
            tau_zero = inverse_torque(
                self.inverse_model, self.inverse_data, q, dq, measured_acceleration
            )
            set_payload(
                self.inverse_model,
                self.inverse_payload,
                self.inverse_mass,
                self.inverse_inertia,
                1.05,
            )
            tau_payload = inverse_torque(
                self.inverse_model, self.inverse_data, q, dq, measured_acceleration
            )
            set_payload(
                self.inverse_model,
                self.inverse_payload,
                self.inverse_mass,
                self.inverse_inertia,
                0.05,
            )
            self.inverse_model.dof_frictionloss[:] = FRICTION_PROFILE
            tau_friction = inverse_torque(
                self.inverse_model, self.inverse_data, q, dq, measured_acceleration
            )
            regressor = np.stack(
                (
                    tau_payload - tau_zero,
                    tau_friction - tau_zero,
                    DRAG_PROFILE * dq * np.abs(dq),
                ),
                axis=-1,
            )
            # The left-null direction is orthogonal to every slow physical
            # parameter column and can therefore excite actuator
            # effectiveness without confounding payload, friction, or drag.
            left_vectors, singular_values, _ = np.linalg.svd(
                regressor, full_matrices=True
            )
            rank = int(
                np.sum(
                    singular_values
                    > max(regressor.shape)
                    * np.finfo(float).eps
                    * max(float(singular_values[0]), 1.0)
                )
            )
            if rank < JOINTS:
                direction = left_vectors[:, rank]
                infinity_norm = float(np.max(np.abs(direction)))
                if infinity_norm > 1.0e-12:
                    self.identification_null_direction[index] = (
                        direction / infinity_norm
                    )
            update_physical_estimator = True
            active_probe_sample = (
                identification_probe is not None
                and np.linalg.norm(identification_probe[index]) > 1.0e-12
            )
            if self.estimator_torque_mode in (
                "orthogonal_effectiveness",
                "orthogonal_isolation",
                "orthogonal_unbiased",
                "orthogonal_block",
                "orthogonal_block_unbiased",
            ) or (
                self.estimator_torque_mode == "event_active_orthogonal"
                and active_probe_sample
            ):
                # Eliminate the three slow physical parameters before
                # estimating the scalar actuator effectiveness.  From
                # tau_zero = eta*tau_command - Phi*theta, projection onto the
                # orthogonal complement of col(Phi) gives a scalar least-
                # squares estimate that is independent of theta.
                projected_torque = torque - regressor @ np.linalg.lstsq(
                    regressor, torque, rcond=None
                )[0]
                projected_zero = tau_zero - regressor @ np.linalg.lstsq(
                    regressor, tau_zero, rcond=None
                )[0]
                projected_energy = float(
                    np.dot(projected_torque, projected_torque)
                )
                if self.estimator_torque_mode in (
                    "orthogonal_block",
                    "orthogonal_block_unbiased",
                ):
                    self.effectiveness_numerator[index] += float(
                        np.dot(projected_torque, projected_zero)
                    )
                    self.effectiveness_denominator[index] += projected_energy
                    self.effectiveness_block_count[index] += 1
                    self.effectiveness_regressor_blocks[index].append(
                        regressor.copy()
                    )
                    self.effectiveness_torque_blocks[index].append(torque.copy())
                    self.effectiveness_zero_blocks[index].append(tau_zero.copy())
                    update_physical_estimator = False
                    if (
                        self.effectiveness_block_count[index]
                        >= self.effectiveness_window_steps
                    ):
                        denominator = self.effectiveness_denominator[index]
                        if denominator > 1.0e-10:
                            block_effectiveness = (
                                self.effectiveness_numerator[index] / denominator
                            )
                            self.estimated_effectiveness[index] = (
                                float(block_effectiveness)
                                if self.estimator_torque_mode
                                == "orthogonal_block_unbiased"
                                else float(
                                    np.clip(
                                        block_effectiveness,
                                        max(
                                            0.25,
                                            1.0 - 0.60 * self.stress_level,
                                        ),
                                        1.0,
                                    )
                                )
                            )
                        block_regressor = np.concatenate(
                            self.effectiveness_regressor_blocks[index], axis=0
                        )
                        block_torque = np.concatenate(
                            self.effectiveness_torque_blocks[index]
                        )
                        block_zero = np.concatenate(
                            self.effectiveness_zero_blocks[index]
                        )
                        estimator.update(
                            block_regressor,
                            self.estimated_effectiveness[index] * block_torque
                            - block_zero,
                            forgetting=(
                                self.estimator_forgetting
                                ** self.effectiveness_window_steps
                            ),
                        )
                        self.effectiveness_numerator[index] = 0.0
                        self.effectiveness_denominator[index] = 0.0
                        self.effectiveness_block_count[index] = 0
                        self.effectiveness_regressor_blocks[index].clear()
                        self.effectiveness_torque_blocks[index].clear()
                        self.effectiveness_zero_blocks[index].clear()
                elif projected_energy > 1.0e-10:
                    unconstrained_effectiveness = float(
                        np.dot(projected_torque, projected_zero)
                        / projected_energy
                    )
                    self.estimated_effectiveness[index] = (
                        unconstrained_effectiveness
                        if self.estimator_torque_mode == "orthogonal_unbiased"
                        else float(
                            np.clip(
                                unconstrained_effectiveness,
                                max(0.25, 1.0 - 0.60 * self.stress_level),
                                1.0,
                            )
                        )
                    )
                if self.estimator_torque_mode in (
                    "orthogonal_block",
                    "orthogonal_block_unbiased",
                ):
                    identified_torque = self.estimated_effectiveness[index] * torque
                elif self.estimator_torque_mode == "orthogonal_isolation":
                    # A one-sided consistency test is sufficient for update
                    # isolation.  It avoids injecting a noisy instantaneous
                    # effectiveness estimate into the slow physical model.
                    update_physical_estimator = (
                        self.estimated_effectiveness[index] >= 1.0
                    )
                    identified_torque = torque
                else:
                    identified_torque = self.estimated_effectiveness[index] * torque
            elif self.estimator_torque_mode == "causal_effectiveness":
                # Fast normalized projection for the scalar effectiveness in
                # tau_applied = eta*tau_command.  The physical parameters are
                # held at their preceding slow estimate while eta is updated,
                # preventing an abrupt actuator loss from being absorbed by
                # payload, friction, and drag in the same sample.
                reconstructed_applied_torque = tau_zero + regressor @ estimator.value
                torque_energy = float(np.dot(torque, torque))
                if torque_energy > 1.0e-10:
                    self.estimated_effectiveness[index] = float(
                        np.clip(
                            np.dot(torque, reconstructed_applied_torque)
                            / torque_energy,
                            max(0.25, 1.0 - 0.60 * self.stress_level),
                            1.0,
                        )
                    )
                identified_torque = self.estimated_effectiveness[index] * torque
            else:
                if self.estimator_torque_mode == "event_active_orthogonal":
                    self.estimated_effectiveness[index] = 1.0
                identified_torque = (
                    effectiveness * torque
                    if self.effectiveness_compensation == "oracle"
                    or self.estimator_torque_mode == "oracle_applied"
                    else torque
                )
                if self.estimator_torque_mode == "oracle_freeze":
                    update_physical_estimator = not in_actuator_event
                elif (
                    self.estimator_torque_mode in (
                        "external_effectiveness",
                        "external_effectiveness_unbiased",
                    )
                    and estimator_effectiveness_override is not None
                ):
                    external_estimate = float(
                        max(estimator_effectiveness_override[index], 0.25)
                    )
                    self.estimated_effectiveness[index] = (
                        min(external_estimate, 1.0)
                        if self.estimator_torque_mode == "external_effectiveness"
                        else external_estimate
                    )
                    identified_torque = (
                        self.estimated_effectiveness[index] * torque
                    )
                    if (
                        self.freeze_physical_on_external_fault
                        and self.estimated_effectiveness[index] < 1.0 - 1.0e-9
                    ):
                        update_physical_estimator = False
                elif self.estimator_torque_mode == "augmented_joint":
                    augmented = self.augmented_estimators[index]
                    augmented_regressor = np.concatenate(
                        (regressor, -torque[:, None]), axis=1
                    )
                    augmented.update(
                        augmented_regressor,
                        -tau_zero,
                        forgetting=self.estimator_forgetting,
                    )
                    estimator.value = augmented.value[:3].copy()
                    full_covariance = np.linalg.inv(
                        augmented.information + 1.0e-10 * np.eye(4)
                    )
                    physical_covariance = full_covariance[:3, :3]
                    estimator.information = np.linalg.inv(
                        physical_covariance + 1.0e-10 * np.eye(3)
                    )
                    estimator.rhs = estimator.information @ estimator.value
                    self.estimated_effectiveness[index] = float(
                        augmented.value[3]
                    )
                    identified_torque = (
                        self.estimated_effectiveness[index] * torque
                    )
                    update_physical_estimator = False
                elif self.estimator_torque_mode == "dual_timescale":
                    slow = self.slow_physical_estimators[index]
                    fast = self.fast_fault_estimators[index]
                    fast_regressor = np.stack(
                        (regressor[:, 2], -torque), axis=1
                    )
                    fast_target = (
                        -tau_zero - regressor[:, :2] @ slow.value
                    )
                    fast.update(
                        fast_regressor,
                        fast_target,
                        forgetting=self.estimator_forgetting,
                    )
                    drag_estimate, effectiveness_estimate = fast.value
                    slow_target = (
                        effectiveness_estimate * torque
                        - tau_zero
                        - regressor[:, 2] * drag_estimate
                    )
                    slow.update(regressor[:, :2], slow_target, forgetting=1.0)
                    estimator.value = np.array(
                        [slow.value[0], slow.value[1], drag_estimate]
                    )
                    slow_covariance = np.linalg.inv(
                        slow.information + 1.0e-10 * np.eye(2)
                    )
                    fast_covariance = np.linalg.inv(
                        fast.information + 1.0e-10 * np.eye(2)
                    )
                    physical_covariance = np.zeros((3, 3))
                    physical_covariance[:2, :2] = slow_covariance
                    physical_covariance[2, 2] = fast_covariance[0, 0]
                    estimator.information = np.linalg.inv(
                        physical_covariance + 1.0e-10 * np.eye(3)
                    )
                    estimator.rhs = estimator.information @ estimator.value
                    self.estimated_effectiveness[index] = float(
                        effectiveness_estimate
                    )
                    identified_torque = effectiveness_estimate * torque
                    update_physical_estimator = False
            response = identified_torque - tau_zero
            innovation = response - regressor @ estimator.value
            if update_physical_estimator:
                estimator.update(
                    regressor, response, forgetting=self.estimator_forgetting
                )
            self.innovation[index] = float(np.sqrt(np.mean(np.square(innovation))))
            self.last_energy[index] = float(np.dot(torque, torque))
            cost[index] = (
                10.0 * np.dot(error, error)
                + 0.1 * np.dot(velocity_error, velocity_error)
                + 1.0e-4 * self.last_energy[index]
                + 0.02 * (speed[index] - self.last_speed[index]) ** 2
            )
            self.maximum_error[index] = max(
                self.maximum_error[index], np.linalg.norm(error)
            )
            sample.ctrl[:] = torque
            mujoco.mj_step(model, sample)
        self.phase += speed * self.dt
        self.age += 1
        self.last_speed = speed
        done = self.age >= self.episode_length
        next_state = self.state()
        return next_state, cost, done
