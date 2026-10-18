"""Causal adaptations of recently published actuator-fault estimators.

The classes in this module share the same residual reconstruction and differ
only in the update mechanism attributed to each published method family.  They
are independent engineering adaptations, not official author implementations.
No class reads ``env.true_effectiveness`` or ``env.actuator_effectiveness``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation_environment import ProgressConstrainedEnvironment
from screen_tracking_task import DRAG_PROFILE, FRICTION_PROFILE
from validate_seven_link_mujoco import inverse_torque, set_payload


def residual_effectiveness(env: ProgressConstrainedEnvironment) -> np.ndarray:
    """Reconstruct a causal scalar effectiveness observation.

    The preceding command, current proprioceptive state, measured acceleration,
    and preceding slow physical estimate are used.  The computation is shared
    by all public comparators so differences arise from their update laws, not
    from unequal measurements.
    """

    output = np.ones(env.count, dtype=float)
    # The estimator may use the physical effectiveness bound, but not the
    # experiment generator's stress level.  The latter determines the hidden
    # fault distribution and would leak privileged test information.
    lower = 0.25
    for index, (sample, estimator) in enumerate(zip(env.data, env.estimators)):
        command = sample.ctrl.copy()
        energy = float(np.dot(command, command))
        if energy <= 1.0e-10:
            continue
        q = sample.qpos.copy()
        dq = sample.qvel.copy()
        acceleration = sample.qacc.copy()

        set_payload(
            env.inverse_model,
            env.inverse_payload,
            env.inverse_mass,
            env.inverse_inertia,
            0.05,
        )
        env.inverse_model.dof_frictionloss[:] = 0.0
        tau_zero = inverse_torque(
            env.inverse_model, env.inverse_data, q, dq, acceleration
        )

        set_payload(
            env.inverse_model,
            env.inverse_payload,
            env.inverse_mass,
            env.inverse_inertia,
            1.05,
        )
        tau_payload = inverse_torque(
            env.inverse_model, env.inverse_data, q, dq, acceleration
        )

        set_payload(
            env.inverse_model,
            env.inverse_payload,
            env.inverse_mass,
            env.inverse_inertia,
            0.05,
        )
        env.inverse_model.dof_frictionloss[:] = FRICTION_PROFILE
        tau_friction = inverse_torque(
            env.inverse_model, env.inverse_data, q, dq, acceleration
        )
        regressor = np.stack(
            (
                tau_payload - tau_zero,
                tau_friction - tau_zero,
                DRAG_PROFILE * dq * np.abs(dq),
            ),
            axis=-1,
        )
        reconstructed = tau_zero + regressor @ estimator.value
        raw = float(np.dot(command, reconstructed) / energy)
        output[index] = float(np.clip(raw, lower, 1.0))
    return output


@dataclass(frozen=True)
class ComparatorMetadata:
    label: str
    year: int
    doi: str
    adaptation: str


class CausalComparator:
    metadata: ComparatorMetadata

    def __init__(self) -> None:
        self.estimate: np.ndarray | None = None
        self.latest_selected: np.ndarray | None = None

    def initialize(self, count: int) -> None:
        self.estimate = np.ones(count, dtype=float)

    def update(
        self, observation: np.ndarray, env: ProgressConstrainedEnvironment
    ) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, env: ProgressConstrainedEnvironment) -> np.ndarray:
        if self.estimate is None:
            self.initialize(env.count)
        selected = self.update(residual_effectiveness(env), env)
        self.latest_selected = np.clip(selected, 0.25, 1.0)
        return self.latest_selected.copy()


class SynchronousESOComparator(CausalComparator):
    """SFTSMC--ESO-inspired bandwidth-limited fault reconstruction."""

    metadata = ComparatorMetadata(
        "SFTSMC--ESO adaptation",
        2022,
        "10.3390/act11070195",
        "common residual reconstruction followed by an ESO bandwidth filter",
    )

    def __init__(self, bandwidth: float = 24.0) -> None:
        super().__init__()
        self.bandwidth = bandwidth

    def update(self, observation: np.ndarray, env: ProgressConstrainedEnvironment) -> np.ndarray:
        assert self.estimate is not None
        gain = 1.0 - np.exp(-self.bandwidth * env.dt)
        self.estimate += gain * (observation - self.estimate)
        return self.estimate


class DynamicRBFModeComparator(CausalComparator):
    """DL--RBF-inspired finite fault-mode bank with normalized RBF selection."""

    metadata = ComparatorMetadata(
        "DL--RBF mode-bank adaptation",
        2023,
        "10.1016/j.jfranklin.2022.11.044",
        "frozen effectiveness-mode bank with normalized Gaussian RBF selection",
    )

    def __init__(self, modes: int = 9, width: float = 0.055, memory: float = 0.72) -> None:
        super().__init__()
        self.modes = modes
        self.width = width
        self.memory = memory

    def update(self, observation: np.ndarray, env: ProgressConstrainedEnvironment) -> np.ndarray:
        assert self.estimate is not None
        centers = np.linspace(0.25, 1.0, self.modes)
        weights = np.exp(
            -0.5 * ((observation[:, None] - centers[None, :]) / self.width) ** 2
        )
        selected = (weights @ centers) / np.maximum(weights.sum(axis=1), 1.0e-12)
        self.estimate = self.memory * self.estimate + (1.0 - self.memory) * selected
        return self.estimate


class ParameterTriggeredComparator(CausalComparator):
    """PET--AFTC-inspired event-triggered estimator output."""

    metadata = ComparatorMetadata(
        "PET--AFTC adaptation",
        2023,
        "10.1016/j.automatica.2023.110954",
        "filtered effectiveness estimate transmitted by error and timeout triggers",
    )

    def __init__(self, bandwidth: float = 30.0, threshold: float = 0.012, timeout: int = 20) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.timeout = timeout
        self.internal: np.ndarray | None = None
        self.age: np.ndarray | None = None

    def initialize(self, count: int) -> None:
        super().initialize(count)
        self.internal = np.ones(count, dtype=float)
        self.age = np.zeros(count, dtype=np.int64)

    def update(self, observation: np.ndarray, env: ProgressConstrainedEnvironment) -> np.ndarray:
        assert self.estimate is not None and self.internal is not None and self.age is not None
        gain = 1.0 - np.exp(-self.bandwidth * env.dt)
        self.internal += gain * (observation - self.internal)
        self.age += 1
        trigger = (np.abs(self.internal - self.estimate) >= self.threshold) | (self.age >= self.timeout)
        self.estimate[trigger] = self.internal[trigger]
        self.age[trigger] = 0
        return self.estimate


class FixedTimeASMOComparator(CausalComparator):
    """FASMO--CFTC-inspired two-power fixed-time observer update."""

    metadata = ComparatorMetadata(
        "FASMO--CFTC adaptation",
        2024,
        "10.1049/cth2.12672",
        "two-power fixed-time sliding-mode update of the common residual estimate",
    )

    def __init__(self, low_gain: float = 7.0, high_gain: float = 5.0) -> None:
        super().__init__()
        self.low_gain = low_gain
        self.high_gain = high_gain

    def update(self, observation: np.ndarray, env: ProgressConstrainedEnvironment) -> np.ndarray:
        assert self.estimate is not None
        error = observation - self.estimate
        rate = (
            self.low_gain * np.sqrt(np.abs(error))
            + self.high_gain * np.abs(error) ** 1.5
        ) * np.sign(error)
        step = np.minimum(np.abs(error), env.dt * np.abs(rate)) * np.sign(error)
        self.estimate += step
        return self.estimate


class LearningEstimatorComparator(CausalComparator):
    """LE--FTC-inspired sign-enhanced learning estimator."""

    metadata = ComparatorMetadata(
        "LE--FTC adaptation",
        2025,
        "10.1002/asjc.3594",
        "sign-enhanced bounded learning update driven by the common residual estimate",
    )

    def __init__(self, gain: float = 18.0, sign_gain: float = 0.018, deadband: float = 0.002) -> None:
        super().__init__()
        self.gain = gain
        self.sign_gain = sign_gain
        self.deadband = deadband

    def update(self, observation: np.ndarray, env: ProgressConstrainedEnvironment) -> np.ndarray:
        assert self.estimate is not None
        error = observation - self.estimate
        robust = np.where(
            np.abs(error) > self.deadband,
            self.sign_gain * np.sign(error),
            0.0,
        )
        increment = env.dt * self.gain * (error + robust)
        increment = np.clip(increment, -np.abs(error), np.abs(error))
        self.estimate += increment
        return self.estimate


COMPARATORS = {
    "sftsmc_eso": SynchronousESOComparator,
    "dl_rbf": DynamicRBFModeComparator,
    "pet_aftc": ParameterTriggeredComparator,
    "fasmo_cftc": FixedTimeASMOComparator,
    "le_ftc": LearningEstimatorComparator,
}

