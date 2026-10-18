"""Short-horizon backup-tube prediction for path-progress recovery."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


Reference = Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]]


def backup_tube_peak_error(
    *,
    error: np.ndarray,
    velocity_error: np.ndarray,
    progress: float,
    speed: float,
    first_acceleration: float,
    maximum_acceleration: float,
    decision_period: float,
    horizon_steps: int,
    actuator_lower_bound: float,
    proportional_gain: np.ndarray,
    derivative_gain: np.ndarray,
    reference: Reference,
    disturbance_acceleration_bound: float = 0.0,
    integration_substeps: int = 5,
) -> float:
    """Predict peak tracking error for one move followed by hard braking.

    The reduced error model is

    ``e_ddot = -eta Kp e - eta Kd e_dot - (1-eta) q_ref_ddot + d``.

    It is the exact decoupled computed-torque error model under a scalar loss
    of effectiveness, augmented by a componentwise worst-direction bounded
    acceleration disturbance.  The first candidate acceleration is followed
    by the admissible hard-braking backup policy.
    """
    if horizon_steps < 1 or integration_substeps < 1:
        raise ValueError("prediction horizon and integration substeps must be positive")
    if not 0.0 < actuator_lower_bound <= 1.0:
        raise ValueError("actuator lower bound must lie in (0, 1]")
    e = np.asarray(error, dtype=float).copy()
    edot = np.asarray(velocity_error, dtype=float).copy()
    kp = np.asarray(proportional_gain, dtype=float)
    kd = np.asarray(derivative_gain, dtype=float)
    if e.shape != edot.shape or e.shape != kp.shape or e.shape != kd.shape:
        raise ValueError("error and gain vectors must have equal shapes")

    s = float(progress)
    v = max(float(speed), 0.0)
    peak = float(np.linalg.norm(e))
    dt = float(decision_period) / integration_substeps
    eta = float(actuator_lower_bound)
    for stage in range(horizon_steps):
        acceleration = (
            float(first_acceleration) if stage == 0 else -float(maximum_acceleration)
        )
        if v <= 0.0 and acceleration < 0.0:
            acceleration = 0.0
        for _ in range(integration_substeps):
            _, tangent, curvature = reference(s)
            reference_acceleration = curvature * v**2 + tangent * acceleration
            disturbance = disturbance_acceleration_bound * np.sign(edot)
            error_acceleration = (
                -eta * kp * e
                - eta * kd * edot
                - (1.0 - eta) * reference_acceleration
                + disturbance
            )
            edot = edot + dt * error_acceleration
            e = e + dt * edot
            next_speed = max(v + dt * acceleration, 0.0)
            s = s + 0.5 * (v + next_speed) * dt
            v = next_speed
            peak = max(peak, float(np.linalg.norm(e)))
    return peak


def feasible_backup_accelerations(
    *,
    lower: float,
    upper: float,
    grid_size: int,
    error_limit: float,
    predictor: Callable[[float], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate accelerations and their predicted peak errors."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    if lower > upper:
        return np.empty(0), np.empty(0)
    candidates = np.linspace(float(lower), float(upper), grid_size)
    peaks = np.array([predictor(float(value)) for value in candidates])
    feasible = peaks <= float(error_limit)
    return candidates[feasible], peaks
