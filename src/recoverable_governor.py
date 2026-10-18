"""Analytic components of a second-order recoverable progress governor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sampled_clf_terms(
    error: np.ndarray,
    velocity_error: np.ndarray,
    measured_acceleration: np.ndarray,
    path_tangent: np.ndarray,
    path_curvature: np.ndarray,
    speed: float,
    proportional_gain: np.ndarray,
) -> tuple[float, float, float]:
    """Return ``V``, ``phi_0`` and ``phi_1`` for ``Vdot=phi_0+phi_1*a``."""
    lyapunov = 0.5 * (
        np.dot(velocity_error, velocity_error)
        + np.dot(proportional_gain * error, error)
    )
    offset = float(
        np.dot(proportional_gain * error, velocity_error)
        + np.dot(
            velocity_error,
            measured_acceleration - speed**2 * path_curvature,
        )
    )
    slope = -float(np.dot(velocity_error, path_tangent))
    return float(lyapunov), offset, slope


def sampled_error_barrier_terms(
    error: np.ndarray,
    velocity_error: np.ndarray,
    measured_acceleration: np.ndarray,
    path_tangent: np.ndarray,
    path_curvature: np.ndarray,
    speed: float,
    error_limit: float,
) -> tuple[float, float, float, float]:
    """Return ``h``, ``hdot``, and affine terms for ``hddot``.

    For ``h=e_max^2-e'e`` and path acceleration ``a``, the returned terms
    satisfy ``hddot = offset + slope*a`` at the sampled state.
    """
    error = np.asarray(error, dtype=float)
    velocity_error = np.asarray(velocity_error, dtype=float)
    h = float(error_limit**2 - np.dot(error, error))
    hdot = -2.0 * float(np.dot(error, velocity_error))
    relative_acceleration_offset = (
        measured_acceleration - speed**2 * path_curvature
    )
    offset = -2.0 * float(
        np.dot(velocity_error, velocity_error)
        + np.dot(error, relative_acceleration_offset)
    )
    slope = 2.0 * float(np.dot(error, path_tangent))
    return h, hdot, offset, slope


def minimum_completion_time(
    distance: float,
    speed: float,
    maximum_speed: float,
    maximum_acceleration: float,
) -> float:
    """Minimum time to travel a nonnegative distance under speed/acceleration bounds."""
    if distance <= 0.0:
        return 0.0
    speed = float(np.clip(speed, 0.0, maximum_speed))
    if maximum_acceleration <= 0.0:
        return np.inf if speed <= 0.0 else distance / speed
    acceleration_distance = (
        maximum_speed**2 - speed**2
    ) / (2.0 * maximum_acceleration)
    if distance <= acceleration_distance:
        return (
            np.sqrt(speed**2 + 2.0 * maximum_acceleration * distance) - speed
        ) / maximum_acceleration
    acceleration_time = (maximum_speed - speed) / maximum_acceleration
    return acceleration_time + (distance - acceleration_distance) / maximum_speed


def affine_box_interval(
    offset: np.ndarray,
    slope: np.ndarray,
    limit: np.ndarray,
    lower: float,
    upper: float,
    slope_tolerance: float = 1.0e-10,
) -> tuple[float, float, float]:
    """Intersect ``|offset + slope*a| <= limit`` with a scalar box.

    The returned violation is zero for a nonempty interval.  When uncertainty
    alone exhausts the control authority, it measures the gap between the
    incompatible lower and upper acceleration bounds.
    """
    lo = float(lower)
    hi = float(upper)
    infeasible_constant = 0.0
    for b, h, cap in zip(offset, slope, limit):
        if abs(h) <= slope_tolerance:
            infeasible_constant = max(infeasible_constant, abs(float(b)) - float(cap))
            continue
        first = (-float(cap) - float(b)) / float(h)
        second = (float(cap) - float(b)) / float(h)
        lo = max(lo, min(first, second))
        hi = min(hi, max(first, second))
    violation = max(infeasible_constant, lo - hi, 0.0)
    return lo, hi, violation


def affine_upper_interval(
    offset: float,
    slope: float,
    upper_bound: float,
    lower: float,
    upper: float,
    slope_tolerance: float = 1.0e-10,
) -> tuple[float, float, float]:
    """Intersect ``offset + slope*a <= upper_bound`` with an interval."""
    lo = float(lower)
    hi = float(upper)
    if abs(slope) <= slope_tolerance:
        return lo, hi, max(float(offset - upper_bound), 0.0)
    boundary = (float(upper_bound) - float(offset)) / float(slope)
    if slope > 0.0:
        hi = min(hi, boundary)
    else:
        lo = max(lo, boundary)
    return lo, hi, max(lo - hi, 0.0)


def nominal_progress_acceleration(
    distance: float,
    speed: float,
    remaining_time: float,
    maximum_acceleration: float,
) -> float:
    """Constant-acceleration command that closes the progress gap at a deadline."""
    if remaining_time <= 1.0e-9:
        return maximum_acceleration
    command = 2.0 * (distance - speed * remaining_time) / remaining_time**2
    return float(np.clip(command, -maximum_acceleration, maximum_acceleration))


@dataclass(frozen=True)
class GovernorDecision:
    acceleration: float
    deadline: float
    deadline_extension: float
    certificate_lower: float
    certificate_upper: float
    certificate_violation: float


def governor_decision(
    *,
    time: float,
    deadline: float,
    distance: float,
    speed: float,
    decision_period: float,
    maximum_speed: float,
    maximum_acceleration: float,
    certificate_lower: float,
    certificate_upper: float,
    certificate_violation: float = 0.0,
    recovery_acceleration: float | None = None,
    previous_acceleration: float = 0.0,
    maximum_recovery_jerk: float | None = None,
) -> GovernorDecision:
    """Solve the scalar governor QP and update the recoverable deadline."""
    remaining_time = max(deadline - time, decision_period)
    nominal = nominal_progress_acceleration(
        distance, speed, remaining_time, maximum_acceleration
    )
    if certificate_violation > 0.0 or certificate_lower > certificate_upper:
        acceleration = (
            -maximum_acceleration
            if recovery_acceleration is None
            else float(
                np.clip(
                    recovery_acceleration,
                    -maximum_acceleration,
                    maximum_acceleration,
                )
            )
        )
    else:
        acceleration = float(np.clip(nominal, certificate_lower, certificate_upper))
    if (
        maximum_recovery_jerk is not None
        and acceleration > previous_acceleration
    ):
        acceleration = min(
            acceleration,
            previous_acceleration + maximum_recovery_jerk * decision_period,
        )

    next_speed = float(
        np.clip(speed + acceleration * decision_period, 0.0, maximum_speed)
    )
    next_distance = max(
        distance
        - decision_period * speed
        - 0.5 * acceleration * decision_period**2,
        0.0,
    )
    earliest_completion = time + decision_period + minimum_completion_time(
        next_distance, next_speed, maximum_speed, maximum_acceleration
    )
    updated_deadline = max(deadline, earliest_completion)
    return GovernorDecision(
        acceleration=acceleration,
        deadline=updated_deadline,
        deadline_extension=updated_deadline - deadline,
        certificate_lower=certificate_lower,
        certificate_upper=certificate_upper,
        certificate_violation=certificate_violation,
    )
