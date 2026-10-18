"""Causal recursive identification for payload-affine robot dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PayloadEstimate:
    value: float
    information: float
    innovation_rms: float


class RobustScalarRLS:
    """Projected scalar RLS with optional Huber innovation weighting.

    The vector observation model at one sampling instant is
    ``response = regressor * payload + disturbance``.  Accumulating the
    scalar normal equations avoids a matrix inversion and is well suited to
    payload-affine inverse dynamics.
    """

    def __init__(
        self,
        initial: float = 1.0,
        prior_information: float = 1.0,
        forgetting: float = 0.995,
        lower: float = 0.4,
        upper: float = 2.0,
        huber_delta: float | None = 1.5,
        min_excitation: float = 1.0e-6,
    ) -> None:
        if prior_information <= 0.0:
            raise ValueError("prior_information must be positive")
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must lie in (0, 1]")
        if lower >= upper:
            raise ValueError("lower must be smaller than upper")
        if huber_delta is not None and huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        self.forgetting = forgetting
        self.lower = lower
        self.upper = upper
        self.huber_delta = huber_delta
        self.min_excitation = min_excitation
        self.information = float(prior_information)
        self.rhs = float(prior_information * initial)
        self.value = float(np.clip(initial, lower, upper))

    def update(
        self, regressor: np.ndarray, response: np.ndarray
    ) -> PayloadEstimate:
        phi = np.asarray(regressor, dtype=np.float64).reshape(-1)
        y = np.asarray(response, dtype=np.float64).reshape(-1)
        if phi.shape != y.shape:
            raise ValueError("regressor and response shapes must match")
        innovation = y - phi * self.value
        if self.huber_delta is None:
            weights = np.ones_like(innovation)
        else:
            magnitude = np.abs(innovation)
            weights = np.minimum(
                1.0, self.huber_delta / np.maximum(magnitude, 1.0e-12)
            )
        excitation = float(np.dot(weights * phi, phi))
        if excitation >= self.min_excitation:
            self.information = self.forgetting * self.information + excitation
            self.rhs = self.forgetting * self.rhs + float(
                np.dot(weights * phi, y)
            )
            self.value = float(
                np.clip(
                    self.rhs / max(self.information, 1.0e-12),
                    self.lower,
                    self.upper,
                )
            )
        return PayloadEstimate(
            value=self.value,
            information=self.information,
            innovation_rms=float(np.sqrt(np.mean(np.square(innovation)))),
        )


class RobustLinearRLS:
    """Projected multi-parameter recursive weighted normal equations."""

    def __init__(
        self,
        initial: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        prior_information: float = 0.5,
        forgetting: float = 0.995,
        huber_delta: float | None = 1.0,
    ) -> None:
        initial = np.asarray(initial, dtype=np.float64)
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        if initial.ndim != 1 or self.lower.shape != initial.shape or self.upper.shape != initial.shape:
            raise ValueError("initial and bounds must be matching vectors")
        if np.any(self.lower >= self.upper):
            raise ValueError("every lower bound must be below its upper bound")
        if prior_information <= 0.0 or not 0.0 < forgetting <= 1.0:
            raise ValueError("invalid information or forgetting factor")
        self.forgetting = forgetting
        self.huber_delta = huber_delta
        self.value = np.clip(initial, self.lower, self.upper)
        self.information = prior_information * np.eye(initial.size)
        self.rhs = self.information @ self.value

    def update(
        self,
        regressor: np.ndarray,
        response: np.ndarray,
        forgetting: float | None = None,
    ) -> np.ndarray:
        matrix = np.asarray(regressor, dtype=np.float64)
        target = np.asarray(response, dtype=np.float64).reshape(-1)
        if matrix.ndim != 2 or matrix.shape[0] != target.size:
            raise ValueError("regressor must have one row per response")
        if matrix.shape[1] != self.value.size:
            raise ValueError("regressor parameter dimension does not match")
        active_forgetting = self.forgetting if forgetting is None else forgetting
        if not 0.0 < active_forgetting <= 1.0:
            raise ValueError("forgetting must lie in (0, 1]")
        innovation = target - matrix @ self.value
        if self.huber_delta is None:
            weights = np.ones_like(innovation)
        else:
            weights = np.minimum(
                1.0,
                self.huber_delta / np.maximum(np.abs(innovation), 1.0e-12),
            )
        weighted = matrix * weights[:, None]
        self.information = (
            active_forgetting * self.information + matrix.T @ weighted
        )
        self.rhs = active_forgetting * self.rhs + weighted.T @ target
        regularized = self.information + 1.0e-10 * np.eye(self.value.size)
        self.value = np.clip(
            np.linalg.solve(regularized, self.rhs), self.lower, self.upper
        )
        return self.value.copy()
