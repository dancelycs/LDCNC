"""Close the learned effectiveness estimate around the adaptive identifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = ROOT
LEGACY_ROOT = ROOT
GOVERNOR_ROOT = ROOT
sys.path.insert(0, str(ROOT / "src"))

from screen_certificate_governor import run_certificate_method  # noqa: E402
from screen_history_identifiability import (  # noqa: E402
    HistoryEncoder,
    SnapshotEncoder,
)
from simulation_environment import ProgressConstrainedEnvironment  # noqa: E402


class OnlineEffectivenessPredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        architecture: str,
        calibration_mode: str = "point",
        radius_scope: str = "point",
        fixed_deadband: float = 0.04,
        acceleration_source: str = "simulator",
        latch_dwell: int = 1,
        sequential_reference: float = 0.03,
        sequential_threshold: float = 1.0,
    ) -> None:
        # This checkpoint is produced locally by the paired training script and
        # also contains NumPy normalization arrays in addition to tensor weights.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if architecture == "history":
            self.model = HistoryEncoder(
                checkpoint["feature_size"],
                checkpoint["hidden_size"],
                checkpoint["output_size"],
            )
            state_dict = checkpoint["state_dict"]
        elif architecture == "snapshot":
            self.model = SnapshotEncoder(
                checkpoint["feature_size"],
                checkpoint["hidden_size"],
                checkpoint["output_size"],
            )
            state_dict = checkpoint["snapshot_state_dict"]
        else:
            raise ValueError("architecture must be 'history' or 'snapshot'")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.history = int(checkpoint["history"])
        self.feature_mean = np.asarray(checkpoint["feature_mean"])
        self.feature_scale = np.asarray(checkpoint["feature_scale"])
        self.label_mean = np.asarray(checkpoint["label_mean"])
        self.label_scale = np.asarray(checkpoint["label_scale"])
        self.mask_phase = bool(checkpoint.get("mask_phase", False))
        self.nominal_conformal_radius = float(
            checkpoint.get(
                "nominal_scenario_conformal_radius",
                checkpoint["conformal_radius"],
            )
        )
        radius_keys = {
            "point": "conformal_radius",
            "nominal_trajectory": "nominal_trajectory_conformal_radius",
            "scenario_marginal": "scenario_marginal_conformal_radius",
            "nominal_scenario": "nominal_scenario_conformal_radius",
        }
        if radius_scope in {"dual", "phase_dual"}:
            self.conformal_radius = max(
                float(checkpoint["scenario_marginal_conformal_radius"]),
                float(checkpoint["nominal_scenario_conformal_radius"]),
            )
        else:
            radius_key = radius_keys[radius_scope]
            self.conformal_radius = float(checkpoint[radius_key])
        self.phase_edges = np.asarray(
            checkpoint.get("phase_stratified_edges", []), dtype=float
        )
        phase_global = np.asarray(
            checkpoint.get("phase_stratified_global_radii", []), dtype=float
        )
        phase_nominal = np.asarray(
            checkpoint.get("phase_stratified_nominal_radii", []), dtype=float
        )
        self.phase_radii = (
            np.maximum(phase_global, phase_nominal)
            if radius_scope == "phase_dual" and phase_global.size
            else phase_global
        )
        if radius_scope != "phase_dual":
            self.phase_radii = np.asarray([], dtype=float)
        self.calibration_mode = calibration_mode
        self.fixed_deadband = float(fixed_deadband)
        if acceleration_source not in {"simulator", "causal_difference"}:
            raise ValueError("unknown acceleration source")
        self.acceleration_source = acceleration_source
        if latch_dwell < 1:
            raise ValueError("latch dwell must be positive")
        self.latch_dwell = int(latch_dwell)
        self.sequential_reference = float(sequential_reference)
        self.sequential_threshold = float(sequential_threshold)
        self.buffer: np.ndarray | None = None
        self.filled = 0
        self.latest_lower: np.ndarray | None = None
        self.latest_point: np.ndarray | None = None
        self.latest_selected: np.ndarray | None = None
        self.fault_latched: np.ndarray | None = None
        self.first_latch_step: np.ndarray | None = None
        self.rejection_count: np.ndarray | None = None
        self.sequential_evidence: np.ndarray | None = None
        self.previous_velocity: np.ndarray | None = None
        self.filtered_acceleration: np.ndarray | None = None
        self.call_count = 0

    @torch.no_grad()
    def __call__(self, env: ProgressConstrainedEnvironment) -> np.ndarray:
        feature_size = self.feature_mean.size
        if self.buffer is None:
            self.buffer = np.zeros(
                (env.count, self.history, feature_size), dtype=np.float32
            )
            self.fault_latched = np.zeros(env.count, dtype=bool)
            self.first_latch_step = np.full(env.count, -1, dtype=int)
            self.rejection_count = np.zeros(env.count, dtype=int)
            self.sequential_evidence = np.zeros(env.count, dtype=float)
            self.previous_velocity = np.stack(
                [sample.qvel.copy() for sample in env.data]
            )
            self.filtered_acceleration = np.zeros_like(self.previous_velocity)
        current = np.zeros((env.count, feature_size), dtype=np.float32)
        for index, sample in enumerate(env.data):
            acceleration = sample.qacc
            if self.acceleration_source == "causal_difference":
                assert self.previous_velocity is not None
                assert self.filtered_acceleration is not None
                raw_acceleration = (
                    sample.qvel - self.previous_velocity[index]
                ) / float(env.dt)
                # First-order causal reconstruction; the fixed coefficient
                # corresponds to no future samples and is shared by methods.
                self.filtered_acceleration[index] = (
                    0.25 * raw_acceleration
                    + 0.75 * self.filtered_acceleration[index]
                )
                acceleration = self.filtered_acceleration[index]
                self.previous_velocity[index] = sample.qvel.copy()
            task_phase = 0.0 if self.mask_phase else env.phase[index]
            current[index] = np.concatenate(
                (
                    sample.qpos,
                    sample.qvel,
                    sample.ctrl,
                    acceleration,
                    np.array(
                        [
                            task_phase,
                            env.last_speed[index],
                            env.innovation[index],
                        ]
                    ),
                )
            )
        current = (current - self.feature_mean) / self.feature_scale
        self.buffer[:, :-1] = self.buffer[:, 1:]
        self.buffer[:, -1] = current
        self.filled = min(self.filled + 1, self.history)
        self.call_count += 1
        if self.filled < self.history:
            self.latest_selected = np.ones(env.count)
            return np.ones(env.count)
        prediction = self.model(
            torch.from_numpy(self.buffer).to(self.device)
        ).cpu().numpy()
        prediction = prediction * self.label_scale + self.label_mean
        effectiveness = prediction[:, 0]
        decision_radius: float | np.ndarray = self.conformal_radius
        if self.phase_radii.size:
            phase = np.asarray(env.phase, dtype=float)
            bin_index = np.searchsorted(
                self.phase_edges[1:], phase, side="right"
            )
            bin_index = np.clip(bin_index, 0, self.phase_radii.size - 1)
            decision_radius = self.phase_radii[bin_index]
        # Retained for diagnostic trajectory export; it does not alter the
        # value returned to the physical-parameter identifier.
        self.latest_point = effectiveness.copy()
        self.latest_lower = np.clip(
            effectiveness - decision_radius, 0.25, 1.0
        )
        if self.calibration_mode == "point":
            self.latest_selected = effectiveness
            return effectiveness
        if self.calibration_mode == "nominal_projection":
            # If the calibrated set contains the known nominal effectiveness,
            # the data do not justify writing a fault into the slow identifier.
            nominal_is_plausible = (
                np.abs(effectiveness - 1.0) <= decision_radius
            )
            selected = np.where(nominal_is_plausible, 1.0, effectiveness)
            self.latest_selected = selected
            return selected
        if self.calibration_mode == "bounded_nominal_correction":
            # The global interval decides whether nominal operation is
            # plausible, while the independently calibrated nominal radius
            # limits how far the interface may move a point estimate. On the
            # nominal-coverage event this still reaches exactly one, but an
            # ambiguous fault estimate cannot be discontinuously projected by
            # the larger global radius.
            nominal_is_plausible = (
                np.abs(effectiveness - 1.0) <= decision_radius
            )
            correction = np.clip(
                1.0 - effectiveness,
                -self.nominal_conformal_radius,
                self.nominal_conformal_radius,
            )
            selected = np.where(
                nominal_is_plausible,
                effectiveness + correction,
                effectiveness,
            )
            self.latest_selected = selected
            return selected
        if self.calibration_mode == "latched_nominal_correction":
            if self.fault_latched is None:
                raise RuntimeError("predictor latch was not initialized")
            nominal_is_plausible = (
                np.abs(effectiveness - 1.0) <= decision_radius
            )
            assert self.rejection_count is not None
            self.rejection_count = np.where(
                nominal_is_plausible, 0, self.rejection_count + 1
            )
            rejection_confirmed = self.rejection_count >= self.latch_dwell
            newly_latched = rejection_confirmed & ~self.fault_latched
            self.fault_latched |= rejection_confirmed
            if self.first_latch_step is not None:
                self.first_latch_step[newly_latched] = self.call_count - 1
            correction = np.clip(
                1.0 - effectiveness,
                -self.nominal_conformal_radius,
                self.nominal_conformal_radius,
            )
            selected = np.where(
                nominal_is_plausible & ~self.fault_latched,
                effectiveness + correction,
                effectiveness,
            )
            self.latest_selected = selected
            return selected
        if self.calibration_mode == "fixed_deadband_latch":
            if self.fault_latched is None:
                raise RuntimeError("predictor latch was not initialized")
            nominal_is_plausible = (
                np.abs(effectiveness - 1.0) <= self.fixed_deadband
            )
            assert self.rejection_count is not None
            self.rejection_count = np.where(
                nominal_is_plausible, 0, self.rejection_count + 1
            )
            rejection_confirmed = self.rejection_count >= self.latch_dwell
            newly_latched = rejection_confirmed & ~self.fault_latched
            self.fault_latched |= rejection_confirmed
            if self.first_latch_step is not None:
                self.first_latch_step[newly_latched] = self.call_count - 1
            selected = np.where(
                nominal_is_plausible & ~self.fault_latched,
                1.0,
                effectiveness,
            )
            self.latest_selected = selected
            return selected
        if self.calibration_mode == "sequential_latch":
            if self.fault_latched is None or self.sequential_evidence is None:
                raise RuntimeError("predictor latch was not initialized")
            increment = 1.0 - effectiveness - self.sequential_reference
            self.sequential_evidence = np.maximum(
                0.0, self.sequential_evidence + increment
            )
            rejection_confirmed = (
                self.sequential_evidence > self.sequential_threshold
            )
            newly_latched = rejection_confirmed & ~self.fault_latched
            self.fault_latched |= rejection_confirmed
            if self.first_latch_step is not None:
                self.first_latch_step[newly_latched] = self.call_count - 1
            correction = np.clip(
                1.0 - effectiveness,
                -self.nominal_conformal_radius,
                self.nominal_conformal_radius,
            )
            selected = np.where(
                ~self.fault_latched,
                effectiveness + correction,
                effectiveness,
            )
            self.latest_selected = selected
            return selected
        raise ValueError("unknown calibration mode")


def certificate_parameters(args: argparse.Namespace) -> dict[str, object]:
    return {
        "decision_interval": args.decision_interval,
        "acceleration_bound": args.acceleration_bound,
        "torque_reserve": 2.0,
        "confidence_scale": 0.75,
        "clf_rate": 1.0,
        "clf_level": 0.5,
        "clf_remainder_bound": 0.08037916898668823,
        "maximum_recovery_jerk": 1.0e6,
        "recovery_mode": "hard_brake",
        "effectiveness_compensation": "none",
        "simulation_timestep": args.simulation_timestep,
        "return_details": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8701)
    parser.add_argument("--scenarios", type=int, default=4)
    parser.add_argument("--stress-level", type=float, default=0.27)
    parser.add_argument("--simulation-timestep", type=float, default=0.005)
    parser.add_argument("--nominal-horizon", type=int, default=1000)
    parser.add_argument("--maximum-horizon", type=int, default=1800)
    parser.add_argument("--decision-interval", type=int, default=20)
    parser.add_argument("--acceleration-bound", type=float, default=2.0)
    parser.add_argument("--include-rejected-unbiased", action="store_true")
    parser.add_argument("--history-only", action="store_true")
    parser.add_argument(
        "--single-method",
        choices=("commanded", "snapshot", "history", "oracle"),
        help=(
            "evaluate one interface and emit one estimator payload for paired "
            "paper statistics"
        ),
    )
    parser.add_argument("--architecture", choices=("history",), default="history")
    parser.add_argument(
        "--calibration-mode",
        choices=(
            "point",
            "nominal_projection",
            "bounded_nominal_correction",
            "latched_nominal_correction",
            "fixed_deadband_latch",
            "sequential_latch",
        ),
        default="point",
    )
    parser.add_argument(
        "--radius-scope",
        choices=(
            "point",
            "nominal_trajectory",
            "scenario_marginal",
            "nominal_scenario",
            "dual",
            "phase_dual",
        ),
        default="point",
    )
    parser.add_argument("--dynamic-effectiveness-bound", action="store_true")
    parser.add_argument(
        "--fixed-deadband", type=float, default=0.04,
        help="nominal deadband used only by fixed_deadband_latch",
    )
    parser.add_argument(
        "--acceleration-source",
        choices=("simulator", "causal_difference"), default="simulator",
    )
    parser.add_argument("--latch-dwell", type=int, default=1)
    parser.add_argument("--sequential-reference", type=float, default=0.03)
    parser.add_argument("--sequential-threshold", type=float, default=1.0)
    parser.add_argument(
        "--learned-effectiveness-compensation", action="store_true"
    )
    parser.add_argument("--fault-speed-derating", action="store_true")
    parser.add_argument(
        "--freeze-physical-on-detected-fault", action="store_true"
    )
    parser.add_argument(
        "--fault-profile",
        choices=("pulse", "ramp", "intermittent", "persistent"),
        default="pulse",
    )
    parser.add_argument(
        "--model", type=Path, default=ROOT / "assets" / "seven_link_arm.xml"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "learned_effectiveness_control.json",
    )
    args = parser.parse_args()
    common = {
        "model_path": args.model,
        "seed": args.seed,
        "scenarios": args.scenarios,
        "stress_level": args.stress_level,
        "nominal_horizon": args.nominal_horizon,
        "maximum_horizon": args.maximum_horizon,
        "dynamic_effectiveness_bound": args.dynamic_effectiveness_bound,
        "fault_profile": args.fault_profile,
        "fault_speed_derating": args.fault_speed_derating,
        "freeze_physical_on_external_fault": (
            args.freeze_physical_on_detected_fault
        ),
        **certificate_parameters(args),
    }
    if args.learned_effectiveness_compensation:
        common["effectiveness_compensation"] = "external"
    result: dict[str, object] = {
        "seed": args.seed,
        "scenarios": args.scenarios,
        "stress_level": args.stress_level,
    }
    if args.single_method is not None:
        if args.single_method == "commanded":
            payload = run_certificate_method(
                **common, estimator_torque_mode="commanded"
            )
        elif args.single_method == "oracle":
            payload = run_certificate_method(
                **common, estimator_torque_mode="oracle_applied"
            )
        else:
            architecture = (
                "snapshot" if args.single_method == "snapshot" else args.architecture
            )
            payload = run_certificate_method(
                **common,
                estimator_torque_mode="external_effectiveness",
                effectiveness_predictor=OnlineEffectivenessPredictor(
                    args.checkpoint,
                    architecture,
                    args.calibration_mode,
                    args.radius_scope,
                    args.fixed_deadband,
                    args.acceleration_source,
                    args.latch_dwell,
                    args.sequential_reference,
                    args.sequential_threshold,
                ),
            )
        result[f"{args.single_method}_effectiveness_estimator"] = payload
    elif args.history_only:
        result[f"{args.architecture}_effectiveness_estimator"] = run_certificate_method(
            **common,
            estimator_torque_mode="external_effectiveness",
            effectiveness_predictor=OnlineEffectivenessPredictor(
                args.checkpoint,
                args.architecture,
                args.calibration_mode,
                args.radius_scope,
                args.fixed_deadband,
                args.acceleration_source,
                args.latch_dwell,
                args.sequential_reference,
                args.sequential_threshold,
            ),
        )
    else:
        result.update({
        "commanded_torque_estimator": run_certificate_method(
            **common, estimator_torque_mode="commanded"
        ),
        "snapshot_effectiveness_estimator": run_certificate_method(
            **common,
            estimator_torque_mode="external_effectiveness",
            effectiveness_predictor=OnlineEffectivenessPredictor(
                args.checkpoint, "snapshot"
            ),
        ),
        "history_effectiveness_estimator": run_certificate_method(
            **common,
            estimator_torque_mode="external_effectiveness",
            effectiveness_predictor=OnlineEffectivenessPredictor(
                args.checkpoint, "history"
            ),
        ),
        "oracle_applied_torque_estimator": run_certificate_method(
            **common, estimator_torque_mode="oracle_applied"
        ),
        })
    if args.include_rejected_unbiased and not args.history_only:
        result["history_unbiased_effectiveness_estimator"] = (
            run_certificate_method(
                **common,
                estimator_torque_mode="external_effectiveness_unbiased",
                effectiveness_predictor=OnlineEffectivenessPredictor(
                    args.checkpoint, "history"
                ),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
