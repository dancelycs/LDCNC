"""Screen whether short causal histories identify transient actuator loss."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = ROOT
LEGACY_ROOT = ROOT
sys.path.insert(0, str(ROOT / "src"))

from simulation_environment import (  # noqa: E402
    MAX_SPEED,
    ProgressConstrainedEnvironment,
)


class HistoryDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        history: int,
    ) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.history = history
        self.horizon = features.shape[1]
        self.samples_per_scenario = self.horizon - history + 1

    def __len__(self) -> int:
        return self.features.shape[0] * self.samples_per_scenario

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        scenario = item // self.samples_per_scenario
        local = item % self.samples_per_scenario
        end = local + self.history
        return (
            self.features[scenario, local:end],
            self.labels[scenario, end - 1],
        )


class HistoryEncoder(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.recurrent = nn.GRU(feature_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(history)
        return self.head(output[:, -1])


class SnapshotEncoder(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.network(history[:, -1])


def collect_rollouts(
    model_path: Path,
    scenarios: int,
    horizon: int,
    seed: int,
    stress_level: float,
    timestep: float,
    command_speed: float = 1.0,
    fault_profile: str = "pulse",
    collection_estimator_mode: str = "commanded",
) -> tuple[np.ndarray, np.ndarray]:
    env = ProgressConstrainedEnvironment(
        model_path,
        scenarios,
        seed,
        horizon,
        strong_switch=True,
        stress_level=stress_level,
        enforce_terminal_progress=False,
        minimum_speed=0.0,
        maximum_speed=MAX_SPEED,
        include_progress_acceleration=True,
        simulation_timestep=timestep,
        fault_profile=fault_profile,
        estimator_torque_mode=collection_estimator_mode,
    )
    feature_size = 4 * env.inverse_model.nv + 3
    features = np.zeros((scenarios, horizon, feature_size), dtype=np.float32)
    labels = np.zeros((scenarios, horizon, 4), dtype=np.float32)
    midpoint = 0.5 * MAX_SPEED
    if not 0.0 < command_speed < MAX_SPEED:
        raise ValueError("command_speed must lie strictly within the speed bounds")
    raw_speed = np.arctanh((command_speed - midpoint) / midpoint)
    raw = np.full((scenarios, 1), raw_speed)
    noise = np.random.default_rng(seed + 100_000)

    for step in range(horizon):
        env.step(raw)
        for index, sample in enumerate(env.data):
            acceleration = sample.qacc + noise.normal(
                scale=0.03, size=env.inverse_model.nv
            )
            features[index, step] = np.concatenate(
                (
                    sample.qpos,
                    sample.qvel,
                    sample.ctrl,
                    acceleration,
                    np.array(
                        [
                            env.phase[index],
                            env.last_speed[index],
                            env.innovation[index],
                        ]
                    ),
                )
            )
            effectiveness = float(env.true_effectiveness[index])
            drag = (
                env.drag_before[index]
                if env.age[index] <= env.switch_step[index]
                else env.drag_after[index]
            )
            labels[index, step] = np.array(
                [
                    effectiveness,
                    env.payload[index],
                    env.friction[index],
                    drag,
                ]
            )
    return features, labels


def collect_profile_mixture(
    model_path: Path,
    scenarios: int,
    horizon: int,
    seed: int,
    stress_level: float,
    timestep: float,
    profiles: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect a balanced scenario mixture and randomize its scenario order."""
    if not profiles:
        raise ValueError("at least one fault profile is required")
    base, remainder = divmod(scenarios, len(profiles))
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for profile_index, profile in enumerate(profiles):
        count = base + int(profile_index < remainder)
        if count == 0:
            continue
        features, labels = collect_rollouts(
            model_path,
            count,
            horizon,
            seed + 10_000 * profile_index,
            stress_level,
            timestep,
            fault_profile=profile,
            collection_estimator_mode=(
                "oracle_applied" if profile == "persistent" else "commanded"
            ),
        )
        feature_parts.append(features)
        label_parts.append(labels)
    features = np.concatenate(feature_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    order = np.random.default_rng(seed + 90_001).permutation(features.shape[0])
    return features[order], labels[order]


def normalization_statistics(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        features.mean(axis=(0, 1), keepdims=True),
        features.std(axis=(0, 1), keepdims=True) + 1.0e-6,
        labels.mean(axis=(0, 1), keepdims=True),
        labels.std(axis=(0, 1), keepdims=True) + 1.0e-6,
    )


def apply_normalization(
    features: np.ndarray,
    labels: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    label_mean: np.ndarray,
    label_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        (features - feature_mean) / feature_scale,
        (labels - label_mean) / label_scale,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    epochs: int,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    for _ in range(epochs):
        model.train()
        for history, target in train_loader:
            history = history.to(device)
            target = target.to(device)
            prediction = model(history)
            loss = torch.mean((prediction - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict(
    model: nn.Module,
    dataset: HistoryDataset,
    label_mean: np.ndarray,
    label_scale: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    for history, target in loader:
        predictions.append(model(history.to(device)).cpu().numpy())
        targets.append(target.numpy())
    prediction = np.concatenate(predictions) * label_scale + label_mean
    target = np.concatenate(targets) * label_scale + label_mean
    return prediction, target


def point_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    absolute_error = np.abs(prediction - target)
    event = target[:, 0] < 0.999
    nominal = ~event
    return {
        "effectiveness_mae_all": float(absolute_error[:, 0].mean()),
        "effectiveness_mae_event": float(absolute_error[event, 0].mean()),
        "effectiveness_mae_nominal": float(absolute_error[nominal, 0].mean()),
        "payload_mae": float(absolute_error[:, 1].mean()),
        "friction_mae": float(absolute_error[:, 2].mean()),
        "drag_mae": float(absolute_error[:, 3].mean()),
    }


def conformal_metrics(
    calibration_prediction: np.ndarray,
    calibration_target: np.ndarray,
    test_prediction: np.ndarray,
    test_target: np.ndarray,
    scenarios: int,
    timestep: float,
    coverage: float = 0.95,
    trajectory_coverage_target: float = 0.90,
) -> dict[str, float]:
    def finite_sample_radius(values: np.ndarray, target_coverage: float) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan")
        level = min(
            np.ceil((values.size + 1) * target_coverage) / values.size,
            1.0,
        )
        return float(np.quantile(values, level, method="higher"))

    scores = np.abs(
        calibration_prediction[:, 0] - calibration_target[:, 0]
    )
    radius = finite_sample_radius(scores, coverage)
    errors = np.abs(test_prediction[:, 0] - test_target[:, 0])
    empirical_coverage = float(np.mean(errors <= radius))
    samples_per_scenario = test_prediction.shape[0] // scenarios
    calibration_scenarios = calibration_prediction.shape[0] // samples_per_scenario
    calibration_errors = scores.reshape(calibration_scenarios, samples_per_scenario)
    calibration_effectiveness = calibration_target[:, 0].reshape(
        calibration_scenarios, samples_per_scenario
    )
    trajectory_radius = finite_sample_radius(
        calibration_errors.max(axis=1), trajectory_coverage_target
    )
    nominal_trajectory_scores = np.full(calibration_scenarios, np.nan)
    for scenario in range(calibration_scenarios):
        nominal = calibration_effectiveness[scenario] >= 0.999
        if np.any(nominal):
            nominal_trajectory_scores[scenario] = calibration_errors[
                scenario, nominal
            ].max()
    nominal_trajectory_radius = finite_sample_radius(
        nominal_trajectory_scores, trajectory_coverage_target
    )
    test_errors_by_scenario = errors.reshape(scenarios, samples_per_scenario)
    test_effectiveness_by_scenario = test_target[:, 0].reshape(
        scenarios, samples_per_scenario
    )
    trajectory_coverage = float(
        np.mean(test_errors_by_scenario.max(axis=1) <= trajectory_radius)
    )
    nominal_trajectory_covered: list[bool] = []
    for scenario in range(scenarios):
        nominal = test_effectiveness_by_scenario[scenario] >= 0.999
        nominal_trajectory_covered.append(
            bool(
                np.all(
                    test_errors_by_scenario[scenario, nominal]
                    <= nominal_trajectory_radius
                )
            )
        )
    predicted = test_prediction[:, 0].reshape(scenarios, samples_per_scenario)
    target = test_target[:, 0].reshape(scenarios, samples_per_scenario)
    fault = target < 0.999
    def detection_statistics(selected_radius: float) -> tuple[float, float, float]:
        detection = predicted + selected_radius < 0.999
        delays: list[float] = []
        detected = 0
        for scenario in range(scenarios):
            fault_indices = np.flatnonzero(fault[scenario])
            if fault_indices.size == 0:
                continue
            start = int(fault_indices[0])
            end = int(fault_indices[-1]) + 1
            detected_indices = np.flatnonzero(detection[scenario, start:end])
            if detected_indices.size:
                detected += 1
                delays.append(float(detected_indices[0] * timestep))
        return (
            detected / scenarios,
            float(np.median(delays)) if delays else np.nan,
            float(np.mean(detection[~fault])),
        )

    point_detection, point_delay, point_false_alarm = detection_statistics(radius)
    group_detection, group_delay, group_false_alarm = detection_statistics(
        nominal_trajectory_radius
    )
    return {
        "conformal_radius": radius,
        "empirical_coverage": empirical_coverage,
        "trajectory_conformal_radius": trajectory_radius,
        "trajectory_coverage_target": trajectory_coverage_target,
        "trajectory_empirical_coverage": trajectory_coverage,
        "nominal_trajectory_conformal_radius": nominal_trajectory_radius,
        "nominal_trajectory_empirical_coverage": float(
            np.mean(nominal_trajectory_covered)
        ),
        "fault_detection_rate": point_detection,
        "median_detection_delay_s": point_delay,
        "nominal_false_alarm_rate": point_false_alarm,
        "group_fault_detection_rate": group_detection,
        "group_median_detection_delay_s": group_delay,
        "group_nominal_false_alarm_rate": group_false_alarm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-scenarios", type=int, default=96)
    parser.add_argument("--test-scenarios", type=int, default=32)
    parser.add_argument("--calibration-scenarios", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--history", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument(
        "--mask-phase", action="store_true",
        help="remove absolute task progress from estimator inputs",
    )
    parser.add_argument("--stress-level", type=float, default=0.27)
    parser.add_argument("--timestep", type=float, default=0.005)
    parser.add_argument("--train-seed", type=int, default=10101)
    parser.add_argument("--test-seed", type=int, default=10201)
    parser.add_argument(
        "--fault-profiles",
        type=str,
        default="pulse",
        help="comma-separated balanced training and test fault profiles",
    )
    parser.add_argument(
        "--model", type=Path, default=ROOT / "assets" / "seven_link_arm.xml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "history_identifiability.json"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)
    profiles = tuple(
        profile.strip() for profile in args.fault_profiles.split(",")
        if profile.strip()
    )
    valid_profiles = {"pulse", "ramp", "intermittent", "persistent"}
    if not profiles or any(profile not in valid_profiles for profile in profiles):
        raise ValueError(f"fault profiles must belong to {sorted(valid_profiles)}")
    train_features, train_labels = collect_profile_mixture(
        args.model,
        args.train_scenarios,
        args.horizon,
        args.train_seed,
        args.stress_level,
        args.timestep,
        profiles,
    )
    test_features, test_labels = collect_profile_mixture(
        args.model,
        args.test_scenarios,
        args.horizon,
        args.test_seed,
        args.stress_level,
        args.timestep,
        profiles,
    )
    if args.mask_phase:
        train_features[:, :, -3] = 0.0
        test_features[:, :, -3] = 0.0
    if not 0 < args.calibration_scenarios < args.train_scenarios:
        raise ValueError("calibration_scenarios must be within the training set")
    fit_count = args.train_scenarios - args.calibration_scenarios
    fit_features = train_features[:fit_count]
    fit_labels = train_labels[:fit_count]
    calibration_features = train_features[fit_count:]
    calibration_labels = train_labels[fit_count:]
    feature_mean, feature_scale, label_mean_full, label_scale_full = (
        normalization_statistics(fit_features, fit_labels)
    )
    fit_features, fit_labels = apply_normalization(
        fit_features,
        fit_labels,
        feature_mean,
        feature_scale,
        label_mean_full,
        label_scale_full,
    )
    calibration_features, calibration_labels = apply_normalization(
        calibration_features,
        calibration_labels,
        feature_mean,
        feature_scale,
        label_mean_full,
        label_scale_full,
    )
    test_features, test_labels = apply_normalization(
        test_features,
        test_labels,
        feature_mean,
        feature_scale,
        label_mean_full,
        label_scale_full,
    )
    label_mean = label_mean_full.reshape(-1)
    label_scale = label_scale_full.reshape(-1)
    train_dataset = HistoryDataset(
        fit_features, fit_labels, args.history
    )
    calibration_dataset = HistoryDataset(
        calibration_features, calibration_labels, args.history
    )
    test_dataset = HistoryDataset(test_features, test_labels, args.history)
    generator = torch.Generator().manual_seed(args.train_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recurrent = HistoryEncoder(
        fit_features.shape[-1], args.hidden_size, fit_labels.shape[-1]
    )
    snapshot = SnapshotEncoder(
        fit_features.shape[-1], args.hidden_size, fit_labels.shape[-1]
    )
    train_model(recurrent, train_loader, device, args.epochs)
    train_model(snapshot, train_loader, device, args.epochs)
    recurrent_calibration = predict(
        recurrent, calibration_dataset, label_mean, label_scale, device
    )
    recurrent_test = predict(
        recurrent, test_dataset, label_mean, label_scale, device
    )
    snapshot_calibration = predict(
        snapshot, calibration_dataset, label_mean, label_scale, device
    )
    snapshot_test = predict(
        snapshot, test_dataset, label_mean, label_scale, device
    )
    result = {
        "device": str(device),
        "history_seconds": args.history * args.timestep,
        "train_scenarios": args.train_scenarios,
        "test_scenarios": args.test_scenarios,
        "calibration_scenarios": args.calibration_scenarios,
        "fault_profiles": list(profiles),
        "recurrent_history": {
            **point_metrics(*recurrent_test),
            **conformal_metrics(
                *recurrent_calibration,
                *recurrent_test,
                args.test_scenarios,
                args.timestep,
            ),
        },
        "snapshot": {
            **point_metrics(*snapshot_test),
            **conformal_metrics(
                *snapshot_calibration,
                *snapshot_test,
                args.test_scenarios,
                args.timestep,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": recurrent.state_dict(),
                "snapshot_state_dict": snapshot.state_dict(),
                "feature_mean": feature_mean.reshape(-1),
                "feature_scale": feature_scale.reshape(-1),
                "label_mean": label_mean,
                "label_scale": label_scale,
                "history": args.history,
                "hidden_size": args.hidden_size,
                "feature_size": fit_features.shape[-1],
                "output_size": fit_labels.shape[-1],
                "mask_phase": args.mask_phase,
                "fault_profiles": list(profiles),
                "conformal_radius": result["recurrent_history"][
                    "conformal_radius"
                ],
                "trajectory_conformal_radius": result["recurrent_history"][
                    "trajectory_conformal_radius"
                ],
                "nominal_trajectory_conformal_radius": result[
                    "recurrent_history"
                ]["nominal_trajectory_conformal_radius"],
            },
            args.checkpoint,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
