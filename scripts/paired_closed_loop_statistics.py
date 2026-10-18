"""Paired scenario statistics for two closed-loop result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def method_payload(document: dict[str, object]) -> dict[str, object]:
    candidates = [
        value
        for key, value in document.items()
        if key.endswith("_effectiveness_estimator") and isinstance(value, dict)
    ]
    if len(candidates) != 1:
        raise ValueError("result file must contain exactly one evaluated estimator")
    return candidates[0]


def bootstrap_interval(
    difference: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    count = difference.size
    draws = rng.integers(0, count, size=(replicates, count))
    means = difference[draws].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=17001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_document = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate_document = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline = method_payload(baseline_document)
    candidate = method_payload(candidate_document)
    baseline_details = baseline["scenario_details"]
    candidate_details = candidate["scenario_details"]
    if len(baseline_details) != len(candidate_details):
        raise ValueError("paired files have different scenario counts")
    if any(
        left["scenario"] != right["scenario"]
        for left, right in zip(baseline_details, candidate_details)
    ):
        raise ValueError("scenario identifiers are not aligned")

    rng = np.random.default_rng(args.seed)
    metrics = (
        "tracking_rmse",
        "mean_stage_cost",
        "maximum_error",
        "completion_step",
        "mean_certificate_violation",
    )
    result: dict[str, object] = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "scenarios": len(baseline_details),
        "difference_definition": "candidate_minus_baseline",
        "metrics": {},
    }
    for metric in metrics:
        baseline_values = np.asarray(
            [row[metric] for row in baseline_details], dtype=float
        )
        candidate_values = np.asarray(
            [row[metric] for row in candidate_details], dtype=float
        )
        difference = candidate_values - baseline_values
        lower, upper = bootstrap_interval(difference, rng, args.replicates)
        result["metrics"][metric] = {
            "baseline_mean": float(baseline_values.mean()),
            "candidate_mean": float(candidate_values.mean()),
            "paired_mean_difference": float(difference.mean()),
            "bootstrap_95_interval": [lower, upper],
            "candidate_better_fraction": float(np.mean(difference < 0.0)),
            "equal_fraction": float(np.mean(difference == 0.0)),
        }
    baseline_failure = np.asarray(
        [row["first_failure_step"] >= 0 for row in baseline_details]
    )
    candidate_failure = np.asarray(
        [row["first_failure_step"] >= 0 for row in candidate_details]
    )
    result["failure_discordance"] = {
        "baseline_only_failures": int(
            np.sum(baseline_failure & ~candidate_failure)
        ),
        "candidate_only_failures": int(
            np.sum(~baseline_failure & candidate_failure)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
