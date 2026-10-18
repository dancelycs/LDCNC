"""Run the numerical experiment groups reported in the manuscript."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
CHECKPOINT = ROOT / "checkpoints" / "history_nominal_scenario_seed10101.pt"
MODEL = ROOT / "assets" / "seven_link_arm.xml"
OUTPUTS = ROOT / "outputs"


def execute(arguments: list[object]) -> None:
    command = [PYTHON, *map(str, arguments)]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def proposed(
    *, seed: int, scenarios: int, stress: float, profile: str, output: str,
    mode: str = "latched_nominal_correction", method: str = "history",
    radius: str = "dual", fixed_deadband: float | None = None,
) -> None:
    arguments: list[object] = [
        ROOT / "src" / "screen_learned_effectiveness_control.py",
        "--checkpoint", CHECKPOINT,
        "--model", MODEL,
        "--single-method", method,
        "--calibration-mode", mode,
        "--radius-scope", radius,
        "--seed", seed,
        "--scenarios", scenarios,
        "--stress-level", stress,
        "--fault-profile", profile,
        "--output", OUTPUTS / output,
    ]
    if fixed_deadband is not None:
        arguments.extend(("--fixed-deadband", fixed_deadband))
    execute(arguments)


def public(
    *, seed: int, scenarios: int, stress: float, profile: str, output: str,
) -> None:
    execute(
        [
            ROOT / "scripts" / "screen_public_comparators.py",
            "--model", MODEL,
            "--seed", seed,
            "--scenarios", scenarios,
            "--stress-level", stress,
            "--fault-profile", profile,
            "--output", OUTPUTS / output,
        ]
    )


def run_pulse() -> None:
    for seed in (18501, 18701):
        public(
            seed=seed, scenarios=64, stress=0.27, profile="pulse",
            output=f"public_comparators_evaluation_noleak_seed{seed}_64.json",
        )
        proposed(
            seed=seed, scenarios=64, stress=0.27, profile="pulse",
            output=f"proposed_matched_public_evaluation_seed{seed}_64.json",
        )


def run_internal() -> None:
    configurations = (
        (18201, "commanded", "point", "dual", "paper_commanded_pulse_seed18201_128.json", None),
        (18201, "snapshot", "point", "dual", "paper_snapshot_pulse_seed18201_128.json", None),
        (18201, "history", "point", "dual", "paper_history_pulse_seed18201_128.json", None),
        (18201, "history", "fixed_deadband_latch", "dual", "fixed_deadband_latch_pulse_seed18201_128.json", 0.04),
        (18201, "history", "latched_nominal_correction", "dual", "paper_latched_pulse_seed18201_128.json", None),
        (18201, "oracle", "point", "dual", "paper_oracle_pulse_seed18201_128.json", None),
        (18101, "history", "point", "dual", "stats_point_pulse_seed18101_128.json", None),
        (18101, "history", "nominal_projection", "dual", "stats_dual_anchor_pulse_seed18101_128.json", None),
        (18101, "history", "bounded_nominal_correction", "dual", "stats_bounded_nominal_pulse_seed18101_128.json", None),
        (18101, "history", "latched_nominal_correction", "dual", "stats_latched_nominal_pulse_seed18101_128.json", None),
        (18001, "history", "point", "dual", "stats_point_pulse_seed18001_128.json", None),
        (18001, "history", "latched_nominal_correction", "dual", "stats_latched_nominal_pulse_seed18001_128.json", None),
    )
    for seed, method, mode, radius, output, deadband in configurations:
        proposed(
            seed=seed, scenarios=128, stress=0.27, profile="pulse",
            output=output, method=method, mode=mode, radius=radius,
            fixed_deadband=deadband,
        )


def run_profiles() -> None:
    for profile, seed in (("ramp", 18901), ("intermittent", 19001)):
        public(
            seed=seed, scenarios=32, stress=0.27, profile=profile,
            output=f"public_comparators_{profile}_seed{seed}_32.json",
        )
        proposed(
            seed=seed, scenarios=32, stress=0.27, profile=profile,
            output=f"proposed_public_{profile}_seed{seed}_32.json",
        )
    for profile, seed in (("ramp", 18002), ("intermittent", 18003)):
        proposed(
            seed=seed, scenarios=64, stress=0.27, profile=profile,
            output=f"stats_point_{profile}_seed{seed}_64.json", mode="point",
        )
        proposed(
            seed=seed, scenarios=64, stress=0.27, profile=profile,
            output=f"stats_latched_nominal_{profile}_seed{seed}_64.json",
        )


def run_severity() -> None:
    for stress in (0.20, 0.24, 0.27, 0.31, 0.35, 0.40, 0.45):
        code = f"s{round(100 * stress):03d}"
        public(
            seed=18301, scenarios=64, stress=stress, profile="pulse",
            output=f"severity_public_{code}_seed18301_64.json",
        )
        proposed(
            seed=18301, scenarios=64, stress=stress, profile="pulse",
            output=f"severity_latched_{code}_seed18301_64.json",
        )
        proposed(
            seed=18301, scenarios=64, stress=stress, profile="pulse",
            output=f"severity_point_{code}_seed18301_64.json", mode="point",
        )


def run_robustness() -> None:
    execute(
        [
            ROOT / "scripts" / "evaluate_robustness_perturbations.py",
            "--checkpoint", CHECKPOINT,
            "--seed", 19201,
            "--scenarios", 64,
            "--output", OUTPUTS / "robustness_perturbations_seed19201_64.json",
        ]
    )


def run_sensitivity() -> None:
    execute(
        [
            ROOT / "scripts" / "evaluate_public_sensitivity.py",
            "--seed", 19301,
            "--scenarios", 32,
            "--output", OUTPUTS / "public_sensitivity_seed19301_32.json",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "suite",
        choices=("internal", "pulse", "profiles", "severity", "robustness", "sensitivity", "figures", "all"),
    )
    args = parser.parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    selected = (
        ("internal", run_internal),
        ("pulse", run_pulse),
        ("profiles", run_profiles),
        ("severity", run_severity),
        ("robustness", run_robustness),
        ("sensitivity", run_sensitivity),
    )
    for name, function in selected:
        if args.suite in (name, "all"):
            function()
    if args.suite in ("figures", "all"):
        execute([ROOT / "scripts" / "build_paper_assets.py"])


if __name__ == "__main__":
    main()
