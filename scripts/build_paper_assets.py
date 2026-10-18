"""Build publication tables and vector figures from immutable JSON results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paired_closed_loop_statistics import method_payload


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reference_results"
GENERATED = ROOT / "outputs"
PAPER = ROOT / "outputs" / "paper_assets"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_path(name: str) -> Path:
    generated = GENERATED / name
    return generated if generated.exists() else RESULTS / name


def payload(name: str) -> dict[str, object]:
    return method_payload(read(result_path(name)))


def details(name: str) -> list[dict[str, object]]:
    return payload(name)["scenario_details"]


def paired_differences(baseline: str, candidate: str, metric: str) -> np.ndarray:
    left = details(baseline)
    right = details(candidate)
    return np.asarray([b[metric] - a[metric] for a, b in zip(left, right)])


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


PUBLIC_LABELS = {
    "sftsmc_eso": r"SFTSMC--ESO~\cite{le2022sftsmc}",
    "dl_rbf": r"DL--RBF~\cite{zhang2023dynamic}",
    "pet_aftc": r"PET--AFTC~\cite{fan2023triggering}",
    "fasmo_cftc": r"FASMO--CFTC~\cite{anjum2024disturbance}",
    "le_ftc": r"LE--FTC~\cite{huang2025learning}",
}

SEVERITY_TABLE_GRID = (
    ("s020", 0.20), ("s027", 0.27), ("s035", 0.35), ("s045", 0.45),
)
SEVERITY_PLOT_GRID = (
    ("s020", 0.20), ("s024", 0.24), ("s027", 0.27),
    ("s031", 0.31), ("s035", 0.35), ("s040", 0.40),
    ("s045", 0.45),
)


def public_document(name: str) -> dict[str, object]:
    return read(result_path(name))["methods"]


def public_rows(method: str, names: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in names:
        rows.extend(public_document(name)[method]["result"]["scenario_details"])
    return rows


def proposed_rows(names: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in names:
        rows.extend(read(result_path(name))["history_effectiveness_estimator"]["scenario_details"])
    return rows


def row_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    rmse = np.asarray([row["tracking_rmse"] for row in rows], dtype=float)
    cost = np.asarray([row["mean_stage_cost"] for row in rows], dtype=float)
    maximum = np.asarray([row["maximum_error"] for row in rows], dtype=float)
    failure = np.asarray([row["first_failure_step"] >= 0 for row in rows], dtype=float)
    tail = max(1, int(np.ceil(0.10 * cost.size)))
    return {
        "rmse": float(rmse.mean()),
        "cost": float(cost.mean()),
        "cvar": float(np.sort(cost)[-tail:].mean()),
        "failure": float(failure.mean()),
        "maximum": float(maximum.max()),
    }


def bold_best(values: list[float], minimize: bool = True) -> list[str]:
    target = min(values) if minimize else max(values)
    return [
        f"\\textbf{{{fmt(value)}}}" if np.isclose(value, target) else fmt(value)
        for value in values
    ]


def write_public_comparison() -> None:
    public_files = [
        "public_comparators_evaluation_noleak_seed18501_64.json",
        "public_comparators_evaluation_noleak_seed18701_64.json",
    ]
    proposed_files = [
        "proposed_matched_public_evaluation_seed18501_64.json",
        "proposed_matched_public_evaluation_seed18701_64.json",
    ]
    names = list(PUBLIC_LABELS) + ["proposed"]
    rows_by_method = {
        name: (
            proposed_rows(proposed_files)
            if name == "proposed"
            else public_rows(name, public_files)
        )
        for name in names
    }
    summaries = {name: row_summary(rows) for name, rows in rows_by_method.items()}
    rmse_text = bold_best([summaries[name]["rmse"] for name in names])
    cost_text = bold_best([summaries[name]["cost"] for name in names])
    cvar_text = bold_best([summaries[name]["cvar"] for name in names])
    failure_text = bold_best([summaries[name]["failure"] for name in names])
    maximum_text = bold_best([summaries[name]["maximum"] for name in names])
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Matched-interface comparison with recent published methods over two independent pulse-fault sets (128 scenarios).}",
        "\\label{tab:public_comparison}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Scenario RMSE & Stage cost & Cost CVaR$_{0.9}$ & Failure (\\%) & Divergence (\\%) & Max. error " + "\\\\",
        "\\midrule",
    ]
    for index, name in enumerate(names):
        label = "Proposed" if name == "proposed" else PUBLIC_LABELS[name]
        divergent = any(
            (not np.isfinite(float(row["maximum_error"])))
            or float(row["maximum_error"]) > 1.0e6
            for row in rows_by_method[name]
        )
        values = [
            "DVG" if divergent else rmse_text[index],
            "DVG" if divergent else cost_text[index],
            "DVG" if divergent else cvar_text[index],
            (
                f"\\textbf{{{100.0 * summaries[name]['failure']:.2f}}}"
                if summaries[name]["failure"]
                == min(summaries[item]["failure"] for item in names)
                else f"{100.0 * summaries[name]['failure']:.2f}"
            ),
            f"{100.0 * np.mean([float(row['maximum_error']) > 1.0e6 for row in rows_by_method[name]]):.2f}",
            "DVG" if divergent else maximum_text[index],
        ]
        lines.append(label + " & " + " & ".join(values) + " " + "\\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\begin{minipage}{0.96\\textwidth}\\footnotesize",
            "All published methods are independent seven-link adaptations from the cited equations, not official author implementations. DVG denotes at least one numerically divergent rollout; bold denotes the best finite value.",
            "\\end{minipage}",
            "\\end{table*}",
        ]
    )
    (TABLES / "public_comparison.tex").write_text("\n".join(lines) + "\n")

    rng = np.random.default_rng(19101)
    candidate_groups = [
        proposed_rows([proposed_files[0]]),
        proposed_rows([proposed_files[1]]),
    ]
    statistics: dict[str, object] = {"scenarios": 128, "replicates": 20_000, "methods": {}}
    for method in PUBLIC_LABELS:
        baseline_groups = [
            public_rows(method, [public_files[0]]),
            public_rows(method, [public_files[1]]),
        ]
        method_stats: dict[str, object] = {}
        for metric in ("tracking_rmse", "mean_stage_cost"):
            differences = []
            bootstrap_parts = []
            for baseline, candidate in zip(baseline_groups, candidate_groups):
                delta = np.asarray(
                    [right[metric] - left[metric] for left, right in zip(baseline, candidate)],
                    dtype=float,
                )
                differences.append(delta)
                indices = rng.integers(0, delta.size, size=(20_000, delta.size))
                bootstrap_parts.append(delta[indices].mean(axis=1))
            combined = np.concatenate(differences)
            boot = np.mean(np.stack(bootstrap_parts, axis=1), axis=1)
            method_stats[metric] = {
                "paired_mean_difference": float(combined.mean()),
                "stratified_bootstrap_95_interval": [
                    float(np.quantile(boot, 0.025)),
                    float(np.quantile(boot, 0.975)),
                ],
            }
        baseline_failure = np.concatenate(
            [
                np.asarray([row["first_failure_step"] >= 0 for row in group])
                for group in baseline_groups
            ]
        )
        proposed_failure = np.concatenate(
            [
                np.asarray([row["first_failure_step"] >= 0 for row in group])
                for group in candidate_groups
            ]
        )
        method_stats["discordant_failures"] = {
            "comparator_only": int(np.sum(baseline_failure & ~proposed_failure)),
            "proposed_only": int(np.sum(~baseline_failure & proposed_failure)),
        }
        statistics["methods"][method] = method_stats
    (TABLES / "public_comparator_statistics_128.json").write_text(
        json.dumps(statistics, indent=2) + "\n"
    )


def profile_summary(public_file: str, proposed_file: str) -> dict[str, dict[str, float]]:
    result = {
        method: row_summary(public_rows(method, [public_file]))
        for method in PUBLIC_LABELS
    }
    result["proposed"] = row_summary(proposed_rows([proposed_file]))
    return result


def write_profile_table() -> None:
    profiles = {
        "Nominal": profile_summary(
            "public_comparators_nominal_noleak_seed18801_32.json",
            "proposed_nominal_seed18801_32.json",
        ),
        "Ramp": profile_summary(
            "public_comparators_ramp_seed18901_32.json",
            "proposed_public_ramp_seed18901_32.json",
        ),
        "Intermittent": profile_summary(
            "public_comparators_intermittent_seed19001_32.json",
            "proposed_public_intermittent_seed19001_32.json",
        ),
    }
    names = list(PUBLIC_LABELS) + ["proposed"]
    best = {
        profile: min(values[name]["rmse"] for name in names)
        for profile, values in profiles.items()
    }
    fault_mean = {
        name: 0.5 * (
            profiles["Ramp"][name]["rmse"]
            + profiles["Intermittent"][name]["rmse"]
        )
        for name in names
    }
    fault_failures = {
        name: int(round(32 * (
            profiles["Ramp"][name]["failure"]
            + profiles["Intermittent"][name]["failure"]
        )))
        for name in names
    }
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Tracking RMSE across operating profiles and aggregate fault-profile robustness (32 scenarios per profile).}",
        "\\label{tab:public_profiles}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Nominal & Ramp & Intermittent & Fault-profile mean & Fault failures " + "\\\\",
        "\\midrule",
    ]
    for name in names:
        label = "Proposed" if name == "proposed" else PUBLIC_LABELS[name]
        values = []
        for profile in profiles:
            value = profiles[profile][name]["rmse"]
            text = fmt(value)
            if np.isclose(value, best[profile]):
                text = f"\\textbf{{{text}}}"
            values.append(text)
        aggregate = fmt(fault_mean[name])
        failures = str(fault_failures[name])
        if np.isclose(fault_mean[name], min(fault_mean.values())):
            aggregate = f"\\textbf{{{aggregate}}}"
        if fault_failures[name] == min(fault_failures.values()):
            failures = f"\\textbf{{{failures}}}"
        values.extend([aggregate, failures])
        lines.append(label + " & " + " & ".join(values) + " " + "\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    (TABLES / "public_profiles.tex").write_text("\n".join(lines) + "\n")


def write_main_table() -> None:
    methods = [
        ("Commanded-torque RLS", "paper_commanded_pulse_seed18201_128.json"),
        ("Snapshot estimator", "paper_snapshot_pulse_seed18201_128.json"),
        ("History point estimate", "paper_history_pulse_seed18201_128.json"),
        ("Fixed deadband + latch", "fixed_deadband_latch_pulse_seed18201_128.json"),
        ("Proposed", "paper_latched_pulse_seed18201_128.json"),
        ("Oracle applied torque$^{\\dagger}$", "paper_oracle_pulse_seed18201_128.json"),
    ]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Matched pulse-fault comparison over 128 scenarios. RMSE is the arithmetic mean of the scenario-level RMSE values.}",
        "\\label{tab:main_comparison}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & RMSE & Stage cost & Cost CVaR$_{0.9}$ & Failure (\\%) & Max. error & Completion (s) \\\\",
        "\\midrule",
    ]
    rows_by_method = []
    for label, filename in methods:
        item = payload(filename)
        rows = item["scenario_details"]
        rows_by_method.append((label, [
            float(np.mean([row["tracking_rmse"] for row in rows])),
            float(item["mean_stage_cost"]), float(item["stage_cost_cvar_90"]),
            100.0 * float(item["failure_rate"]),
            float(item["maximum_tracking_error"]),
            float(item["mean_completion_time"])]))
    deployable = np.asarray([values for label, values in rows_by_method
                             if "Oracle" not in label])
    best = deployable.min(axis=0)
    for label, numerical in rows_by_method:
        digits = (4, 4, 4, 2, 4, 3)
        values = [fmt(value, digit) for value, digit in zip(numerical, digits)]
        if "Oracle" not in label:
            values = [f"\\textbf{{{text}}}" if np.isclose(value, optimum)
                      else text for text, value, optimum
                      in zip(values, numerical, best)]
        lines.append(label + " & " + " & ".join(values) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\begin{minipage}{0.96\\textwidth}\\footnotesize",
            "$^{\\dagger}$The oracle is a same-state counterfactual identifier update and is not deployable. "
            "Bold identifies the best deployable entry in each column.",
            "\\end{minipage}",
            "\\end{table*}",
        ]
    )
    (TABLES / "main_comparison.tex").write_text("\n".join(lines) + "\n")


def write_ablation_table() -> None:
    methods = [
        ("History point", "stats_point_pulse_seed18101_128.json"),
        ("Hard projection", "stats_dual_anchor_pulse_seed18101_128.json"),
        ("Bounded correction", "stats_bounded_nominal_pulse_seed18101_128.json"),
        ("Bounded + latch", "stats_latched_nominal_pulse_seed18101_128.json"),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Interface ablation on the independent falsification set.}",
        "\\label{tab:interface_ablation}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Interface & RMSE & Cost & Failure (\\%) & Max. error \\\\",
        "\\midrule",
    ]
    for label, filename in methods:
        item = payload(filename)
        rows = item["scenario_details"]
        rmse = np.mean([row["tracking_rmse"] for row in rows])
        maximum = float(item["maximum_tracking_error"])
        values = [
            fmt(float(rmse)),
            fmt(float(item["mean_stage_cost"])),
            fmt(100.0 * float(item["failure_rate"]), 2),
            fmt(maximum),
        ]
        if label == "Bounded + latch":
            values = [f"\\textbf{{{value}}}" for value in values]
        lines.append(label + " & " + " & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (TABLES / "ablation.tex").write_text("\n".join(lines) + "\n")


def write_generalization_table_legacy() -> None:
    rows = [
        ("Pulse, three seeds", "paper_latched_vs_history_pulse_three_seeds_384.json", "stratified_bootstrap_95_interval"),
        ("Ramp", "stats_latched_nominal_vs_point_ramp_seed18002_64.json", "bootstrap_95_interval"),
        ("Intermittent", "stats_latched_nominal_vs_point_intermittent_seed18003_64.json", "bootstrap_95_interval"),
    ]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Paired generalization results; differences are proposed minus point estimate.}",
        "\\label{tab:generalization}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Profile & Scenarios & $\\Delta$RMSE [95\\% CI] & $\\Delta$cost [95\\% CI] \\\\",
        "\\midrule",
    ]
    for label, filename, interval_key in rows:
        item = read(result_path(filename))
        rmse = item["metrics"]["tracking_rmse"]
        cost = item["metrics"]["mean_stage_cost"]
        ri = rmse[interval_key]
        ci = cost[interval_key]
        rmse_text = f"{rmse['paired_mean_difference']:.5f} [{ri[0]:.5f}, {ri[1]:.5f}]"
        cost_text = f"{cost['paired_mean_difference']:.5f} [{ci[0]:.5f}, {ci[1]:.5f}]"
        lines.append(f"{label} & {item['scenarios']} & {rmse_text} & {cost_text} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    (TABLES / "generalization.tex").write_text("\n".join(lines) + "\n")


def write_severity_table_legacy() -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Frozen-method fault-severity boundary study.}",
        "\\label{tab:severity}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        "Stress & Point RMSE & Proposed RMSE & Point fail. (\\%) & Proposed fail. (\\%) & Paired $\\Delta$RMSE \\\\",
        "\\midrule",
    ]
    for code, stress in (("s020", 0.20), ("s027", 0.27), ("s035", 0.35), ("s045", 0.45)):
        point = payload(f"severity_point_{code}_seed18301_64.json")
        proposed = payload(f"severity_latched_{code}_seed18301_64.json")
        paired = read(result_path(f"severity_latched_vs_point_{code}_seed18301_64.json"))
        delta = paired["metrics"]["tracking_rmse"]["paired_mean_difference"]
        lines.append(
            f"{stress:.2f} & {point['tracking_rmse']:.4f} & {proposed['tracking_rmse']:.4f} & "
            f"{100*point['failure_rate']:.2f} & {100*proposed['failure_rate']:.2f} & {delta:.5f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    (TABLES / "severity.tex").write_text("\n".join(lines) + "\n")


def public_profile_pairs() -> dict[str, dict[str, tuple[list[dict], list[dict]]]]:
    specifications = {
        "Pulse": (
            ["public_comparators_evaluation_noleak_seed18501_64.json",
             "public_comparators_evaluation_noleak_seed18701_64.json"],
            ["proposed_matched_public_evaluation_seed18501_64.json",
             "proposed_matched_public_evaluation_seed18701_64.json"]),
        "Ramp": (["public_comparators_ramp_seed18901_32.json"],
                 ["proposed_public_ramp_seed18901_32.json"]),
        "Intermittent": (["public_comparators_intermittent_seed19001_32.json"],
                         ["proposed_public_intermittent_seed19001_32.json"]),
    }
    return {
        profile: {
            method: (public_rows(method, public_files),
                     proposed_rows(proposed_files))
            for method in PUBLIC_LABELS
        }
        for profile, (public_files, proposed_files) in specifications.items()
    }


def bootstrap_interval(delta: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, delta.size, size=(20_000, delta.size))
    values = delta[indices].mean(axis=1)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def write_generalization_table() -> None:
    profiles = public_profile_pairs()
    rng = np.random.default_rng(23201)
    lines = [
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Paired generalization against all recent published-method adaptations. Entries are proposed-minus-comparator mean differences with bootstrap 95\\% intervals; bold intervals favor the proposed method.}",
        "\\label{tab:generalization}", "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Method & Pulse $\\Delta$RMSE & Pulse $\\Delta J$ & Ramp $\\Delta$RMSE & Ramp $\\Delta J$ & Intermittent $\\Delta$RMSE & Intermittent $\\Delta J$ " + "\\\\",
        "\\midrule"]
    for method in PUBLIC_LABELS:
        cells = []
        for profile in ("Pulse", "Ramp", "Intermittent"):
            baseline, candidate = profiles[profile][method]
            for metric in ("tracking_rmse", "mean_stage_cost"):
                divergent = any((not np.isfinite(item["maximum_error"])) or
                                item["maximum_error"] > 1.0e6
                                for item in baseline)
                if divergent:
                    cells.append("DVG")
                    continue
                delta = np.asarray([right[metric] - left[metric]
                                    for left, right in zip(baseline, candidate)])
                low, high = bootstrap_interval(delta, rng)
                cell = f"{delta.mean():.4f} [{low:.4f}, {high:.4f}]"
                if high < 0:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
        lines.append(PUBLIC_LABELS[method] + " & " + " & ".join(cells)
                     + " " + "\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    (TABLES / "generalization.tex").write_text("\n".join(lines) + "\n")


def severity_summaries(
    grid: tuple[tuple[str, float], ...] = SEVERITY_TABLE_GRID,
) -> dict[str, list[dict[str, float]]]:
    codes = tuple(code for code, _ in grid)
    names = ["point", *PUBLIC_LABELS, "proposed"]
    summaries: dict[str, list[dict[str, float]]] = {name: [] for name in names}
    for code in codes:
        summaries["point"].append(row_summary(details(
            f"severity_point_{code}_seed18301_64.json")))
        summaries["proposed"].append(row_summary(details(
            f"severity_latched_{code}_seed18301_64.json")))
        document = public_document(f"severity_public_{code}_seed18301_64.json")
        for method in PUBLIC_LABELS:
            result = document[method]["result"]
            item = row_summary(result["scenario_details"])
            item["divergence"] = float(result.get("divergence_rate", 0.0))
            summaries[method].append(item)
        summaries["point"][-1]["divergence"] = 0.0
        summaries["proposed"][-1]["divergence"] = 0.0
    return summaries


def write_severity_table() -> None:
    summaries = severity_summaries(SEVERITY_TABLE_GRID)
    names = list(summaries)
    best_rmse = [min(summaries[name][i]["rmse"] for name in names)
                 for i in range(len(SEVERITY_TABLE_GRID))]
    best_failure = [min(summaries[name][i]["failure"] for name in names)
                    for i in range(len(SEVERITY_TABLE_GRID))]
    lines = [
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Fixed-parameter comparison across fault severity. Each pair reports RMSE/failure rate (\\%); bold denotes the best value at that severity.}",
        "\\label{tab:severity}", "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
        "Method & \\multicolumn{2}{c}{Stress 0.20} & \\multicolumn{2}{c}{Stress 0.27} & \\multicolumn{2}{c}{Stress 0.35} & \\multicolumn{2}{c}{Stress 0.45} " + "\\\\",
        " & RMSE & Fail. & RMSE & Fail. & RMSE & Fail. & RMSE & Fail. " + "\\\\",
        "\\midrule"]
    for name in names:
        label = "History point" if name == "point" else (
            "Proposed" if name == "proposed" else PUBLIC_LABELS[name])
        cells = []
        for index, item in enumerate(summaries[name]):
            rmse = "DVG" if item.get("divergence", 0.0) > 0 else fmt(item["rmse"])
            failure = f"{100 * item['failure']:.2f}"
            if item.get("divergence", 0.0) == 0 and np.isclose(
                    item["rmse"], best_rmse[index]):
                rmse = f"\\textbf{{{rmse}}}"
            if np.isclose(item["failure"], best_failure[index]):
                failure = f"\\textbf{{{failure}}}"
            cells.extend([rmse, failure])
        lines.append(label + " & " + " & ".join(cells) + " " + "\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    (TABLES / "severity.tex").write_text("\n".join(lines) + "\n")


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_paired_differences_legacy() -> None:
    groups = [
        ("Pulse-1", "stats_point_pulse_seed18001_128.json", "stats_latched_nominal_pulse_seed18001_128.json"),
        ("Pulse-2", "stats_point_pulse_seed18101_128.json", "stats_latched_nominal_pulse_seed18101_128.json"),
        ("Pulse-3", "paper_history_pulse_seed18201_128.json", "paper_latched_pulse_seed18201_128.json"),
        ("Ramp", "stats_point_ramp_seed18002_64.json", "stats_latched_nominal_ramp_seed18002_64.json"),
        ("Intermittent", "stats_point_intermittent_seed18003_64.json", "stats_latched_nominal_intermittent_seed18003_64.json"),
    ]
    rng = np.random.default_rng(23201)
    metrics = {
        "rmse": [paired_differences(left, right, "tracking_rmse")
                 for _, left, right in groups],
        "cost": [paired_differences(left, right, "mean_stage_cost")
                 for _, left, right in groups],
        "completion": [
            0.005 * paired_differences(left, right, "completion_step")
            for _, left, right in groups],
    }
    labels = [item[0] for item in groups]

    def intervals(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means, lower, upper = [], [], []
        for delta in values:
            indices = rng.integers(0, delta.size, size=(20_000, delta.size))
            boot = delta[indices].mean(axis=1)
            means.append(delta.mean())
            lower.append(np.quantile(boot, 0.025))
            upper.append(np.quantile(boot, 0.975))
        return np.asarray(means), np.asarray(lower), np.asarray(upper)

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.65),
                             constrained_layout=True)
    y = np.arange(len(groups))
    for panel, axis, key, xlabel in [
        ("a", axes[0, 0], "rmse", r"$\Delta$ tracking RMSE"),
        ("b", axes[0, 1], "cost", r"$\Delta$ stage cost"),
        ("d", axes[1, 1], "completion", r"$\Delta$ completion time (s)")]:
        mean, lower, upper = intervals(metrics[key])
        axis.errorbar(mean, y,
                      xerr=np.vstack((mean - lower, upper - mean)),
                      fmt="o", color="#0072B2", markersize=5, capsize=3)
        axis.axvline(0, color="#B22222", linewidth=0.9, linestyle="--")
        axis.set_yticks(y, labels)
        axis.set_xlabel(xlabel + r" (proposed $-$ point)")
        axis.invert_yaxis()
        axis.grid(color="0.88", linewidth=0.5)
        axis.text(0.015, 0.97, rf"\textbf{{({panel})}}",
                  transform=axis.transAxes, va="top")
    joint = [100 * np.mean((rmse < 0) & (cost < 0))
             for rmse, cost in zip(metrics["rmse"], metrics["cost"])]
    axes[1, 0].bar(np.arange(5), joint, color="#59A14F", width=0.62)
    axes[1, 0].set_xticks(np.arange(5), labels, rotation=18, ha="right")
    axes[1, 0].set(ylabel=r"Joint improvement (\%)", ylim=(0, 105))
    axes[1, 0].grid(axis="y", color="0.88", linewidth=0.5)
    axes[1, 0].text(0.015, 0.97, r"\textbf{(c)}",
                    transform=axes[1, 0].transAxes, va="top")
    fig.savefig(FIGURES / "paired_rmse_differences.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "paired_rmse_differences.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_severity_legacy() -> None:
    stresses = np.asarray([0.20, 0.27, 0.35, 0.45])
    point_rmse, proposed_rmse, point_cost, proposed_cost = [], [], [], []
    point_fail, proposed_fail, deltas = [], [], []
    rng = np.random.default_rng(23301)
    for code in ("s020", "s027", "s035", "s045"):
        point = payload(f"severity_point_{code}_seed18301_64.json")
        proposed = payload(f"severity_latched_{code}_seed18301_64.json")
        point_rmse.append(point["tracking_rmse"])
        proposed_rmse.append(proposed["tracking_rmse"])
        point_cost.append(point["mean_stage_cost"])
        proposed_cost.append(proposed["mean_stage_cost"])
        point_fail.append(100.0 * point["failure_rate"])
        proposed_fail.append(100.0 * proposed["failure_rate"])
        deltas.append(paired_differences(
            f"severity_point_{code}_seed18301_64.json",
            f"severity_latched_{code}_seed18301_64.json",
            "tracking_rmse"))
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.45),
                             sharex=True, constrained_layout=True)
    styles = (("History point", "#777777", "o"), ("Proposed", "#1F77B4", "s"))
    for axis, point_values, proposed_values, ylabel in [
        (axes[0, 0], point_rmse, proposed_rmse, "Tracking RMSE"),
        (axes[0, 1], point_cost, proposed_cost, "Mean stage cost"),
        (axes[1, 0], point_fail, proposed_fail, r"Failure rate (\%)")]:
        axis.plot(stresses, point_values, color=styles[0][1],
                  marker=styles[0][2], label=styles[0][0])
        axis.plot(stresses, proposed_values, color=styles[1][1],
                  marker=styles[1][2], label=styles[1][0])
        axis.set_ylabel(ylabel)
    axes[0, 0].legend(loc="upper right", frameon=True, ncol=2)
    means, lower, upper = [], [], []
    for delta in deltas:
        indices = rng.integers(0, delta.size, size=(20_000, delta.size))
        boot = delta[indices].mean(axis=1)
        means.append(delta.mean())
        lower.append(np.quantile(boot, 0.025))
        upper.append(np.quantile(boot, 0.975))
    means = np.asarray(means)
    axes[1, 1].errorbar(stresses, means,
                        yerr=np.vstack((means - lower,
                                        np.asarray(upper) - means)),
                        color="#0072B2", marker="o", capsize=3)
    axes[1, 1].axhline(0, color="#B22222", linestyle="--", linewidth=0.9)
    axes[1, 1].set_ylabel(r"Paired $\Delta$RMSE")
    for index, ax in enumerate(axes.flat):
        ax.grid(color="0.88", linewidth=0.5)
        ax.text(0.015, 0.97, rf"\textbf{{({chr(ord('a') + index)})}}",
                transform=ax.transAxes, va="top")
    axes[1, 0].set_xlabel("Fault stress level")
    axes[1, 1].set_xlabel("Fault stress level")
    fig.savefig(FIGURES / "severity_boundary.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "severity_boundary.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_paired_differences() -> None:
    profiles = public_profile_pairs()
    method_names = list(PUBLIC_LABELS)
    profile_names = ["Pulse", "Ramp", "Intermittent"]
    rmse_delta = np.full((5, 3), np.nan)
    cost_delta = np.full((5, 3), np.nan)
    rmse_low = np.full((5, 3), np.nan)
    rmse_high = np.full((5, 3), np.nan)
    cost_low = np.full((5, 3), np.nan)
    cost_high = np.full((5, 3), np.nan)
    joint_win = np.zeros((5, 3))
    failure_advantage = np.zeros((5, 3))
    proposed_only_failure = np.zeros((5, 3))
    rng = np.random.default_rng(23901)
    for row, method in enumerate(method_names):
        for column, profile in enumerate(profile_names):
            baseline, proposed = profiles[profile][method]
            divergent = any((not np.isfinite(item["maximum_error"])) or
                            item["maximum_error"] > 1.0e6
                            for item in baseline)
            dr = np.asarray([right["tracking_rmse"] - left["tracking_rmse"]
                             for left, right in zip(baseline, proposed)])
            dc = np.asarray([right["mean_stage_cost"] - left["mean_stage_cost"]
                             for left, right in zip(baseline, proposed)])
            if not divergent:
                rmse_delta[row, column] = dr.mean()
                cost_delta[row, column] = dc.mean()
                rmse_low[row, column], rmse_high[row, column] = \
                    bootstrap_interval(dr, rng)
                cost_low[row, column], cost_high[row, column] = \
                    bootstrap_interval(dc, rng)
            joint_win[row, column] = 100 * np.mean((dr < 0) & (dc < 0))
            base_fail = np.asarray([item["first_failure_step"] >= 0
                                    for item in baseline])
            prop_fail = np.asarray([item["first_failure_step"] >= 0
                                    for item in proposed])
            failure_advantage[row, column] = np.sum(base_fail & ~prop_fail)
            proposed_only_failure[row, column] = np.sum(~base_fail & prop_fail)

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.85),
                             constrained_layout=True)
    ylabels = [PUBLIC_LABELS[name].split("~")[0] for name in method_names]
    colors = ["#4E79A7", "#D55E00", "#009E73"]
    markers = ["o", "s", "D"]
    offsets = [-0.20, 0.0, 0.20]
    y = np.arange(5, dtype=float)
    for column, profile in enumerate(profile_names):
        positions = y + offsets[column]
        for axis, mean, low, high in (
            (axes[0, 0], rmse_delta[:, column], rmse_low[:, column],
             rmse_high[:, column]),
            (axes[0, 1], cost_delta[:, column], cost_low[:, column],
             cost_high[:, column])):
            finite = np.isfinite(mean)
            axis.errorbar(mean[finite], positions[finite],
                          xerr=np.vstack((mean[finite] - low[finite],
                                          high[finite] - mean[finite])),
                          fmt=markers[column], color=colors[column],
                          markerfacecolor="white", markeredgewidth=1.0,
                          markersize=4.5, elinewidth=0.9, capsize=2.2,
                          label=profile)
        axes[1, 0].plot(joint_win[:, column], positions,
                        linestyle="none", marker=markers[column],
                        color=colors[column], markerfacecolor="white",
                        markeredgewidth=1.0, markersize=4.5, label=profile)
        axes[1, 1].plot(failure_advantage[:, column], positions,
                        linestyle="none", marker=markers[column],
                        color=colors[column], markerfacecolor="white",
                        markeredgewidth=1.0, markersize=4.5, label=profile)
        axes[1, 1].plot(proposed_only_failure[:, column], positions,
                        linestyle="none", marker="x", color=colors[column],
                        markeredgewidth=0.9, markersize=4.2)
    # DL--RBF diverges for the pulse profile, so no finite effect estimate
    # exists.  Place the categorical outcome at the row margin rather than at
    # an invented numerical coordinate.
    for axis in axes[0]:
        axis.text(0.02, 1 + offsets[0], r"\textsc{dvg}",
                  transform=axis.get_yaxis_transform(), color=colors[0],
                  fontsize=6.5, ha="left", va="center")
    for axis in axes[0]:
        axis.axvline(0, color="0.25", linewidth=0.8, linestyle="--")
    axes[0, 0].set_xlabel(r"$\Delta$RMSE (LDCNC $-$ comparator)")
    axes[0, 1].set_xlabel(r"$\Delta J$ (LDCNC $-$ comparator)")
    axes[1, 0].set_xlabel(r"Joint RMSE--cost wins (\%)")
    axes[1, 0].set_xlim(45, 103)
    axes[1, 1].set_xlabel("Discordant failures (open: comparator; $\times$: LDCNC)")
    axes[1, 1].set_xlim(-0.5, 12.0)
    axes[1, 0].legend(loc="lower right", ncol=1, frameon=True,
                      framealpha=0.92, borderpad=0.25)
    for index, axis in enumerate(axes.flat):
        axis.set_yticks(y, ylabels)
        axis.invert_yaxis()
        axis.grid(axis="x", color="0.88", linewidth=0.5)
        axis.text(0.015, 0.97, rf"\textbf{{({chr(ord('a') + index)})}}",
                  transform=axis.transAxes, va="top")
    fig.savefig(FIGURES / "paired_rmse_differences.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "paired_rmse_differences.png", dpi=400,
                bbox_inches="tight")
    plt.close(fig)


def plot_severity() -> None:
    stresses = np.asarray([stress for _, stress in SEVERITY_PLOT_GRID])
    summaries = severity_summaries(SEVERITY_PLOT_GRID)
    labels = {
        "point": "History point", "sftsmc_eso": "SFTSMC--ESO",
        "dl_rbf": "DL--RBF", "pet_aftc": "PET--AFTC",
        "fasmo_cftc": "FASMO--CFTC", "le_ftc": "LE--FTC",
        "proposed": "LDCNC"}
    colors = {
        "point": "#777777", "sftsmc_eso": "#4E79A7",
        "dl_rbf": "#F28E2B", "pet_aftc": "#59A14F",
        "fasmo_cftc": "#E15759", "le_ftc": "#B07AA1",
        "proposed": "#0072B2"}
    markers = ["o", "^", "v", "D", "P", "X", "*"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), sharex=True,
                             constrained_layout=True)

    def wilson(count: int, total: int) -> tuple[float, float]:
        z = 1.96
        p = count / total
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
        return 100 * (center - half), 100 * (center + half)

    for method_index, name in enumerate(summaries):
        external = name not in {"point", "proposed"}
        linewidth = 0.85 if external else 1.8
        alpha = 0.42 if external else 1.0
        zorder = 2 if external else 5
        linestyle = "--" if name == "point" else "-"
        rmse = [item["rmse"] if item.get("divergence", 0.0) == 0 else np.nan
                for item in summaries[name]]
        axes[0].plot(stresses, rmse, color=colors[name],
                     marker=markers[method_index], markersize=4.2,
                     linewidth=linewidth, linestyle=linestyle, alpha=alpha,
                     label=labels[name], zorder=zorder)
        rates = np.asarray([100 * item["failure"] for item in summaries[name]])
        bounds = np.asarray([
            wilson(int(round(item["failure"] * 64)), 64)
            for item in summaries[name]])
        axes[1].errorbar(stresses, rates,
                         yerr=np.vstack((rates - bounds[:, 0], bounds[:, 1] - rates)),
                         color=colors[name], marker=markers[method_index],
                         markersize=4.2, linewidth=linewidth,
                         linestyle=linestyle, capsize=1.8, alpha=alpha,
                         label=labels[name], zorder=zorder)
        for stress, item in zip(stresses, summaries[name]):
            if item.get("divergence", 0.0) > 0:
                axes[0].plot(stress, 0.94, marker="x", color=colors[name],
                             markersize=5.5, markeredgewidth=1.1,
                             transform=axes[0].get_xaxis_transform(), zorder=7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Tracking RMSE")
    axes[1].set_ylabel(r"Failure rate (\%)")
    axes[1].set_ylim(-4, 64)
    for index, axis in enumerate(axes):
        axis.set_xlabel("Fault stress level")
        axis.grid(color="0.90", linewidth=0.45)
        axis.text(0.015, 0.97, rf"\textbf{{({chr(ord('a') + index)})}}",
                  transform=axis.transAxes, va="top")
    axes[0].text(0.985, 0.965, r"$\times$: DVG",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=6.5)
    axes[1].legend(loc="upper left", ncol=2, frameon=True,
                   framealpha=0.90, borderpad=0.25,
                   columnspacing=0.7, handlelength=1.4, fontsize=6.2)
    fig.savefig(FIGURES / "severity_boundary.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "severity_boundary.png", dpi=400,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_main_table()
    write_ablation_table()
    write_generalization_table()
    write_severity_table()
    write_public_comparison()
    write_profile_table()
    configure_plotting()
    plot_paired_differences()
    plot_severity()
    print(f"paper assets written to {PAPER}")


if __name__ == "__main__":
    main()
