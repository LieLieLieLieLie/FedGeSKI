from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import METHOD_LABELS, METHODS, MODELS_DIR, TABLES_DIR


DATASET_LABELS = {"edgeiiot": "Edge-IIoTset", "ciciot": "CICIoT2023"}
METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "macro_auroc",
    "ece",
    "average_forgetting",
    "backward_transfer",
    "communication_mb",
    "uplink_communication_mb",
    "downlink_communication_mb",
    "peak_auxiliary_memory_mb",
    "client_auxiliary_memory_mb",
    "server_auxiliary_memory_mb",
    "total_resident_auxiliary_memory_mb",
    "wall_time_seconds",
    "model_parameters",
)


def load_runs() -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for path in sorted(MODELS_DIR.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "final" in item and "config" in item:
            item["_path"] = str(path)
            runs.append(item)
    return runs


def is_main(run: Dict[str, object]) -> bool:
    cfg = run["config"]
    return (
        cfg["variant"] == "full"
        and cfg["dirichlet_alpha"] == 0.3
        and cfg["max_staleness"] == 2
        and cfg["participation"] == 0.6
        and cfg["seed"] in range(5)
    )


def rows_from(runs: Iterable[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        cfg, final = run["config"], run["final"]
        if cfg.get("method") not in METHOD_LABELS:
            continue
        row = {
            "dataset": cfg["dataset"],
            "dataset_label": DATASET_LABELS[cfg["dataset"]],
            "method": cfg["method"],
            "method_label": METHOD_LABELS[cfg["method"]],
            "seed": cfg["seed"],
            "variant": cfg.get("variant", "full"),
            "alpha": cfg["dirichlet_alpha"],
            "staleness": cfg["max_staleness"],
            "participation": cfg["participation"],
        }
        for metric in METRICS:
            row[metric] = final.get(metric, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def mean_std(value: pd.Series, digits: int = 3) -> str:
    return f"{value.mean():.{digits}f} ± {value.std(ddof=1):.{digits}f}"


def export_main(frame: pd.DataFrame) -> None:
    main = frame[
        (frame.variant == "full")
        & (frame.method.isin(METHODS))
        & (frame.alpha == 0.3)
        & (frame.staleness == 2)
        & (frame.participation == 0.6)
        & (frame.seed.isin(range(5)))
    ].copy()
    main.to_csv(TABLES_DIR / "main_runs.csv", index=False)
    grouped = main.groupby(["dataset", "dataset_label", "method", "method_label"], sort=False)
    records = []
    for keys, group in grouped:
        record = dict(zip(("dataset", "dataset_label", "method", "method_label"), keys))
        for metric in METRICS:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_std"] = group[metric].std(ddof=1)
            record[metric] = mean_std(group[metric])
        records.append(record)
    summary = pd.DataFrame(records)
    order = {method: idx for idx, method in enumerate(METHODS)}
    summary["_order"] = summary.method.map(order)
    summary = summary.sort_values(["dataset", "_order"]).drop(columns="_order")
    summary.to_csv(TABLES_DIR / "main_results.csv", index=False)

    # Compact manuscript-ready table. Lower is better for ECE/forgetting/resources.
    compact = summary[[
        "dataset_label", "method_label", "macro_f1", "balanced_accuracy",
        "macro_auroc", "ece", "average_forgetting", "communication_mb",
        "peak_auxiliary_memory_mb",
    ]].copy()
    compact.columns = [
        "Dataset", "Method", "Macro-F1", "Balanced accuracy", "AUROC",
        "ECE", "Forgetting", "Comm. (MiB)", "Aux. memory (MiB)",
    ]
    compact.to_csv(TABLES_DIR / "main_results_manuscript.csv", index=False)


def export_ablation(frame: pd.DataFrame) -> None:
    variants = [
        "full", "stats_only", "stats_fusion", "simplex_only", "simplex_stats",
        "no_server_consolidation", "no_stats", "no_simplex", "no_subspace",
        "uniform_fusion", "with_calibration",
    ]
    labels = {
        "full": "FedGeSKI",
        "no_stats": "w/o statistical consolidation",
        "no_simplex": "w/o simplex anchoring",
        "no_subspace": "w/o stable-plastic projection",
        "uniform_fusion": "w/o reliability-staleness fusion",
        "with_calibration": "with optional posterior correction",
        "stats_only": "moment consolidation + asynchronous averaging",
        "stats_fusion": "statistics + reliability-staleness fusion",
        "simplex_only": "simplex only",
        "simplex_stats": "simplex + statistics",
        "no_server_consolidation": "w/o server consolidation",
    }
    ab = frame[
        (frame.method == "fedgeski")
        & (frame.variant.isin(variants))
        & (frame.alpha == 0.3)
        & (frame.staleness == 2)
        & (frame.participation == 0.6)
        & (frame.seed.isin(range(5)))
    ].copy()
    ab["variant_label"] = ab.variant.map(labels)
    ab.to_csv(TABLES_DIR / "ablation_runs.csv", index=False)
    out = (
        ab.groupby(["dataset_label", "variant", "variant_label"], sort=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
            ece_mean=("ece", "mean"), ece_std=("ece", "std"),
            forgetting_mean=("average_forgetting", "mean"),
            forgetting_std=("average_forgetting", "std"),
            bwt_mean=("backward_transfer", "mean"), bwt_std=("backward_transfer", "std"),
        )
        .reset_index()
    )
    out["_order"] = out.variant.map({v: i for i, v in enumerate(variants)})
    out = out.sort_values(["dataset_label", "_order"]).drop(columns="_order")
    out.to_csv(TABLES_DIR / "ablation_results.csv", index=False)


def export_stress(frame: pd.DataFrame) -> None:
    stress = frame[
        (frame.seed == 0)
        & (frame.method.isin(METHODS))
        & (frame.variant == "full")
        & (
            ((frame.alpha.isin([0.1, 0.3, 0.6, 1.0])) & (frame.staleness == 2) & (frame.participation == 0.6))
            | ((frame.alpha == 0.3) & (frame.staleness.isin([0, 1, 2, 3, 4])) & (frame.participation == 0.6))
            | ((frame.alpha == 0.3) & (frame.staleness == 2) & (frame.participation.isin([0.3, 0.6, 0.8, 1.0])))
        )
    ].drop_duplicates(subset=["dataset", "method", "seed", "variant", "alpha", "staleness", "participation"])
    stress.to_csv(TABLES_DIR / "stress_results.csv", index=False)


def export_sensitivity(frame: pd.DataFrame) -> None:
    sens = frame[(frame.method == "fedgeski") & frame.variant.str.startswith("sens_")].copy()
    sens.to_csv(TABLES_DIR / "sensitivity_results.csv", index=False)


def export_asynchronous(frame: pd.DataFrame) -> None:
    async_frame = frame[
        (frame.method.isin(["fedasync", "fedbuff", "afcl_csc", "fedgeski"]))
        & (frame.variant == "full")
        & (frame.alpha == 0.3)
        & (frame.staleness.isin([0, 2, 4]))
        & (frame.participation == 0.6)
        & (frame.seed.isin(range(5)))
    ].copy()
    grouped = (
        async_frame.groupby(
            ["dataset_label", "method", "method_label", "staleness"], sort=False
        )
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
            forgetting_mean=("average_forgetting", "mean"),
            forgetting_std=("average_forgetting", "std"),
            communication_mb_mean=("communication_mb", "mean"),
            uplink_communication_mb_mean=("uplink_communication_mb", "mean"),
            downlink_communication_mb_mean=("downlink_communication_mb", "mean"),
            wall_time_seconds_mean=("wall_time_seconds", "mean"),
        )
        .reset_index()
    )
    grouped.to_csv(TABLES_DIR / "asynchronous_comparison.csv", index=False)


def export_mechanism_five_seed(frame: pd.DataFrame) -> None:
    mechanism = frame[
        (frame.method == "fedgeski")
        & (frame.variant.isin(["full", "stats_only"]))
        & (frame.alpha == 0.3)
        & (frame.staleness == 2)
        & (frame.participation == 0.6)
        & (frame.seed.isin([0, 1, 2, 3, 4]))
    ].copy()
    grouped = (
        mechanism.groupby(["dataset_label", "variant"], sort=False)
        .agg(
            n=("seed", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
            forgetting_mean=("average_forgetting", "mean"),
            forgetting_std=("average_forgetting", "std"),
        )
        .reset_index()
    )
    grouped.to_csv(TABLES_DIR / "mechanism_five_seed.csv", index=False)


def export_paired_effects(frame: pd.DataFrame) -> None:
    main = frame[
        (frame.variant == "full") & (frame.method.isin(METHODS))
        & (frame.alpha == 0.3) & (frame.staleness == 2)
        & (frame.participation == 0.6) & (frame.seed.isin(range(5)))
    ]
    rows = []
    for dataset in DATASET_LABELS:
        proposal = main[(main.dataset == dataset) & (main.method == "fedgeski")].sort_values("seed")
        for baseline in METHODS[:-1]:
            other = main[(main.dataset == dataset) & (main.method == baseline)].sort_values("seed")
            merged = proposal[["seed", "macro_f1"]].merge(
                other[["seed", "macro_f1"]], on="seed", suffixes=("_ours", "_base")
            )
            diff = merged.macro_f1_ours - merged.macro_f1_base
            rng = np.random.default_rng(20261001)
            bootstrap = np.asarray([
                rng.choice(diff.to_numpy(), size=len(diff), replace=True).mean()
                for _ in range(20000)
            ])
            if np.allclose(diff, 0):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(diff, alternative="two-sided", method="exact")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            rows.append({
                "dataset": DATASET_LABELS[dataset],
                "comparison": f"FedGeSKI vs. {METHOD_LABELS[baseline]}",
                "n_pairs": len(diff),
                "mean_f1_gain": diff.mean(),
                "std_f1_gain": diff.std(ddof=1),
                "cohen_dz": diff.mean() / (diff.std(ddof=1) + 1e-12),
                "bootstrap_ci95_low": np.quantile(bootstrap, 0.025),
                "bootstrap_ci95_high": np.quantile(bootstrap, 0.975),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_two_sided": p_value,
                "all_differences_positive": bool((diff > 0).all()),
            })
    pd.DataFrame(rows).to_csv(TABLES_DIR / "paired_effects.csv", index=False)


def export_extreme(frame: pd.DataFrame) -> None:
    extreme = frame[
        (frame.method.isin(["fedgcc", "afcl_csc", "fedgeski"]))
        & (frame.variant == "full")
        & (frame.seed.isin(range(5)))
        & (
            ((frame.alpha == 0.1) & (frame.staleness == 2) & (frame.participation == 0.6))
            | ((frame.alpha == 0.3) & (frame.staleness == 4) & (frame.participation == 0.6))
            | ((frame.alpha == 0.3) & (frame.staleness == 2) & (frame.participation == 0.3))
        )
    ].copy()
    extreme["condition"] = np.select(
        [extreme.alpha == 0.1, extreme.staleness == 4, extreme.participation == 0.3],
        ["Dirichlet alpha=0.1", "Maximum model age=4", "Participation=0.3"],
    )
    grouped = (
        extreme.groupby(["dataset_label", "condition", "method", "method_label"], sort=False)
        .agg(
            n=("seed", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
            forgetting_mean=("average_forgetting", "mean"),
            forgetting_std=("average_forgetting", "std"),
        )
        .reset_index()
    )
    extreme.to_csv(TABLES_DIR / "extreme_runs.csv", index=False)
    grouped.to_csv(TABLES_DIR / "extreme_five_seed.csv", index=False)


def export_stress_ablation(frame: pd.DataFrame) -> None:
    variants = ("full", "no_subspace", "uniform_fusion", "freshness_only", "reliability_only")
    selected = frame[
        (frame.method == "fedgeski")
        & (frame.variant.isin(variants))
        & (frame.seed.isin(range(5)))
        & (frame.alpha == 0.3)
        & (
            ((frame.staleness == 4) & (frame.participation == 0.6))
            | ((frame.staleness == 2) & (frame.participation == 0.3))
        )
    ].copy()
    selected["condition"] = np.where(
        selected.staleness == 4, "Maximum model age=4", "Participation=0.3"
    )
    selected.to_csv(TABLES_DIR / "stress_ablation_runs.csv", index=False)
    grouped = (
        selected.groupby(["dataset_label", "condition", "variant"], sort=False)
        .agg(
            n=("seed", "count"),
            macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
            ece_mean=("ece", "mean"), ece_std=("ece", "std"),
            forgetting_mean=("average_forgetting", "mean"),
            forgetting_std=("average_forgetting", "std"),
        )
        .reset_index()
    )
    grouped.to_csv(TABLES_DIR / "stress_ablation_results.csv", index=False)

    rows = []
    rng = np.random.default_rng(20261002)
    for (dataset, condition), group in selected.groupby(["dataset_label", "condition"]):
        full = group[group.variant == "full"].sort_values("seed")
        for variant in variants[1:]:
            other = group[group.variant == variant].sort_values("seed")
            merged = full[["seed", "macro_f1", "ece", "average_forgetting"]].merge(
                other[["seed", "macro_f1", "ece", "average_forgetting"]],
                on="seed", suffixes=("_full", "_variant")
            )
            for metric, favorable_sign in (("macro_f1", 1.0), ("ece", -1.0), ("average_forgetting", -1.0)):
                raw = merged[f"{metric}_full"] - merged[f"{metric}_variant"]
                favorable = favorable_sign * raw.to_numpy()
                bootstrap = np.asarray([
                    rng.choice(favorable, size=len(favorable), replace=True).mean()
                    for _ in range(20000)
                ])
                rows.append({
                    "dataset": dataset,
                    "condition": condition,
                    "comparison": f"Full vs. {variant}",
                    "metric": metric,
                    "n_pairs": len(favorable),
                    "favorable_delta_mean": favorable.mean(),
                    "bootstrap_ci95_low": np.quantile(bootstrap, 0.025),
                    "bootstrap_ci95_high": np.quantile(bootstrap, 0.975),
                })
    pd.DataFrame(rows).to_csv(TABLES_DIR / "stress_ablation_paired.csv", index=False)


def main() -> None:
    frame = rows_from(load_runs())
    if frame.empty:
        raise RuntimeError("No completed result JSON files were found.")
    export_main(frame)
    export_ablation(frame)
    export_stress(frame)
    export_sensitivity(frame)
    export_asynchronous(frame)
    export_mechanism_five_seed(frame)
    export_paired_effects(frame)
    export_extreme(frame)
    export_stress_ablation(frame)
    print(f"Exported result tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
