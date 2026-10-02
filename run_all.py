from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Iterable, List

from config import DATASETS, METHODS, ExperimentConfig
from federated import run_experiment


def main_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(dataset=dataset, method=method, seed=seed)
        for dataset in DATASETS
        for method in METHODS
        for seed in range(5)
    ]


def stress_configs() -> List[ExperimentConfig]:
    configs: List[ExperimentConfig] = []
    for dataset in DATASETS:
        for method in METHODS:
            for lag in (0, 1, 3, 4):
                configs.append(
                    ExperimentConfig(dataset=dataset, method=method, seed=0, max_staleness=lag)
                )
            for alpha in (0.1, 0.6, 1.0):
                configs.append(
                    ExperimentConfig(dataset=dataset, method=method, seed=0, dirichlet_alpha=alpha)
                )
            for participation in (0.3, 0.8, 1.0):
                configs.append(
                    ExperimentConfig(
                        dataset=dataset,
                        method=method,
                        seed=0,
                        participation=participation,
                    )
                )
    return configs


def ablation_configs() -> List[ExperimentConfig]:
    variants = (
        "no_stats", "no_simplex", "no_subspace", "uniform_fusion", "with_calibration",
        "stats_only", "stats_fusion", "simplex_stats", "simplex_only", "no_server_consolidation",
    )
    configs: List[ExperimentConfig] = []
    for dataset in DATASETS:
        for seed in range(5):
            for variant in variants:
                kwargs = {"calibration_weight": 0.15} if variant == "with_calibration" else {}
                configs.append(
                    ExperimentConfig(
                        dataset=dataset,
                        method="fedgeski",
                        seed=seed,
                        variant=variant,
                        **kwargs,
                    )
                )
    return configs


def sensitivity_configs() -> List[ExperimentConfig]:
    configs: List[ExperimentConfig] = []
    for dataset in DATASETS:
        for weight in (0.3, 0.6, 1.2, 1.5):
            configs.append(
                ExperimentConfig(
                    dataset=dataset,
                    method="fedgeski",
                    seed=0,
                    variant=f"sens_anchor_{weight:g}",
                    anchor_weight=weight,
                )
            )
        for weight in (0.0, 0.1, 0.4, 0.6):
            configs.append(
                ExperimentConfig(
                    dataset=dataset,
                    method="fedgeski",
                    seed=0,
                    variant=f"sens_stability_{weight:g}",
                    stability_weight=weight,
                )
            )
        for length in (2, 4, 6, 8):
            configs.append(
                ExperimentConfig(
                    dataset=dataset,
                    method="fedgeski",
                    seed=0,
                    variant=f"sens_history_{length}",
                    history_length=length,
                )
            )
    return configs


def asynchronous_configs() -> List[ExperimentConfig]:
    """Matched multi-seed delay comparison for dedicated asynchronous methods."""
    return [
        ExperimentConfig(dataset=dataset, method=method, seed=seed, max_staleness=lag)
        for dataset in DATASETS
        for method in ("fedasync", "fedbuff", "afcl_csc", "fedgeski")
        for lag in (0, 2, 4)
        for seed in range(5)
    ]


def extreme_configs() -> List[ExperimentConfig]:
    """Five-seed checks for the three prespecified high-risk conditions."""
    configs: List[ExperimentConfig] = []
    for dataset in DATASETS:
        for method in ("fedgcc", "afcl_csc", "fedgeski"):
            for seed in range(5):
                configs.extend(
                    [
                        ExperimentConfig(dataset=dataset, method=method, seed=seed, max_staleness=4),
                        ExperimentConfig(dataset=dataset, method=method, seed=seed, dirichlet_alpha=0.1),
                        ExperimentConfig(dataset=dataset, method=method, seed=seed, participation=0.3),
                    ]
                )
    return configs


def stress_ablation_configs() -> List[ExperimentConfig]:
    """Five-seed targeted module controls under delay and sparse participation."""
    variants = ("full", "no_subspace", "uniform_fusion", "freshness_only", "reliability_only")
    configs: List[ExperimentConfig] = []
    for dataset in DATASETS:
        for seed in range(5):
            for variant in variants:
                configs.append(
                    ExperimentConfig(
                        dataset=dataset,
                        method="fedgeski",
                        seed=seed,
                        variant=variant,
                        max_staleness=4,
                    )
                )
                configs.append(
                    ExperimentConfig(
                        dataset=dataset,
                        method="fedgeski",
                        seed=seed,
                        variant=variant,
                        participation=0.3,
                    )
                )
    return configs


def diagnostic_configs() -> List[ExperimentConfig]:
    """Five-seed model-age grid for the validation-only subspace audit."""
    return [
        ExperimentConfig(dataset=dataset, method="fedgeski", seed=seed, max_staleness=lag)
        for dataset in DATASETS
        for lag in (0, 1, 2, 3, 4)
        for seed in range(5)
    ]


def mechanism_configs() -> List[ExperimentConfig]:
    """Five-seed Full versus statistics-only mechanism control."""
    configs: List[ExperimentConfig] = []
    for dataset in DATASETS:
        for seed in range(5):
            configs.append(ExperimentConfig(dataset=dataset, method="fedgeski", seed=seed))
            configs.append(
                ExperimentConfig(
                    dataset=dataset,
                    method="fedgeski",
                    seed=seed,
                    variant="stats_only",
                )
            )
    return configs


def execute(configs: Iterable[ExperimentConfig], overwrite: bool) -> None:
    configs = list(configs)
    for index, config in enumerate(configs, start=1):
        print(f"\n=== Run {index}/{len(configs)}: {config.run_id} ===")
        run_experiment(config, overwrite=overwrite, verbose=False)
        print(f"[complete] {config.run_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("main", "stress", "ablation", "sensitivity", "asynchronous", "mechanism", "extreme", "stress_ablation", "diagnostic", "all"),
        default="main",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    suites = {
        "main": main_configs,
        "stress": stress_configs,
        "ablation": ablation_configs,
        "sensitivity": sensitivity_configs,
        "asynchronous": asynchronous_configs,
        "mechanism": mechanism_configs,
        "extreme": extreme_configs,
        "stress_ablation": stress_ablation_configs,
        "diagnostic": diagnostic_configs,
    }
    selected = tuple(suites) if args.suite == "all" else (args.suite,)
    for suite in selected:
        print(f"\n##### SUITE: {suite} #####")
        execute(suites[suite](), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
