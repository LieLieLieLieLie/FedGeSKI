"""Refresh every FedGeSKI result after a method-level implementation change."""
from config import ExperimentConfig
from federated import run_experiment
from run_all import ablation_configs, sensitivity_configs, stress_configs


def main() -> None:
    configs = [
        ExperimentConfig(dataset=dataset, method="fedgeski", seed=seed)
        for dataset in ("edgeiiot", "ciciot") for seed in (0, 1, 2)
    ]
    configs += [c for c in stress_configs() if c.method == "fedgeski"]
    configs += ablation_configs()
    configs += sensitivity_configs()
    seen = set()
    unique = []
    for config in configs:
        if config.run_id not in seen:
            seen.add(config.run_id); unique.append(config)
    for index, config in enumerate(unique, 1):
        print(f"[{index}/{len(unique)}] {config.run_id}", flush=True)
        run_experiment(config, overwrite=True, verbose=False)


if __name__ == "__main__":
    main()
