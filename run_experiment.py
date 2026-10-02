from __future__ import annotations

import argparse
from dataclasses import replace

from config import DATASETS, METHODS, ExperimentConfig
from federated import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="edgeiiot")
    parser.add_argument("--method", choices=METHODS, default="fedgeski")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--max-staleness", type=int, default=2)
    parser.add_argument("--participation", type=float, default=0.6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig(
        dataset=args.dataset,
        method=args.method,
        seed=args.seed,
        rounds=args.rounds,
        local_steps=args.local_steps,
        dirichlet_alpha=args.alpha,
        max_staleness=args.max_staleness,
        participation=args.participation,
    )
    run_experiment(config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
