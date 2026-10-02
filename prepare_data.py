from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config import DATASETS, DATA_DIR


def _read_numeric_csv(path: Path, label_column: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    dtypes: Dict[str, object] = {
        column: (np.int64 if column == label_column else np.float32)
        for column in header.columns
    }
    frame = pd.read_csv(path, dtype=dtypes)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return frame


def _group_disjoint_sample(
    frame: pd.DataFrame,
    label_column: str,
    per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Create deterministic exact-feature-group-disjoint 70/15/15 subsets.

    The available public derivatives do not retain time, session, device, or
    flow identifiers.  We therefore use the strongest split that their fields
    support: every row sharing an identical feature vector is assigned to one
    subset before class-wise sampling.  This prevents exact duplicate feature
    vectors from crossing train/validation/test boundaries without inventing
    unavailable provenance metadata.
    """
    feature_names = [column for column in frame.columns if column != label_column]
    hashes = pd.util.hash_pandas_object(frame[feature_names], index=False).to_numpy(
        dtype=np.uint64, copy=False
    )
    # Mix the stable feature hash with the split seed; all duplicate vectors
    # still receive exactly the same subset assignment.
    mixed = hashes ^ np.uint64(seed * 0x9E3779B1)
    buckets = mixed % np.uint64(100)
    split_id = np.where(buckets < 70, 0, np.where(buckets < 85, 1, 2))

    targets = (int(round(per_class * 0.70)), int(round(per_class * 0.15)))
    targets = (targets[0], targets[1], per_class - targets[0] - targets[1])
    split_frames: list[list[pd.DataFrame]] = [[], [], []]
    shortfalls: dict[str, dict[str, int]] = {}
    for class_value, class_frame in frame.groupby(label_column, sort=True):
        class_positions = class_frame.index.to_numpy()
        for sid, target in enumerate(targets):
            candidates = class_frame.loc[split_id[class_positions] == sid]
            take = min(target, len(candidates))
            sampled = candidates.sample(n=take, random_state=seed + sid)
            split_frames[sid].append(sampled)
            if take < target:
                shortfalls.setdefault(str(int(class_value)), {})[("train", "validation", "test")[sid]] = target - take

    if shortfalls:
        raise RuntimeError(
            "Group-disjoint allocation cannot satisfy the requested per-class "
            f"sample counts: {shortfalls}. Reduce --per-class."
        )

    outputs = tuple(
        pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed + sid).reset_index(drop=True)
        for sid, parts in enumerate(split_frames)
    )
    split_hashes = [
        set(pd.util.hash_pandas_object(part[feature_names], index=False).astype("uint64").tolist())
        for part in outputs
    ]
    overlap = {
        "train_validation": len(split_hashes[0] & split_hashes[1]),
        "train_test": len(split_hashes[0] & split_hashes[2]),
        "validation_test": len(split_hashes[1] & split_hashes[2]),
    }
    return outputs[0], outputs[1], outputs[2], overlap


def prepare_dataset(
    dataset: str,
    per_class: int = 6000,
    seed: int = 2026,
    overwrite: bool = False,
) -> Path:
    spec = DATASETS[dataset]
    csv_path = Path(spec["csv"])
    label_column = str(spec["label"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing source dataset: {csv_path}")

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / f"{dataset}_pc{per_class}_seed{seed}.npz"
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and metadata_path.exists() and not overwrite:
        print(f"[cached] {output_path}")
        return output_path

    frame = _read_numeric_csv(csv_path, label_column)
    train_frame, val_frame, test_frame, overlap = _group_disjoint_sample(
        frame, label_column, per_class, seed
    )
    feature_names = [column for column in frame.columns if column != label_column]
    classes = np.unique(frame[label_column].to_numpy())
    mapping = {int(value): index for index, value in enumerate(classes.tolist())}

    def arrays(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x_part = part[feature_names].to_numpy(dtype=np.float32, copy=True)
        y_part = np.asarray(
            [mapping[int(value)] for value in part[label_column].to_numpy()], dtype=np.int64
        )
        return x_part, y_part

    x_train, y_train = arrays(train_frame)
    x_val, y_val = arrays(val_frame)
    x_test, y_test = arrays(test_frame)

    scaler = RobustScaler(quantile_range=(5.0, 95.0), unit_variance=True)
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    x_train = np.clip(x_train, -12.0, 12.0)
    x_val = np.clip(x_val, -12.0, 12.0)
    x_test = np.clip(x_test, -12.0, 12.0)

    np.savez_compressed(
        output_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        center=np.asarray(scaler.center_, dtype=np.float32),
        scale=np.asarray(scaler.scale_, dtype=np.float32),
        classes=classes,
        feature_names=np.asarray(feature_names),
    )

    metadata = {
        "dataset": dataset,
        "source_name": spec["name"],
        "source_csv": str(csv_path.resolve()),
        "label_column": label_column,
        "feature_names": feature_names,
        "class_mapping": {str(key): value for key, value in mapping.items()},
        "samples_per_class_requested": per_class,
        "split_sizes": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "seed": seed,
        "split_strategy": (
            "Exact-feature-group-disjoint 70/15/15 allocation before class-wise "
            "sampling; identical feature vectors cannot cross subsets"
        ),
        "exact_feature_hash_overlap": overlap,
        "provenance_limitation": (
            "The public numerical derivative lacks timestamp, device, session, and "
            "flow identifiers; this split does not claim temporal/device/session disjointness"
        ),
        "preprocessing": "RobustScaler(5th,95th), training split only; clipped to [-12,12]",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[prepared] {output_path} | train={len(y_train)} val={len(y_val)} test={len(y_test)}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="edgeiiot")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--per-class", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    targets = tuple(DATASETS) if args.all else (args.dataset,)
    for dataset in targets:
        prepare_dataset(
            dataset,
            per_class=args.per_class,
            seed=args.seed,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
