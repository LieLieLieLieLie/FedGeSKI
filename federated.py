from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import LOGS_DIR, MODELS_DIR, ExperimentConfig
from data_stream import FederatedContinualStream
from metrics import classification_metrics
from methods import LocalResult, make_client_states, train_client
from models import (
    SimplexStatistics,
    clone_model,
    flatten_delta,
    flatten_gradients,
    make_model,
    trainable_numel,
    unflatten_like,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def _predict(
    model: nn.Module,
    x: torch.Tensor,
    server_stats: SimplexStatistics,
    method: str,
    calibration_weight: float,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    probabilities: List[torch.Tensor] = []
    for start in range(0, len(x), batch_size):
        logits, features = model(x[start : start + batch_size], return_features=True)
        if method == "fedgeski":
            logits = server_stats.calibrated_logits(
                logits,
                features,
                anchor_scale=5.5,
                calibration_weight=calibration_weight,
            )
        elif method == "fedta" and server_stats.active.any():
            logits = server_stats.calibrated_logits(
                logits,
                features,
                anchor_scale=4.0,
                calibration_weight=0.10,
            )
        probabilities.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(probabilities, dim=0).numpy()


def _stable_basis(update_history: Sequence[torch.Tensor], rank: int = 4) -> Optional[torch.Tensor]:
    if not update_history:
        return None
    matrix = torch.stack(list(update_history), dim=1).float()
    matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp_min(1e-8)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, : min(rank, q.shape[1])]


@torch.no_grad()
def _project_fresh_updates(
    results: Sequence[LocalResult],
    basis: Optional[torch.Tensor],
    stability_weight: float,
) -> None:
    """Apply stable--plastic projection after server-side ID screening."""
    if basis is None or basis.numel() == 0:
        return
    for result in results:
        flat = flatten_delta(result.delta)
        q = basis.to(flat.device)
        projected = flat - stability_weight * (q @ (q.T @ flat))
        result.delta = unflatten_like(projected, result.delta)


def _historical_gradient_coverage(
    model: nn.Module,
    x_diagnostic: torch.Tensor,
    y_diagnostic: torch.Tensor,
    first_learned_round: np.ndarray,
    round_id: int,
    basis: Optional[torch.Tensor],
    max_examples: int = 512,
) -> Tuple[float, float]:
    """Post-hoc audit of the old-class validation gradient captured by Q_t.

    This diagnostic is recorded only after the model update.  It is never used
    by local training, aggregation, model selection, or hyperparameter tuning.
    """
    if basis is None or basis.numel() == 0:
        return float("nan"), float("nan")
    old_classes = np.flatnonzero((first_learned_round >= 0) & (first_learned_round < round_id))
    if old_classes.size == 0:
        return float("nan"), float("nan")
    class_tensor = torch.as_tensor(
        old_classes, device=y_diagnostic.device, dtype=y_diagnostic.dtype
    )
    mask = (y_diagnostic.unsqueeze(1) == class_tensor.unsqueeze(0)).any(dim=1)
    indices = torch.nonzero(mask, as_tuple=False).flatten()[:max_examples]
    if indices.numel() == 0:
        return float("nan"), float("nan")
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    logits = model(x_diagnostic[indices])
    F.cross_entropy(logits, y_diagnostic[indices]).backward()
    gradient = flatten_gradients(model).detach()
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    q = basis.to(gradient.device)
    protected = q @ (q.T @ gradient)
    denominator = gradient.norm().clamp_min(1e-12)
    coverage = float((protected.norm() / denominator).item())
    residual = float(((gradient - protected).norm() / denominator).item())
    return coverage, residual


def _server_consolidate(
    model: nn.Module,
    server_stats: SimplexStatistics,
    config: ExperimentConfig,
    round_id: int,
) -> None:
    if config.server_consolidation_steps <= 0 or config.variant in {"no_stats", "simplex_only", "no_server_consolidation"}:
        return
    generator = torch.Generator(device=server_stats.anchors.device)
    generator.manual_seed(config.seed * 10000 + round_id + 811)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.server_learning_rate, weight_decay=config.weight_decay
    )
    model.train()
    for _ in range(config.server_consolidation_steps):
        synthetic = server_stats.sample_inputs(config.synthetic_per_class, generator)
        if synthetic is None:
            return
        x_synthetic, y_synthetic = synthetic
        optimizer.zero_grad(set_to_none=True)
        logits, features = model(x_synthetic, return_features=True)
        normalized = F.normalize(features, dim=1)
        anchor_logits = normalized @ server_stats.anchors.T / config.temperature
        if config.variant in {"stats_only", "stats_fusion"}:
            loss = F.cross_entropy(logits, y_synthetic)
        else:
            loss = F.cross_entropy(logits + 0.25 * anchor_logits, y_synthetic)
        if config.variant not in {"no_simplex", "stats_only", "stats_fusion"}:
            loss = loss + 0.35 * F.cross_entropy(anchor_logits, y_synthetic)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
        optimizer.step()


@torch.no_grad()
def _aggregate(
    model: nn.Module,
    results: Sequence[LocalResult],
    method: str,
    velocity: Optional[Dict[str, torch.Tensor]],
    momentum: float,
    optimizer_state: Optional[Dict[str, torch.Tensor]] = None,
    optimizer_lr: float = 0.10,
    beta1: float = 0.90,
    beta2: float = 0.99,
    tau: float = 1e-3,
    reliability_weighted: bool = True,
    staleness_tau: float = 2.0,
    fusion_mode: str = "full",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if not results:
        raise ValueError("At least one client result is required")
    if method == "fedgeski" and fusion_mode == "freshness_only":
        raw_weights = torch.tensor(
            [result.samples * math.exp(-result.staleness / max(staleness_tau, 1e-6)) for result in results],
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
    elif method == "fedgeski" and fusion_mode == "reliability_only":
        raw_weights = torch.tensor(
            [result.samples * result.reliability for result in results],
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
    elif method == "fedgeski" and reliability_weighted:
        raw_weights = torch.tensor(
            [
                result.samples
                * result.reliability
                * math.exp(-result.staleness / max(staleness_tau, 1e-6))
                for result in results
            ],
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
    elif method == "fedasync":
        raw_weights = torch.tensor(
            [result.samples / (1.0 + result.staleness) for result in results],
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
    elif method == "fedbuff":
        raw_weights = torch.tensor(
            [
                result.samples * math.exp(-result.staleness / max(staleness_tau, 1e-6))
                for result in results
            ],
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
    elif method == "afcl_csc":
        # Client--server cooperative weighting: reward agreement with the
        # current update consensus while exponentially discounting model age.
        flats = torch.stack([flatten_delta(result.delta).float() for result in results])
        consensus = flats.mean(dim=0)
        similarities = F.cosine_similarity(
            flats, consensus.unsqueeze(0).expand_as(flats), dim=1
        ).clamp_min(0.0)
        age_weights = torch.tensor(
            [math.exp(-result.staleness / max(staleness_tau, 1e-6)) for result in results],
            dtype=torch.float32,
            device=flats.device,
        )
        support = torch.tensor(
            [result.samples for result in results], dtype=torch.float32, device=flats.device
        )
        raw_weights = support * age_weights * (0.05 + similarities)
    else:
        raw_weights = torch.tensor(
            [result.samples for result in results],
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
    weights = raw_weights / raw_weights.sum().clamp_min(1e-8)
    state = model.state_dict()
    aggregated: Dict[str, torch.Tensor] = {}
    new_velocity: Dict[str, torch.Tensor] = {}
    new_optimizer_state: Dict[str, torch.Tensor] = {}
    new_state: Dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        update = sum(weight * result.delta[name] for weight, result in zip(weights, results))
        if not torch.is_floating_point(tensor):
            new_state[name] = tensor
            new_velocity[name] = torch.zeros_like(tensor, dtype=torch.float32)
            continue
        previous = torch.zeros_like(update) if velocity is None else velocity.get(name, torch.zeros_like(update))
        if method == "fedavgm":
            current_velocity = 0.9 * previous + 0.1 * update
            server_step = current_velocity
        elif method in {"fedadam", "fedyogi"}:
            old_m = (
                torch.zeros_like(update)
                if optimizer_state is None
                else optimizer_state.get(f"m::{name}", torch.zeros_like(update))
            )
            old_v = (
                torch.zeros_like(update)
                if optimizer_state is None
                else optimizer_state.get(f"v::{name}", torch.zeros_like(update))
            )
            current_velocity = beta1 * old_m + (1.0 - beta1) * update
            squared = update.pow(2)
            if method == "fedadam":
                second = beta2 * old_v + (1.0 - beta2) * squared
            else:
                second = old_v - (1.0 - beta2) * torch.sign(old_v - squared) * squared
            second = second.clamp_min(0.0)
            server_step = optimizer_lr * current_velocity / (second.sqrt() + tau)
            new_optimizer_state[f"m::{name}"] = current_velocity
            new_optimizer_state[f"v::{name}"] = second
        else:
            current_velocity = momentum * previous + (1.0 - momentum) * update
            server_step = current_velocity
        aggregated[name] = server_step
        new_velocity[name] = current_velocity
        new_state[name] = tensor + server_step
    model.load_state_dict(new_state)
    return aggregated, new_velocity, new_optimizer_state


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def run_experiment(config: ExperimentConfig, overwrite: bool = False, verbose: bool = True) -> Path:
    history_path = MODELS_DIR / f"{config.run_id}.json"
    checkpoint_path = MODELS_DIR / f"{config.run_id}.pt"
    predictions_path = MODELS_DIR / f"{config.run_id}.predictions.npz"
    if history_path.exists() and checkpoint_path.exists() and not overwrite:
        if verbose:
            print(f"[cached] {config.run_id}")
        return history_path

    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    stream = FederatedContinualStream(config)
    model = make_model(
        stream.num_features,
        stream.num_classes,
        config.hidden_dim,
        config.embed_dim,
        config.dropout,
        device,
    )
    server_stats = SimplexStatistics(
        stream.num_classes, config.embed_dim, device, input_dim=stream.num_features
    )
    client_states = make_client_states(config)
    model_history: List[nn.Module] = [clone_model(model).to(device).eval()]
    update_history: deque[torch.Tensor] = deque(maxlen=config.history_length)
    velocity: Optional[Dict[str, torch.Tensor]] = None
    server_optimizer_state: Optional[Dict[str, torch.Tensor]] = None
    seen_update_ids: set[Tuple[int, int]] = set()
    x_validation, y_validation = stream.tensors("val", device)
    x_test, y_test = stream.tensors("test", device)
    y_test_np = y_test.cpu().numpy()
    rng = np.random.default_rng(config.seed + 707)
    records: List[Dict[str, object]] = []
    stage_matrix: List[List[float]] = []
    cumulative_communication = 0
    cumulative_uplink_communication = 0
    cumulative_downlink_communication = 0
    peak_auxiliary_memory = 0
    client_auxiliary_memory = 0
    server_auxiliary_memory = 0
    total_resident_auxiliary_memory = 0
    first_learned_round = np.full(stream.num_classes, -1, dtype=np.int64)
    start_time = time.perf_counter()

    for round_id in range(config.rounds):
        participants = max(1, int(math.ceil(config.num_clients * config.participation)))
        selected = np.sort(rng.choice(config.num_clients, size=participants, replace=False))
        basis = _stable_basis(tuple(update_history)) if config.method == "fedgeski" else None
        results: List[LocalResult] = []
        stalenesses: List[int] = []
        for client_id_np in selected:
            client_id = int(client_id_np)
            staleness = int(rng.integers(0, min(config.max_staleness, round_id) + 1))
            stale_index = max(0, round_id - staleness)
            stale_model = model_history[stale_index]
            result = train_client(
                config.method,
                model,
                stale_model,
                stream,
                client_states[client_id],
                config,
                server_stats,
                basis,
                client_id,
                round_id,
                staleness,
            )
            results.append(result)
            stalenesses.append(staleness)
            for class_id in result.class_ids.detach().cpu().tolist():
                if first_learned_round[int(class_id)] < 0:
                    first_learned_round[int(class_id)] = round_id

        # Every class-statistic capsule describes only the minibatches sampled
        # for this client update.  The immutable (client_id, round_id) key makes
        # duplicate or replayed messages idempotent at the server.
        fresh_results = [result for result in results if result.update_id not in seen_update_ids]
        seen_update_ids.update(result.update_id for result in fresh_results)
        results = fresh_results
        if not results:
            continue

        if (
            config.method == "fedgeski"
            and config.variant not in {"no_subspace", "stats_only", "stats_fusion", "simplex_only", "simplex_stats"}
        ):
            _project_fresh_updates(results, basis, config.stability_weight)

        momentum = config.server_momentum if config.method == "fedgeski" else 0.0
        aggregate_delta, velocity, new_optimizer_state = _aggregate(
            model,
            results,
            config.method,
            velocity,
            momentum,
            optimizer_state=server_optimizer_state,
            optimizer_lr=config.server_optimizer_lr,
            beta1=config.server_beta1,
            beta2=config.server_beta2,
            tau=config.server_tau,
            reliability_weighted=config.variant not in {"uniform_fusion", "stats_only", "simplex_only", "simplex_stats"},
            staleness_tau=config.staleness_tau,
            fusion_mode=config.variant,
        )
        if config.method in {"fedadam", "fedyogi"}:
            server_optimizer_state = new_optimizer_state
        flat_update = flatten_delta(aggregate_delta).detach()
        if flat_update.norm() > 1e-10:
            update_history.append(flat_update.to(torch.float16))

        if config.method in {"fedta", "fedgeski"}:
            for result in results:
                if config.method == "fedgeski":
                    server_stats.update_input(
                        result.class_ids,
                        result.input_means,
                        result.input_variances,
                        result.input_counts,
                    )
                else:
                    server_stats.update(
                        result.class_ids,
                        result.feature_means,
                        result.feature_variances,
                        result.feature_counts,
                        result.input_means,
                        result.input_variances,
                        result.input_counts,
                    )
        if config.method == "fedgeski":
            _server_consolidate(model, server_stats, config, round_id)

        model_history.append(clone_model(model).to(device).eval())
        round_communication = sum(result.communication_bytes for result in results)
        round_uplink_communication = sum(result.uplink_communication_bytes for result in results)
        round_downlink_communication = sum(result.downlink_communication_bytes for result in results)
        cumulative_communication += round_communication
        cumulative_uplink_communication += round_uplink_communication
        cumulative_downlink_communication += round_downlink_communication
        # Persistent client memory is measured over the entire federation rather
        # than only the clients selected in the current round. FedGeSKI and
        # FedTA retain no client-side records; their local capsules are transient.
        if config.method in {"glfc", "evofedids", "fedagc"}:
            client_auxiliary_memory = sum(
                state.replay.bytes(stream.num_features) for state in client_states
            )
        elif config.method == "fedgcc":
            client_auxiliary_memory = sum(len(state.class_counts) * 8 for state in client_states)
        else:
            client_auxiliary_memory = 0

        server_auxiliary_memory = 0
        if config.method in {"fedta", "fedgeski"}:
            server_auxiliary_memory += server_stats.resident_bytes(
                include_latent=config.method == "fedta"
            )
        if config.method == "fedgeski":
            server_auxiliary_memory += sum(
                tensor.numel() * tensor.element_size() for tensor in update_history
            )
        if config.method == "fedavgm" and velocity is not None:
            server_auxiliary_memory += sum(
                tensor.numel() * tensor.element_size() for tensor in velocity.values()
            )
        if config.method in {"fedadam", "fedyogi"} and server_optimizer_state is not None:
            server_auxiliary_memory += sum(
                tensor.numel() * tensor.element_size() for tensor in server_optimizer_state.values()
            )
        total_resident_auxiliary_memory = client_auxiliary_memory + server_auxiliary_memory

        # Peak algorithmic auxiliary memory additionally includes the QR basis
        # and one decoded capsule. Generic model, autograd, and optimizer
        # workspace are excluded for every method.
        transient_bytes = 0
        if config.method in {"fedta", "fedgeski"} and results:
            transient_bytes += max(result.auxiliary_memory_bytes for result in results)
        if config.method == "fedgeski" and basis is not None:
            transient_bytes += basis.numel() * basis.element_size()
        peak_auxiliary_memory = max(
            peak_auxiliary_memory,
            total_resident_auxiliary_memory + transient_bytes,
        )

        if (round_id + 1) % config.eval_every == 0 or round_id == config.rounds - 1:
            probabilities = _predict(
                model,
                x_test,
                server_stats,
                config.method,
                0.0 if config.variant in {"no_calibration", "stats_only", "simplex_only"} else config.calibration_weight,
            )
            metrics = classification_metrics(y_test_np, probabilities)
            history_coverage, history_residual = (
                _historical_gradient_coverage(
                    model,
                    x_validation,
                    y_validation,
                    first_learned_round,
                    round_id,
                    basis,
                )
                if config.method == "fedgeski"
                else (float("nan"), float("nan"))
            )
            alignment_by_staleness = {
                str(age): float(np.mean([r.anchor_alignment for r in results if r.staleness == age]))
                for age in sorted({r.staleness for r in results})
            }
            client_scores: List[float] = []
            for partition in stream.test_partitions:
                if len(partition) == 0:
                    continue
                client_scores.append(
                    float(
                        classification_metrics(y_test_np[partition], probabilities[partition])["macro_f1"]
                    )
                )
            record: Dict[str, object] = {
                "round": round_id + 1,
                **metrics,
                "train_loss": float(np.mean([result.train_loss for result in results])),
                "mean_reliability": float(np.mean([result.reliability for result in results])),
                "mean_staleness": float(np.mean(stalenesses)),
                "history_subspace_coverage": history_coverage,
                "history_subspace_residual": history_residual,
                "anchor_alignment_by_staleness": alignment_by_staleness,
                "selected_clients": selected.tolist(),
                "client_macro_f1": client_scores,
                "communication_mb": cumulative_communication / (1024**2),
                "uplink_communication_mb": cumulative_uplink_communication / (1024**2),
                "downlink_communication_mb": cumulative_downlink_communication / (1024**2),
                "peak_auxiliary_memory_mb": peak_auxiliary_memory / (1024**2),
                "client_auxiliary_memory_mb": client_auxiliary_memory / (1024**2),
                "server_auxiliary_memory_mb": server_auxiliary_memory / (1024**2),
                "total_resident_auxiliary_memory_mb": total_resident_auxiliary_memory / (1024**2),
                "drift_events": int(sum(state.drift_events for state in client_states)),
            }
            records.append(record)
            if (round_id + 1) % config.rounds_per_stage == 0:
                stage_matrix.append([float(value) for value in metrics["per_class_recall"]])
            if verbose:
                print(
                    f"[{config.dataset}/{config.method}/s{config.seed}] "
                    f"r={round_id + 1:02d} F1={metrics['macro_f1']:.4f} "
                    f"BAcc={metrics['balanced_accuracy']:.4f} ECE={metrics['ece']:.4f}"
                )

    elapsed = time.perf_counter() - start_time
    recall_trajectory = np.asarray([record["per_class_recall"] for record in records], dtype=float)
    final_recall = recall_trajectory[-1]
    # A class is evaluated for retention only after the first global update that
    # contains evidence for that class. This prevents initial acquisition from
    # being misreported as backward transfer.
    forgetting_per_class = np.zeros(stream.num_classes, dtype=float)
    bwt_per_class = np.zeros(stream.num_classes, dtype=float)
    acquisition_recall_per_class = np.full(stream.num_classes, np.nan, dtype=float)
    for class_id in range(stream.num_classes):
        first = int(first_learned_round[class_id])
        if first < 0:
            forgetting_per_class[class_id] = np.nan
            bwt_per_class[class_id] = np.nan
            continue
        acquisition_recall_per_class[class_id] = recall_trajectory[first, class_id]
        forgetting_per_class[class_id] = (
            np.nanmax(recall_trajectory[first:, class_id]) - final_recall[class_id]
        )
        bwt_per_class[class_id] = final_recall[class_id] - recall_trajectory[first, class_id]
    final_metrics = dict(records[-1])
    final_metrics.update(
        {
            "average_forgetting": float(np.nanmean(forgetting_per_class)),
            "backward_transfer": float(np.nanmean(bwt_per_class)),
            "wall_time_seconds": float(elapsed),
            "model_parameters": trainable_numel(model),
            "communication_mb": float(cumulative_communication / (1024**2)),
            "uplink_communication_mb": float(cumulative_uplink_communication / (1024**2)),
            "downlink_communication_mb": float(cumulative_downlink_communication / (1024**2)),
            "peak_auxiliary_memory_mb": float(peak_auxiliary_memory / (1024**2)),
            "client_auxiliary_memory_mb": float(client_auxiliary_memory / (1024**2)),
            "server_auxiliary_memory_mb": float(server_auxiliary_memory / (1024**2)),
            "total_resident_auxiliary_memory_mb": float(total_resident_auxiliary_memory / (1024**2)),
        }
    )
    payload = {
        "run_id": config.run_id,
        "config": config.to_dict(),
        "dataset": config.dataset,
        "method": config.method,
        "seed": config.seed,
        "records": records,
        "stage_matrix": stage_matrix,
        "forgetting_per_class": forgetting_per_class.tolist(),
        "bwt_per_class": bwt_per_class.tolist(),
        "first_learned_round": (first_learned_round + 1).tolist(),
        "acquisition_recall_per_class": acquisition_recall_per_class.tolist(),
        "final": final_metrics,
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
    }
    history_path.write_text(
        json.dumps(payload, indent=2, default=_json_default, allow_nan=True), encoding="utf-8"
    )
    torch.save(
        {
            "config": config.to_dict(),
            "model_state": model.state_dict(),
            "anchors": server_stats.anchors.detach().cpu(),
            "anchor_means": server_stats.means.detach().cpu(),
            "anchor_variances": server_stats.variances.detach().cpu(),
            "anchor_counts": server_stats.counts.detach().cpu(),
            "anchor_active": server_stats.active.detach().cpu(),
            "input_means": server_stats.input_means.detach().cpu(),
            "input_variances": server_stats.input_variances.detach().cpu(),
            "input_counts": server_stats.input_counts.detach().cpu(),
            "records": records,
        },
        checkpoint_path,
    )
    # Store the final probabilities and embeddings so every figure can be
    # regenerated without rerunning federated training.
    model.eval()
    with torch.no_grad():
        final_logits, final_embeddings = model(x_test, return_features=True)
        if config.method == "fedgeski":
            final_logits = server_stats.calibrated_logits(
                final_logits, final_embeddings, anchor_scale=5.5, calibration_weight=0.0
            )
        elif config.method == "fedta" and server_stats.active.any():
            final_logits = server_stats.calibrated_logits(
                final_logits, final_embeddings, anchor_scale=4.0, calibration_weight=0.10
            )
        final_probabilities = torch.softmax(final_logits, dim=1).cpu().numpy()
    np.savez_compressed(
        predictions_path,
        y_true=y_test_np,
        probabilities=final_probabilities,
        embeddings=final_embeddings.detach().cpu().numpy(),
    )
    if verbose:
        print(f"[saved] {history_path.name} ({elapsed:.1f} s)")
    return history_path
