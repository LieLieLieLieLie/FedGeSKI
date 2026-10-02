from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import ExperimentConfig
from data_stream import FederatedContinualStream
from models import (
    SimplexStatistics,
    assign_flat_gradients,
    class_feature_statistics,
    clone_model,
    flatten_delta,
    flatten_gradients,
    state_delta,
    unflatten_like,
)


class ReplayBuffer:
    def __init__(self, per_class: int, seed: int) -> None:
        self.per_class = per_class
        self.rng = np.random.default_rng(seed)
        self.indices: Dict[int, List[int]] = {}
        self.seen: Dict[int, int] = {}

    def update(self, indices: np.ndarray, labels: np.ndarray) -> None:
        for index, label_value in zip(indices.tolist(), labels.tolist()):
            label = int(label_value)
            bucket = self.indices.setdefault(label, [])
            self.seen[label] = self.seen.get(label, 0) + 1
            if len(bucket) < self.per_class:
                bucket.append(int(index))
                continue
            position = int(self.rng.integers(0, self.seen[label]))
            if position < self.per_class:
                bucket[position] = int(index)

    def sample(
        self,
        stream: FederatedContinualStream,
        client_id: int,
        batch_size: int,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        candidates = [index for bucket in self.indices.values() for index in bucket]
        if not candidates:
            return None
        chosen = self.rng.choice(candidates, size=min(batch_size, len(candidates)), replace=False)
        x = stream.x_train[chosen] * stream.client_scale[client_id] + stream.client_shift[client_id]
        y = stream.y_train[chosen]
        return (
            torch.as_tensor(x, dtype=torch.float32, device=device),
            torch.as_tensor(y, dtype=torch.long, device=device),
        )

    def bytes(self, feature_dim: int) -> int:
        samples = sum(len(bucket) for bucket in self.indices.values())
        return int(samples * (feature_dim * 4 + 8))


@dataclass
class ClientState:
    replay: ReplayBuffer
    class_counts: Dict[int, int] = field(default_factory=dict)
    drift_mean: float = 0.0
    drift_cumulative: float = 0.0
    drift_minimum: float = 0.0
    drift_events: int = 0


@dataclass
class LocalResult:
    delta: Dict[str, torch.Tensor]
    samples: int
    train_loss: float
    reliability: float
    anchor_alignment: float
    staleness: int
    class_ids: torch.Tensor
    feature_means: torch.Tensor
    feature_variances: torch.Tensor
    feature_counts: torch.Tensor
    input_means: torch.Tensor
    input_variances: torch.Tensor
    input_counts: torch.Tensor
    communication_bytes: int
    uplink_communication_bytes: int
    downlink_communication_bytes: int
    auxiliary_memory_bytes: int
    drift_event: bool
    update_id: Tuple[int, int]


def _encode_input_bank_snapshot(
    server_stats: SimplexStatistics,
) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], int]:
    """Encode the server-to-client input-moment bank used for local augmentation.

    Only active class identifiers and the corresponding input-space means and
    variances are required to draw the transient samples.  The snapshot uses
    the same FP16 moment representation as the uplink capsule and is decoded
    before local arithmetic.  Counts and latent moments are intentionally not
    broadcast because local sampling does not consume them.
    """
    available = torch.nonzero(server_stats.input_counts > 0, as_tuple=False).flatten()
    if available.numel() == 0 or server_stats.input_means.shape[1] == 0:
        return None, 0
    encoded_ids = available.to(torch.uint8)
    encoded_means = server_stats.input_means[available].to(torch.float16)
    encoded_variances = server_stats.input_variances[available].to(torch.float16)
    payload_bytes = int(
        encoded_ids.numel() * encoded_ids.element_size()
        + encoded_means.numel() * encoded_means.element_size()
        + encoded_variances.numel() * encoded_variances.element_size()
    )
    return (
        encoded_ids.to(torch.long),
        encoded_means.to(torch.float32),
        encoded_variances.to(torch.float32),
    ), payload_bytes


def _sample_input_bank_snapshot(
    snapshot: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    samples_per_class: int,
    generator: torch.Generator,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if snapshot is None:
        return None
    class_ids, means, variances = snapshot
    samples: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for row, class_id in enumerate(class_ids):
        noise = torch.randn(
            samples_per_class,
            means.shape[1],
            generator=generator,
            device=means.device,
        )
        samples.append((means[row] + noise * variances[row].clamp_min(1e-5).sqrt()).clamp(-12.0, 12.0))
        labels.append(
            torch.full(
                (samples_per_class,),
                int(class_id.item()),
                dtype=torch.long,
                device=means.device,
            )
        )
    return torch.cat(samples, dim=0), torch.cat(labels, dim=0)


def _encode_statistic_capsule(
    class_ids: torch.Tensor,
    feature_means: torch.Tensor,
    feature_variances: torch.Tensor,
    feature_counts: torch.Tensor,
    input_means: torch.Tensor,
    input_variances: torch.Tensor,
    input_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Simulate the compact wire format and restore server arithmetic to FP32.

    Moments use IEEE FP16, counts use signed 16-bit integers, and class
    identifiers use unsigned 8-bit integers.  The returned tensors deliberately
    round-trip through those dtypes so experiments include codec error instead
    of merely reporting a smaller byte count.
    """
    if torch.any(feature_counts < 0) or torch.any(input_counts < 0):
        raise ValueError("Moment-statistic counts must be non-negative")
    if torch.any(feature_counts > 32767) or torch.any(input_counts > 32767):
        raise ValueError("Statistic capsule Int16 count overflow")

    encoded_class_ids = class_ids.to(torch.uint8)
    encoded_feature_counts = feature_counts.round().to(torch.int16)
    encoded_input_counts = input_counts.round().to(torch.int16)
    encoded_moments = tuple(
        tensor.to(torch.float16)
        for tensor in (feature_means, feature_variances, input_means, input_variances)
    )
    payload_bytes = int(
        encoded_class_ids.numel() * encoded_class_ids.element_size()
        + encoded_feature_counts.numel() * encoded_feature_counts.element_size()
        + encoded_input_counts.numel() * encoded_input_counts.element_size()
        + sum(tensor.numel() * tensor.element_size() for tensor in encoded_moments)
    )
    return (
        encoded_moments[0].to(torch.float32),
        encoded_moments[1].to(torch.float32),
        encoded_feature_counts.to(torch.float32),
        encoded_moments[2].to(torch.float32),
        encoded_moments[3].to(torch.float32),
        encoded_input_counts.to(torch.float32),
        payload_bytes,
    )


def _encode_input_statistic_capsule(
    class_ids: torch.Tensor,
    input_means: torch.Tensor,
    input_variances: torch.Tensor,
    input_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Encode the training-relevant FedGeSKI capsule.

    Pooled latent moments are deliberately excluded: the server does not use
    them for aggregation or consolidation.  Alignment is reduced locally to a
    scalar reliability value, while the transmitted bank contains only the
    input moments required by transient moment replay.
    """
    if torch.any(input_counts < 0) or torch.any(input_counts > 32767):
        raise ValueError("Input-statistic Int16 count overflow")
    encoded_ids = class_ids.to(torch.uint8)
    encoded_means = input_means.to(torch.float16)
    encoded_variances = input_variances.to(torch.float16)
    encoded_counts = input_counts.round().to(torch.int16)
    # Four bytes for reliability and four bytes for the immutable update ID.
    metadata_bytes = 8
    payload_bytes = int(
        encoded_ids.numel() * encoded_ids.element_size()
        + encoded_means.numel() * encoded_means.element_size()
        + encoded_variances.numel() * encoded_variances.element_size()
        + encoded_counts.numel() * encoded_counts.element_size()
        + metadata_bytes
    )
    return (
        encoded_means.to(torch.float32),
        encoded_variances.to(torch.float32),
        encoded_counts.to(torch.float32),
        payload_bytes,
    )


def make_client_states(config: ExperimentConfig) -> List[ClientState]:
    return [
        ClientState(ReplayBuffer(config.replay_per_class, config.seed * 1000 + client_id))
        for client_id in range(config.num_clients)
    ]


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if len(features) < 2:
        return features.sum() * 0.0
    normalized = F.normalize(features, dim=1)
    logits = normalized @ normalized.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    identity = torch.eye(len(features), device=features.device, dtype=torch.bool)
    positives = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~identity
    denominator = torch.logsumexp(logits.masked_fill(identity, -1e9), dim=1)
    log_probability = logits - denominator.unsqueeze(1)
    valid = positives.sum(dim=1) > 0
    if not valid.any():
        return features.sum() * 0.0
    return -(log_probability * positives).sum(dim=1)[valid].div(positives.sum(dim=1)[valid]).mean()


def _reference_penalty(local: nn.Module, reference: nn.Module) -> torch.Tensor:
    return sum(
        (local_parameter - reference_parameter.detach()).pow(2).sum()
        for local_parameter, reference_parameter in zip(local.parameters(), reference.parameters())
    )


def _focal_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 1.5) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    probability = torch.exp(-losses)
    return (((1.0 - probability) ** gamma) * losses).mean()


@torch.no_grad()
def _page_hinkley(state: ClientState, residual: float, threshold: float = 0.12) -> bool:
    state.drift_mean = 0.95 * state.drift_mean + 0.05 * residual
    state.drift_cumulative += residual - state.drift_mean - 0.002
    state.drift_minimum = min(state.drift_minimum, state.drift_cumulative)
    event = state.drift_cumulative - state.drift_minimum > threshold
    if event:
        state.drift_events += 1
        state.drift_cumulative = 0.0
        state.drift_minimum = 0.0
    return event


def train_client(
    method: str,
    current_global: nn.Module,
    stale_global: nn.Module,
    stream: FederatedContinualStream,
    client_state: ClientState,
    config: ExperimentConfig,
    server_stats: SimplexStatistics,
    stable_basis: Optional[torch.Tensor],
    client_id: int,
    round_id: int,
    staleness: int,
) -> LocalResult:
    device = next(current_global.parameters()).device
    local = clone_model(stale_global).to(device)
    reference = clone_model(current_global).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    local.train()
    optimizer = torch.optim.AdamW(
        local.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed * 100000 + round_id * 100 + client_id)
    raw_indices: List[np.ndarray] = []
    feature_batches: List[torch.Tensor] = []
    label_batches: List[torch.Tensor] = []
    losses: List[float] = []
    anchor_residuals: List[float] = []
    input_batches: List[torch.Tensor] = []
    input_label_batches: List[torch.Tensor] = []
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(config.seed * 100000 + round_id * 100 + client_id + 17)

    uses_global_input_bank = (
        method == "fedgeski" and config.variant not in {"no_stats", "simplex_only"}
    )
    input_bank_snapshot, input_bank_downlink_bytes = (
        _encode_input_bank_snapshot(server_stats) if uses_global_input_bank else (None, 0)
    )

    uses_replay = method in {"glfc", "evofedids", "fedagc"}
    for _ in range(config.local_steps):
        x_current, y_current, indices = stream.sample_batch(
            client_id, round_id, config.batch_size, rng, device
        )
        raw_indices.append(indices)
        replay = (
            client_state.replay.sample(stream, client_id, config.batch_size // 2, device)
            if uses_replay
            else None
        )

        if method == "fedagc" and replay is not None:
            optimizer.zero_grad(set_to_none=True)
            logits_current, features_current = local(x_current, return_features=True)
            current_loss = F.cross_entropy(logits_current, y_current)
            current_loss.backward()
            current_gradient = flatten_gradients(local).detach().clone()

            optimizer.zero_grad(set_to_none=True)
            x_replay, y_replay = replay
            logits_replay, _ = local(x_replay, return_features=True)
            replay_loss = F.cross_entropy(logits_replay, y_replay)
            replay_loss.backward()
            replay_gradient = flatten_gradients(local).detach().clone()
            dot = torch.dot(current_gradient, replay_gradient)
            if dot < 0:
                magnitude = current_gradient.norm() / replay_gradient.norm().clamp_min(1e-8)
                correction = dot / replay_gradient.pow(2).sum().clamp_min(1e-8)
                corrected = current_gradient - correction * replay_gradient
                corrected = corrected + 0.35 * magnitude.clamp(max=2.0) * replay_gradient
            else:
                corrected = current_gradient + 0.25 * replay_gradient
            assign_flat_gradients(local, corrected)
            optimizer.step()
            loss = current_loss + 0.25 * replay_loss
            features = features_current
            labels = y_current
        else:
            if uses_global_input_bank:
                synthetic = _sample_input_bank_snapshot(
                    input_bank_snapshot,
                    samples_per_class=config.synthetic_per_class, generator=torch_generator
                )
            else:
                synthetic = None
            if replay is not None:
                x_replay, y_replay = replay
                x_batch = torch.cat([x_current, x_replay], dim=0)
                y_batch = torch.cat([y_current, y_replay], dim=0)
            elif synthetic is not None:
                x_synthetic, y_synthetic = synthetic
                x_batch = torch.cat([x_current, x_synthetic], dim=0)
                y_batch = torch.cat([y_current, y_synthetic], dim=0)
            else:
                x_batch, y_batch = x_current, y_current

            optimizer.zero_grad(set_to_none=True)
            logits, features = local(x_batch, return_features=True)
            did_step = False
            if method == "glfc":
                loss = _focal_cross_entropy(logits, y_batch)
                with torch.no_grad():
                    teacher_logits = reference(x_batch)
                old_classes = torch.unique(y_batch)
                kd = F.kl_div(
                    F.log_softmax(logits[:, old_classes] / 2.0, dim=1),
                    F.softmax(teacher_logits[:, old_classes] / 2.0, dim=1),
                    reduction="batchmean",
                ) * 4.0
                loss = loss + config.distill_weight * kd
            elif method == "fedgcc":
                # Fixed-head adaptation of global classifier consensus: the
                # cumulative quantity term counteracts client-local imbalance,
                # while confidence-weighted distillation keeps every client
                # aligned with the current global classifier without task IDs.
                for class_id, count in zip(*torch.unique(y_current, return_counts=True)):
                    key = int(class_id.item())
                    client_state.class_counts[key] = client_state.class_counts.get(key, 0) + int(count.item())
                class_weight = torch.ones(logits.shape[1], device=device)
                for class_id, count in client_state.class_counts.items():
                    class_weight[class_id] = 1.0 / math.sqrt(max(count, 1))
                active = torch.tensor(list(client_state.class_counts), device=device, dtype=torch.long)
                class_weight = class_weight / class_weight[active].mean().clamp_min(1e-6)
                with torch.no_grad():
                    teacher_logits = reference(x_batch)
                    teacher_confidence = torch.softmax(teacher_logits, dim=1).max(dim=1).values
                ce = F.cross_entropy(logits, y_batch, weight=class_weight)
                kd_per_sample = F.kl_div(
                    F.log_softmax(logits / 2.0, dim=1),
                    F.softmax(teacher_logits / 2.0, dim=1),
                    reduction="none",
                ).sum(dim=1) * 4.0
                quality_kd = (teacher_confidence * kd_per_sample).sum() / teacher_confidence.sum().clamp_min(1e-6)
                loss = ce + 0.55 * quality_kd + 2e-5 * _reference_penalty(local, reference)
            elif method == "afcl_csc":
                # Protocol-aligned client--server cooperative control: local
                # sharpness-aware training preserves a flat solution, and the
                # server later combines it with similarity/staleness weighting.
                with torch.no_grad():
                    teacher_logits = reference(x_batch)
                base_loss = F.cross_entropy(logits, y_batch)
                base_loss = base_loss + 0.45 * F.kl_div(
                    F.log_softmax(logits / 2.0, dim=1),
                    F.softmax(teacher_logits / 2.0, dim=1),
                    reduction="batchmean",
                ) * 4.0
                base_loss.backward()
                grad_norm = torch.sqrt(sum(
                    parameter.grad.detach().pow(2).sum()
                    for parameter in local.parameters() if parameter.grad is not None
                )).clamp_min(1e-8)
                perturbations = []
                with torch.no_grad():
                    for parameter in local.parameters():
                        perturbation = (
                            torch.zeros_like(parameter)
                            if parameter.grad is None
                            else 0.025 * parameter.grad / grad_norm
                        )
                        parameter.add_(perturbation)
                        perturbations.append(perturbation)
                optimizer.zero_grad(set_to_none=True)
                logits_adv, _ = local(x_batch, return_features=True)
                loss = F.cross_entropy(logits_adv, y_batch)
                loss = loss + 0.45 * F.kl_div(
                    F.log_softmax(logits_adv / 2.0, dim=1),
                    F.softmax(teacher_logits / 2.0, dim=1),
                    reduction="batchmean",
                ) * 4.0
                loss.backward()
                with torch.no_grad():
                    for parameter, perturbation in zip(local.parameters(), perturbations):
                        parameter.sub_(perturbation)
                torch.nn.utils.clip_grad_norm_(local.parameters(), max_norm=8.0)
                optimizer.step()
                did_step = True
            elif method == "evofedids":
                loss = F.cross_entropy(logits, y_batch)
                loss = loss + 0.18 * supervised_contrastive_loss(
                    features, y_batch, config.temperature
                )
                loss = loss + 2e-5 * _reference_penalty(local, reference)
            elif method == "fedta":
                normalized = F.normalize(features, dim=1)
                anchor_logits = normalized @ server_stats.anchors.T / config.temperature
                target_anchor = server_stats.anchors[y_batch]
                loss = F.cross_entropy(logits, y_batch)
                loss = loss + 0.55 * F.cross_entropy(anchor_logits, y_batch)
                loss = loss + 0.20 * (1.0 - (normalized * target_anchor).sum(dim=1)).mean()
            elif method == "fedgeski":
                normalized = F.normalize(features, dim=1)
                anchor_logits = normalized @ server_stats.anchors.T / config.temperature
                target_anchor = server_stats.anchors[y_batch]
                alignment = (1.0 - (normalized * target_anchor).sum(dim=1)).mean()
                if config.variant in {"no_simplex", "stats_only", "stats_fusion"}:
                    loss = F.cross_entropy(logits, y_batch)
                else:
                    loss = F.cross_entropy(logits + 0.35 * anchor_logits, y_batch)
                    loss = loss + config.anchor_weight * F.cross_entropy(anchor_logits, y_batch)
                    loss = loss + 0.30 * alignment
                loss = loss + config.proximal_weight * _reference_penalty(local, reference)
                anchor_residuals.append(float(alignment.detach().item()))
            else:
                loss = F.cross_entropy(logits, y_batch)
            if not did_step:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local.parameters(), max_norm=8.0)
                optimizer.step()
            labels = y_batch

        losses.append(float(loss.detach().item()))
        feature_batches.append(features[: len(y_current)].detach())
        label_batches.append(y_current.detach())
        input_batches.append(x_current.detach())
        input_label_batches.append(y_current.detach())

    all_indices = np.concatenate(raw_indices)
    if uses_replay:
        client_state.replay.update(all_indices, stream.y_train[all_indices])

    features_all = torch.cat(feature_batches, dim=0)
    labels_all = torch.cat(label_batches, dim=0)
    class_ids, means, variances, counts = class_feature_statistics(features_all, labels_all)
    inputs_all = torch.cat(input_batches, dim=0)
    input_labels_all = torch.cat(input_label_batches, dim=0)
    input_class_ids, input_means, input_variances, input_counts = class_feature_statistics(
        inputs_all, input_labels_all
    )
    if not torch.equal(class_ids, input_class_ids):
        raise RuntimeError("Feature and input moment-statistic class order diverged")
    base_delta = state_delta(local, stale_global)

    with torch.no_grad():
        normalized_means = F.normalize(means, dim=1)
        alignment = (normalized_means * server_stats.anchors[class_ids]).sum(dim=1).mean()
        reliability = float(torch.sigmoid(4.0 * (alignment - 0.35)).item())

    drift_event = _page_hinkley(
        client_state,
        float(np.mean(anchor_residuals)) if anchor_residuals else float(1.0 - alignment.item()),
    )
    # Every method uses the same FP16 wire codec. The server restores FP32
    # arithmetic after decoding, so communication comparisons are not driven by
    # method-specific numeric precision.
    base_delta = type(base_delta)(
        (name, tensor.to(torch.float16).to(tensor.dtype))
        for name, tensor in base_delta.items()
    )
    parameter_bytes = int(sum(tensor.numel() * 2 for tensor in base_delta.values()))
    statistics_bytes = int(
        (
            means.numel()
            + variances.numel()
            + counts.numel()
            + input_means.numel()
            + input_variances.numel()
            + input_counts.numel()
        )
        * 4
    )
    transmitted_statistics_bytes = statistics_bytes
    if method == "fedgeski":
        input_means, input_variances, input_counts, transmitted_statistics_bytes = (
            _encode_input_statistic_capsule(
            class_ids,
            input_means,
            input_variances,
            input_counts,
            )
        )
    auxiliary_memory = (
        client_state.replay.bytes(stream.num_features)
        if uses_replay
        else (len(client_state.class_counts) * 8 if method == "fedgcc" else statistics_bytes)
    )
    uplink_bytes = parameter_bytes + (
        transmitted_statistics_bytes if method in {"fedta", "fedgeski"} else 0
    )
    return LocalResult(
        delta=base_delta,
        samples=int(config.local_steps * config.batch_size),
        train_loss=float(np.mean(losses)),
        reliability=max(reliability, 1e-4),
        anchor_alignment=float(alignment.item()),
        staleness=staleness,
        class_ids=class_ids.detach(),
        feature_means=means.detach(),
        feature_variances=variances.detach(),
        feature_counts=counts.detach(),
        input_means=input_means.detach(),
        input_variances=input_variances.detach(),
        input_counts=input_counts.detach(),
        communication_bytes=uplink_bytes + parameter_bytes + input_bank_downlink_bytes,
        uplink_communication_bytes=uplink_bytes,
        downlink_communication_bytes=parameter_bytes + input_bank_downlink_bytes,
        auxiliary_memory_bytes=auxiliary_memory,
        drift_event=drift_event,
        update_id=(client_id, round_id),
    )
