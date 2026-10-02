from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TabularEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 96,
        embed_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes, bias=True)
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


def make_model(
    input_dim: int,
    num_classes: int,
    hidden_dim: int,
    embed_dim: int,
    dropout: float,
    device: torch.device,
) -> TabularEncoder:
    return TabularEncoder(input_dim, num_classes, hidden_dim, embed_dim, dropout).to(device)


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def trainable_numel(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])


def flatten_gradients(model: nn.Module) -> torch.Tensor:
    pieces = []
    for parameter in model.parameters():
        if parameter.grad is None:
            pieces.append(torch.zeros_like(parameter).reshape(-1))
        else:
            pieces.append(parameter.grad.reshape(-1))
    return torch.cat(pieces)


def assign_flat_gradients(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        block = flat[offset : offset + count].view_as(parameter)
        if parameter.grad is None:
            parameter.grad = block.clone()
        else:
            parameter.grad.copy_(block)
        offset += count


def state_delta(local: nn.Module, base: nn.Module) -> OrderedDict[str, torch.Tensor]:
    base_state = base.state_dict()
    return OrderedDict(
        (name, tensor.detach() - base_state[name].detach())
        for name, tensor in local.state_dict().items()
    )


def flatten_delta(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in delta.values()])


def unflatten_like(flat: torch.Tensor, template: Dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    offset = 0
    for name, tensor in template.items():
        count = tensor.numel()
        output[name] = flat[offset : offset + count].view_as(tensor)
        offset += count
    return output


def regular_simplex(num_classes: int, embed_dim: int, device: torch.device) -> torch.Tensor:
    if embed_dim < num_classes - 1:
        raise ValueError("embed_dim must be at least num_classes - 1")
    identity = torch.eye(num_classes, dtype=torch.float32, device=device)
    centered = identity - torch.ones_like(identity) / num_classes
    centered = F.normalize(centered, dim=1)
    if embed_dim == num_classes:
        return centered
    # A deterministic orthogonal embedding retains all pairwise inner products.
    generator = torch.Generator(device=device)
    generator.manual_seed(2026)
    random_matrix = torch.randn(num_classes, embed_dim, generator=generator, device=device)
    q, _ = torch.linalg.qr(random_matrix.T, mode="reduced")
    return F.normalize(centered @ q.T, dim=1)


class SimplexStatistics:
    """Server-side moment statistics; no raw feature or traffic sample is retained."""

    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        device: torch.device,
        input_dim: int = 0,
    ) -> None:
        self.anchors = regular_simplex(num_classes, embed_dim, device)
        self.means = torch.zeros(num_classes, embed_dim, device=device)
        self.variances = torch.ones(num_classes, embed_dim, device=device)
        self.counts = torch.zeros(num_classes, device=device)
        self.active = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.input_means = torch.zeros(num_classes, input_dim, device=device)
        self.input_variances = torch.ones(num_classes, input_dim, device=device)
        self.input_counts = torch.zeros(num_classes, device=device)

    @torch.no_grad()
    def update(
        self,
        class_ids: torch.Tensor,
        means: torch.Tensor,
        variances: torch.Tensor,
        counts: torch.Tensor,
        input_means: Optional[torch.Tensor] = None,
        input_variances: Optional[torch.Tensor] = None,
        input_counts: Optional[torch.Tensor] = None,
        momentum: float = 0.0,
    ) -> None:
        for row, class_id_tensor in enumerate(class_ids):
            class_id = int(class_id_tensor.item())
            incoming = float(counts[row].item())
            if incoming <= 0:
                continue
            if not self.active[class_id]:
                self.means[class_id] = means[row]
                self.variances[class_id] = variances[row].clamp_min(1e-5)
                self.counts[class_id] = counts[row]
                self.active[class_id] = True
            else:
                old_count = float(self.counts[class_id].item())
                adaptive = incoming / (incoming + old_count)
                previous_mean = self.means[class_id].clone()
                between = (1.0 - adaptive) * adaptive * (previous_mean - means[row]).pow(2)
                self.means[class_id].lerp_(means[row], adaptive)
                self.variances[class_id].mul_(1.0 - adaptive).add_(
                    variances[row].clamp_min(1e-5), alpha=adaptive
                ).add_(between).clamp_min_(1e-5)
                self.counts[class_id] += counts[row]
            if input_means is not None and self.input_means.shape[1] > 0:
                incoming_input = float(input_counts[row].item()) if input_counts is not None else incoming
                if self.input_counts[class_id] <= 0:
                    self.input_means[class_id] = input_means[row]
                    self.input_variances[class_id] = input_variances[row].clamp_min(1e-5)
                    self.input_counts[class_id] = incoming_input
                else:
                    old_input_count = float(self.input_counts[class_id].item())
                    adaptive_input = incoming_input / (incoming_input + old_input_count)
                    previous_input_mean = self.input_means[class_id].clone()
                    between_input = (
                        (1.0 - adaptive_input)
                        * adaptive_input
                        * (previous_input_mean - input_means[row]).pow(2)
                    )
                    self.input_means[class_id].lerp_(input_means[row], adaptive_input)
                    self.input_variances[class_id].mul_(1.0 - adaptive_input).add_(
                        input_variances[row].clamp_min(1e-5), alpha=adaptive_input
                    ).add_(between_input).clamp_min_(1e-5)
                    self.input_counts[class_id] += incoming_input

    @torch.no_grad()
    def update_input(
        self,
        class_ids: torch.Tensor,
        input_means: torch.Tensor,
        input_variances: torch.Tensor,
        input_counts: torch.Tensor,
    ) -> None:
        """Pool only the input moments used by FedGeSKI training.

        Latent moments are intentionally not retained by the default protocol;
        client-side anchor alignment is transmitted only as scalar reliability.
        """
        for row, class_id_tensor in enumerate(class_ids):
            class_id = int(class_id_tensor.item())
            incoming = float(input_counts[row].item())
            if incoming <= 0:
                continue
            if self.input_counts[class_id] <= 0:
                self.input_means[class_id] = input_means[row]
                self.input_variances[class_id] = input_variances[row].clamp_min(1e-5)
                self.input_counts[class_id] = incoming
            else:
                old_count = float(self.input_counts[class_id].item())
                adaptive = incoming / (incoming + old_count)
                previous_mean = self.input_means[class_id].clone()
                between = (
                    (1.0 - adaptive) * adaptive
                    * (previous_mean - input_means[row]).pow(2)
                )
                self.input_means[class_id].lerp_(input_means[row], adaptive)
                self.input_variances[class_id].mul_(1.0 - adaptive).add_(
                    input_variances[row].clamp_min(1e-5), alpha=adaptive
                ).add_(between).clamp_min_(1e-5)
                self.input_counts[class_id] += incoming

    def resident_bytes(self, include_latent: bool = True) -> int:
        """Return persistent class-knowledge state, including fixed anchors."""
        tensors = [self.anchors, self.input_means, self.input_variances, self.input_counts]
        if include_latent:
            tensors.extend((self.means, self.variances, self.counts, self.active))
        return int(sum(t.numel() * t.element_size() for t in tensors))

    def sample_inputs(
        self,
        samples_per_class: int,
        generator: torch.Generator,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        available = torch.nonzero(self.input_counts > 0, as_tuple=False).flatten()
        if len(available) == 0 or self.input_means.shape[1] == 0:
            return None
        samples: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        for class_id in available:
            noise = torch.randn(
                samples_per_class,
                self.input_means.shape[1],
                generator=generator,
                device=self.input_means.device,
            )
            generated = self.input_means[class_id] + noise * self.input_variances[class_id].sqrt()
            samples.append(generated.clamp(-12.0, 12.0))
            labels.append(
                torch.full(
                    (samples_per_class,),
                    int(class_id.item()),
                    dtype=torch.long,
                    device=self.input_means.device,
                )
            )
        return torch.cat(samples, dim=0), torch.cat(labels, dim=0)

    def calibrated_logits(
        self,
        model_logits: torch.Tensor,
        features: torch.Tensor,
        anchor_scale: float = 5.0,
        calibration_weight: float = 0.2,
    ) -> torch.Tensor:
        normalized = F.normalize(features, dim=1)
        anchor_logits = anchor_scale * (normalized @ self.anchors.T)
        active = self.active if self.active.any() else self.input_counts > 0
        counts = self.counts if self.active.any() else self.input_counts
        if active.any():
            source_variances = self.variances if self.active.any() else self.input_variances
            dispersion = source_variances.mean(dim=1).sqrt().clamp(0.05, 5.0)
            dispersion = dispersion / dispersion[active].mean().clamp_min(1e-5)
            anchor_logits = anchor_logits / (1.0 + calibration_weight * dispersion.unsqueeze(0))
            counts = counts.clamp_min(1.0)
            active_mean = counts[active].log().mean()
            bias = -(counts.log() - active_mean) * calibration_weight
            anchor_logits = anchor_logits + bias.unsqueeze(0)
        inactive = ~active
        anchor_logits[:, inactive] = -20.0
        return (1.0 - calibration_weight) * model_logits + calibration_weight * anchor_logits


def class_feature_statistics(
    features: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    class_ids = torch.unique(labels)
    means: List[torch.Tensor] = []
    variances: List[torch.Tensor] = []
    counts: List[torch.Tensor] = []
    for class_id in class_ids:
        selected = features[labels == class_id]
        means.append(selected.mean(dim=0))
        variances.append(selected.var(dim=0, unbiased=False).clamp_min(1e-5))
        counts.append(torch.tensor(float(len(selected)), device=features.device))
    return class_ids, torch.stack(means), torch.stack(variances), torch.stack(counts)
