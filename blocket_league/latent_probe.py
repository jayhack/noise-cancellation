from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ClipDataset
from .latent_model import (
    CausalLatentDiT,
    FlowMatchingSchedule,
    LatentWorldModelConfig,
    load_latent_world_checkpoint,
    normalize_latents,
    unnormalize_latents,
)
from .metrics import _soft_centroid
from .env import PALETTE


CONTINUOUS_TARGETS = (
    "player_x",
    "player_y",
    "player_vx",
    "player_vy",
    "puck_x",
    "puck_y",
    "puck_vx",
    "puck_vy",
    "player_speed",
    "puck_speed",
    "relative_distance",
    "bearing_cos",
    "bearing_sin",
)

BINARY_TARGETS = (
    "any_collision",
    "disc_impact",
    "wall_hit",
)

TASK_TARGETS = {
    "position": ("player_x", "player_y", "puck_x", "puck_y"),
    "velocity": ("player_vx", "player_vy", "puck_vx", "puck_vy"),
    "speed": ("player_speed", "puck_speed"),
    "polar": ("relative_distance", "bearing_cos", "bearing_sin"),
}


@dataclass
class RidgeFit:
    r2: torch.Tensor
    weights: torch.Tensor
    x_scale: torch.Tensor
    y_scale: torch.Tensor
    feature_dimension: int
    ridge: float

    def raw_unit_direction(self, target: int) -> torch.Tensor:
        gradient = self.weights[:, target] / (self.x_scale * sqrt(self.feature_dimension))
        return gradient / gradient.square().sum().clamp_min(1e-12)


@dataclass
class PuckLocator:
    weight: torch.Tensor
    bias: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor

    def scores(self, tokens: torch.Tensor) -> torch.Tensor:
        return ((tokens - self.mean) / self.scale) @ self.weight + self.bias

    def cells(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.scores(tokens).argmax(dim=1)


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _video(batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.cat((batch["context"], batch["target"]), dim=1).to(device, non_blocking=True)


def derive_probe_targets(states: torch.Tensor, events: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build end-of-pair state targets and collision-in-the-next-pair labels."""

    final = states[:, -1, :8].float()
    player_velocity = final[:, 2:4]
    puck_velocity = final[:, 6:8]
    relative = final[:, 4:6] - final[:, :2]
    distance = torch.linalg.vector_norm(relative, dim=1).clamp_min(1e-6)
    continuous = torch.cat(
        (
            final,
            torch.linalg.vector_norm(player_velocity, dim=1, keepdim=True),
            torch.linalg.vector_norm(puck_velocity, dim=1, keepdim=True),
            distance[:, None],
            relative[:, :1] / distance[:, None],
            relative[:, 1:2] / distance[:, None],
        ),
        dim=1,
    )
    impact = (events == 2).any(dim=1)
    wall = (events == 3).any(dim=1)
    binary = torch.stack((impact | wall, impact, wall), dim=1).float()
    return continuous, binary


def _input_tokens(
    model: CausalLatentDiT,
    noisy_sequence: torch.Tensor,
    clean_past: torch.Tensor,
) -> torch.Tensor:
    batch, frames, channels, height, width = noisy_sequence.shape
    tokens = noisy_sequence.permute(0, 1, 3, 4, 2).reshape(batch, frames, height * width, channels)
    past = clean_past.permute(0, 1, 3, 4, 2).reshape(batch, frames, height * width, channels)
    return (
        model.input_projection(tokens)
        + model.past_projection(past)
        + model.spatial_position
        + model.temporal_position[:, :frames]
    )


def _visual_puck_cells(frame: torch.Tensor, grid: int) -> torch.Tensor:
    """Locate the visually white puck without using simulator coordinates."""

    color = torch.as_tensor(PALETTE["puck"], device=frame.device, dtype=frame.dtype).div(127.5).sub(1.0)
    distance = (frame - color[None, :, None, None]).square().sum(dim=1)
    weights = torch.exp(-distance / 0.04)
    pooled = F.adaptive_avg_pool2d(weights[:, None], (grid, grid)).flatten(1)
    return pooled.argmax(dim=1)


@torch.no_grad()
def fit_activation_puck_locator(
    model: CausalLatentDiT,
    codec: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    block_index: int,
    dataset_seed: int,
    dataset_offset: int,
    samples: int = 256,
    batch_size: int = 32,
) -> tuple[PuckLocator, dict[str, Any]]:
    """Fit a shared token scorer using visual self-supervision, never coordinate labels."""

    device = next(model.parameters()).device
    temporal = model.config.temporal_downsample
    context_frames = (model.config.max_sequence_latents - 1) * temporal
    grid = model.config.latent_grid_size
    dataset = ClipDataset(
        samples,
        seed=dataset_seed + dataset_offset * 9_973,
        context_frames=context_frames,
        future_frames=temporal,
        image_size=codec.config.image_size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    activation_chunks: list[torch.Tensor] = []
    visual_cell_chunks: list[torch.Tensor] = []
    evaluation_cell_chunks: list[torch.Tensor] = []
    for batch in loader:
        context = batch["context"].to(device)
        raw_latents, _ = codec.encode(context)
        latents = normalize_latents(raw_latents.float(), mean, std)
        actions = torch.zeros(
            latents.shape[0],
            latents.shape[1],
            temporal,
            device=device,
            dtype=torch.long,
        )
        times = torch.ones(latents.shape[:2], device=device)
        result = model(latents, actions, times, model.shifted_clean_past(latents), return_hidden=True)
        if isinstance(result, torch.Tensor):
            raise RuntimeError("Latent model did not return hidden states")
        hidden = result[1][block_index].reshape(
            latents.shape[0], latents.shape[1], grid * grid, model.config.hidden_size
        )[:, -1]
        activation_chunks.append(hidden.cpu())
        visual_cell_chunks.append(_visual_puck_cells(context[:, -1], grid).cpu())
        evaluation_xy = batch["state"][:, 0, 4:6]
        evaluation_cell_chunks.append(
            ((evaluation_xy * grid).long().clamp(0, grid - 1)[:, 1] * grid
             + (evaluation_xy * grid).long().clamp(0, grid - 1)[:, 0]).cpu()
        )

    activations = torch.cat(activation_chunks).to(device)
    visual_cells = torch.cat(visual_cell_chunks).to(device)
    evaluation_cells = torch.cat(evaluation_cell_chunks).to(device)
    train_count = int(samples * 0.6)
    validation_count = int(samples * 0.2)
    fit_end = train_count + validation_count
    train_tokens = activations[:fit_end]
    feature_mean = train_tokens.reshape(-1, model.config.hidden_size).mean(dim=0)
    feature_scale = train_tokens.reshape(-1, model.config.hidden_size).std(dim=0).clamp_min(1e-5)
    standardized = (activations - feature_mean) / feature_scale
    token_targets = F.one_hot(visual_cells, num_classes=grid * grid).float()
    x_fit = standardized[:fit_end].reshape(-1, model.config.hidden_size)
    y_fit = token_targets[:fit_end].reshape(-1)
    x_fit = torch.cat((x_fit, torch.ones(x_fit.shape[0], 1, device=device)), dim=1)
    sample_weights = torch.where(y_fit > 0, torch.full_like(y_fit, grid * grid - 1), torch.ones_like(y_fit))
    weighted_x = x_fit * sample_weights.sqrt()[:, None]
    weighted_y = y_fit[:, None] * sample_weights.sqrt()[:, None]
    solution = _solve_ridge(weighted_x, weighted_y, 1.0).flatten()
    locator = PuckLocator(
        weight=solution[:-1],
        bias=solution[-1],
        mean=feature_mean,
        scale=feature_scale,
    )
    test_tokens = activations[fit_end:]
    predicted = locator.cells(test_tokens)
    visual_test = visual_cells[fit_end:]
    evaluation_test = evaluation_cells[fit_end:]
    row_error = predicted.div(grid, rounding_mode="floor") - visual_test.div(grid, rounding_mode="floor")
    column_error = predicted.remainder(grid) - visual_test.remainder(grid)
    return locator, {
        "trainingSignal": "puck-colored pixels in observed RGB only; no simulator coordinates",
        "deploymentSignal": "block activations only",
        "layer": f"block-{block_index + 1}",
        "samples": samples,
        "trainTrajectories": train_count,
        "validationTrajectories": validation_count,
        "testTrajectories": samples - fit_end,
        "visualCellAccuracy": _round((predicted == visual_test).float().mean()),
        "simulatorCellAccuracyEvaluationOnly": _round((predicted == evaluation_test).float().mean()),
        "withinOneCellAccuracy": _round(((row_error.abs() <= 1) & (column_error.abs() <= 1)).float().mean()),
    }


@torch.no_grad()
def collect_latent_features(
    checkpoint_path: Path,
    *,
    samples: int,
    batch_size: int,
    seed: int,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    CausalLatentDiT,
    torch.nn.Module,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, codec, mean, std, checkpoint = load_latent_world_checkpoint(checkpoint_path)
    model = model.to(device).eval()
    codec = codec.to(device).eval()
    model.requires_grad_(False)
    codec.requires_grad_(False)
    mean = mean.to(device)
    std = std.to(device)

    random_model = CausalLatentDiT(LatentWorldModelConfig(**checkpoint["model_config"]))
    random_model = random_model.to(device).eval()
    random_model.requires_grad_(False)

    temporal = model.config.temporal_downsample
    history_latents = model.config.max_sequence_latents - 1
    context_frames = history_latents * temporal
    dataset = ClipDataset(
        samples,
        seed=seed,
        context_frames=context_frames,
        future_frames=temporal,
        image_size=codec.config.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=min(6, max(0, batch_size // 8)),
        pin_memory=device.type == "cuda",
    )
    representation_names = ["codec", "input", *[f"block-{index + 1}" for index in range(model.config.depth)]]
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in representation_names}
    chunks["untrained"] = []
    chunks["action-only"] = []
    continuous_chunks: list[torch.Tensor] = []
    binary_chunks: list[torch.Tensor] = []
    generator = torch.Generator(device=device).manual_seed(seed + 811_731)

    for batch in loader:
        video = _video(batch, device)
        raw_latents, _ = codec.encode(video)
        latents = normalize_latents(raw_latents.float(), mean, std)
        history = latents[:, :-1]
        noise = torch.randn(
            latents[:, -1:].shape,
            device=device,
            dtype=latents.dtype,
            generator=generator,
        )
        noisy_sequence = torch.cat((history, noise), dim=1)
        clean_sequence = torch.cat((history, torch.zeros_like(noise)), dim=1)
        clean_past = model.shifted_clean_past(clean_sequence)
        context_actions = torch.zeros(
            history.shape[0],
            history.shape[1],
            temporal,
            device=device,
            dtype=torch.long,
        )
        future_actions = batch["actions"].to(device, non_blocking=True).reshape(-1, 1, temporal)
        action_pairs = torch.cat((context_actions, future_actions), dim=1)
        times = torch.cat(
            (
                torch.ones(history.shape[0], history.shape[1], device=device),
                torch.zeros(history.shape[0], 1, device=device),
            ),
            dim=1,
        )
        result = model(noisy_sequence, action_pairs, times, clean_past, return_hidden=True)
        if isinstance(result, torch.Tensor):
            raise RuntimeError("Latent model did not return hidden states")
        _, hidden_states = result

        chunks["codec"].append(history[:, -1].flatten(1).to(torch.float16).cpu())
        chunks["input"].append(_input_tokens(model, noisy_sequence, clean_past)[:, -1].flatten(1).to(torch.float16).cpu())
        patches = model.config.latent_grid_size**2
        for index, hidden in enumerate(hidden_states):
            chunks[f"block-{index + 1}"].append(hidden[:, -patches:].flatten(1).to(torch.float16).cpu())

        random_clean_past = random_model.shifted_clean_past(clean_sequence)
        random_result = random_model(
            noisy_sequence,
            action_pairs,
            times,
            random_clean_past,
            return_hidden=True,
        )
        if isinstance(random_result, torch.Tensor):
            raise RuntimeError("Untrained model did not return hidden states")
        chunks["untrained"].append(random_result[1][-1][:, -patches:].flatten(1).to(torch.float16).cpu())
        chunks["action-only"].append(
            F.one_hot(future_actions.flatten(1), num_classes=model.config.action_count)
            .flatten(1)
            .float()
            .cpu()
        )
        continuous, binary = derive_probe_targets(batch["state"], batch["events"])
        continuous_chunks.append(continuous)
        binary_chunks.append(binary)

    features = {name: torch.cat(values).float() for name, values in chunks.items()}
    return (
        features,
        torch.cat(continuous_chunks),
        torch.cat(binary_chunks),
        model,
        codec,
        mean,
        std,
        checkpoint,
    )


def _standardize(
    x_fit: torch.Tensor,
    x_eval: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x_fit.mean(dim=0)
    scale = x_fit.std(dim=0).clamp_min(1e-5)
    dimension_scale = sqrt(x_fit.shape[1])
    return (x_fit - mean) / scale / dimension_scale, (x_eval - mean) / scale / dimension_scale, scale


def _r2(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    residual = (target - prediction).square().sum(dim=0)
    total = (target - target.mean(dim=0)).square().sum(dim=0).clamp_min(1e-8)
    return 1.0 - residual / total


def _solve_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    if x.shape[1] > x.shape[0]:
        identity = torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
        dual = torch.linalg.solve(x @ x.T + ridge * identity, y)
        return x.T @ dual
    identity = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    return torch.linalg.solve(x.T @ x + ridge * identity, x.T @ y)


def fit_ridge_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    train_count: int,
    validation_count: int,
    device: torch.device,
) -> RidgeFit:
    fit_end = train_count + validation_count
    x_train = features[:train_count].to(device)
    x_validation = features[train_count:fit_end].to(device)
    y_train = targets[:train_count].to(device)
    y_validation = targets[train_count:fit_end].to(device)
    x_train, x_validation, _ = _standardize(x_train, x_validation)
    y_mean = y_train.mean(dim=0)
    y_scale = y_train.std(dim=0).clamp_min(1e-5)
    y_train_normalized = (y_train - y_mean) / y_scale
    ridge_grid = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    best_ridge = ridge_grid[0]
    best_score = -float("inf")
    for ridge in ridge_grid:
        weights = _solve_ridge(x_train, y_train_normalized, ridge)
        prediction = (x_validation @ weights) * y_scale + y_mean
        score = float(_r2(y_validation, prediction).mean())
        if score > best_score:
            best_score = score
            best_ridge = ridge

    x_fit_raw = features[:fit_end].to(device)
    x_test_raw = features[fit_end:].to(device)
    y_fit = targets[:fit_end].to(device)
    y_test = targets[fit_end:].to(device)
    x_fit, x_test, x_scale = _standardize(x_fit_raw, x_test_raw)
    y_mean = y_fit.mean(dim=0)
    y_scale = y_fit.std(dim=0).clamp_min(1e-5)
    weights = _solve_ridge(x_fit, (y_fit - y_mean) / y_scale, best_ridge)
    prediction = (x_test @ weights) * y_scale + y_mean
    return RidgeFit(
        r2=_r2(y_test, prediction).detach().cpu(),
        weights=weights.detach().cpu(),
        x_scale=x_scale.detach().cpu(),
        y_scale=y_scale.detach().cpu(),
        feature_dimension=features.shape[1],
        ridge=best_ridge,
    )


def binary_auc(target: torch.Tensor, score: torch.Tensor) -> float:
    target = target.float().flatten()
    score = score.float().flatten()
    positives = int(target.sum())
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(score)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(score) + 1, device=score.device, dtype=torch.float32)
    rank_sum = ranks[target.bool()].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(target: torch.Tensor, score: torch.Tensor) -> float:
    target = target.float().flatten()
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    ordered = target[torch.argsort(score, descending=True)]
    precision = ordered.cumsum(0) / torch.arange(1, len(ordered) + 1, device=ordered.device)
    return float((precision * ordered).sum() / positives)


def fit_linear_classifier(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    train_count: int,
    validation_count: int,
    device: torch.device,
) -> dict[str, list[float] | float]:
    fit_end = train_count + validation_count
    x_fit_raw = features[:fit_end].to(device)
    x_test_raw = features[fit_end:].to(device)
    y_fit = targets[:fit_end].to(device)
    y_test = targets[fit_end:].to(device)
    x_fit, x_test, _ = _standardize(x_fit_raw, x_test_raw)
    weights = _solve_ridge(x_fit, y_fit, 0.1)
    scores = x_test @ weights
    auc = [binary_auc(y_test[:, index], scores[:, index]) for index in range(targets.shape[1])]
    ap = [average_precision(y_test[:, index], scores[:, index]) for index in range(targets.shape[1])]
    return {
        "auc": auc,
        "average_precision": ap,
        "prevalence": [float(value) for value in y_test.mean(dim=0).cpu()],
    }


def _representation_payload(
    ridge: RidgeFit,
    classifier: dict[str, list[float] | float],
) -> dict[str, Any]:
    target_r2 = {name: _round(value) for name, value in zip(CONTINUOUS_TARGETS, ridge.r2)}
    tasks = {
        task: _round(sum(target_r2[name] for name in names) / len(names))
        for task, names in TASK_TARGETS.items()
    }
    auc = classifier["auc"]
    ap = classifier["average_precision"]
    prevalence = classifier["prevalence"]
    assert isinstance(auc, list) and isinstance(ap, list) and isinstance(prevalence, list)
    classification = {
        name: {
            "auc": _round(auc[index]),
            "averagePrecision": _round(ap[index]),
            "prevalence": _round(prevalence[index]),
        }
        for index, name in enumerate(BINARY_TARGETS)
    }
    tasks["collision"] = classification["any_collision"]["auc"]
    return {
        "taskScores": tasks,
        "continuousR2": target_r2,
        "classification": classification,
        "selectedRidge": ridge.ridge,
    }


@torch.no_grad()
def _sample_intervened_pair(
    model: CausalLatentDiT,
    history: torch.Tensor,
    actions: torch.Tensor,
    *,
    block_index: int,
    direction: torch.Tensor,
    strength: float,
    integration_steps: int,
    seed: int,
) -> torch.Tensor:
    calls = 0

    def intervene(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        edited = output
        if calls == 0:
            edited = output.clone()
            edited[:, -1] = edited[:, -1] + strength * direction[None]
        calls += 1
        return edited

    hook = model.blocks[block_index].register_forward_hook(intervene)
    try:
        return FlowMatchingSchedule().sample_autoregressive(
            model,
            history,
            actions,
            rollout_frames=model.config.temporal_downsample,
            integration_steps=integration_steps,
            generator=torch.Generator(device=history.device).manual_seed(seed),
        )
    finally:
        hook.remove()


@torch.no_grad()
def run_velocity_interventions(
    model: CausalLatentDiT,
    codec: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    ridge: RidgeFit,
    *,
    block_index: int,
    dataset_seed: int,
    dataset_offset: int,
    samples: int,
    batch_size: int,
    strength: float = 1.0,
    integration_steps: int = 6,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    temporal = model.config.temporal_downsample
    context_frames = (model.config.max_sequence_latents - 1) * temporal
    dataset = ClipDataset(
        samples,
        seed=dataset_seed + dataset_offset * 9_973,
        context_frames=context_frames,
        future_frames=temporal,
        image_size=codec.config.image_size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    target_specs = (
        ("player_vx", "player", 0, 2),
        ("player_vy", "player", 1, 3),
        ("puck_vx", "puck", 0, 6),
        ("puck_vy", "puck", 1, 7),
    )
    deltas: dict[str, list[torch.Tensor]] = {name: [] for name, *_ in target_specs}
    direction_norms: dict[str, float] = {}
    sample_cursor = 0
    for batch in loader:
        video = _video(batch, device)
        raw_latents, _ = codec.encode(video)
        latents = normalize_latents(raw_latents.float(), mean, std)
        history = latents[:, :-1]
        actions = batch["actions"].to(device)
        for name, entity, axis, target_index in target_specs:
            raw_direction = ridge.raw_unit_direction(target_index)
            direction = raw_direction.reshape(
                model.config.latent_grid_size**2,
                model.config.hidden_size,
            ).to(device)
            direction_norms[name] = float(direction.norm())
            base_seed = dataset_seed + 5_000_003 + sample_cursor
            plus = _sample_intervened_pair(
                model,
                history,
                actions,
                block_index=block_index,
                direction=direction,
                strength=strength,
                integration_steps=integration_steps,
                seed=base_seed,
            )
            minus = _sample_intervened_pair(
                model,
                history,
                actions,
                block_index=block_index,
                direction=direction,
                strength=-strength,
                integration_steps=integration_steps,
                seed=base_seed,
            )
            plus_video = codec.decode(unnormalize_latents(torch.cat((history, plus), dim=1), mean, std))[:, -temporal:]
            minus_video = codec.decode(unnormalize_latents(torch.cat((history, minus), dim=1), mean, std))[:, -temporal:]
            plus_position, _ = _soft_centroid(plus_video.float(), entity)
            minus_position, _ = _soft_centroid(minus_video.float(), entity)
            plus_displacement = plus_position[:, -1, axis] - plus_position[:, 0, axis]
            minus_displacement = minus_position[:, -1, axis] - minus_position[:, 0, axis]
            deltas[name].append((plus_displacement - minus_displacement).cpu() * codec.config.image_size)
        sample_cursor += history.shape[0]

    payload: dict[str, Any] = {}
    for name, values in deltas.items():
        delta = torch.cat(values)
        payload[name] = {
            "plusMinusDisplacementPx": _round(delta.mean()),
            "medianDisplacementPx": _round(delta.median()),
            "expectedSignRate": _round((delta > 0).float().mean()),
            "directionL2": _round(direction_norms[name]),
        }
    return {
        "layer": f"block-{block_index + 1}",
        "samples": samples,
        "probeStandardDeviations": strength,
        "integrationSteps": integration_steps,
        "targets": payload,
    }


def _fit_downstream_velocity_lens(
    model: CausalLatentDiT,
    codec: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    block_index: int,
    dataset_seed: int,
    dataset_offset: int,
    samples: int,
    batch_size: int,
    integration_steps: int,
    target: str = "puck_vx",
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    """Average d(rendered puck vx)/d(block activation) across contexts and solver steps."""

    device = next(model.parameters()).device
    temporal = model.config.temporal_downsample
    context_frames = (model.config.max_sequence_latents - 1) * temporal
    dataset = ClipDataset(
        samples,
        seed=dataset_seed + dataset_offset * 9_973,
        context_frames=context_frames,
        future_frames=temporal,
        image_size=codec.config.image_size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    schedule = FlowMatchingSchedule()
    integration_times = schedule.inference_times(integration_steps, device)
    gradient_sum = torch.zeros(
        model.config.latent_grid_size**2,
        model.config.hidden_size,
        device=device,
    )
    activation_chunks: list[torch.Tensor] = []
    gradient_calls = 0
    grid = model.config.latent_grid_size
    center = (grid // 2, grid // 2)

    def align_to_puck(tokens: torch.Tensor, puck_xy: torch.Tensor) -> torch.Tensor:
        maps = tokens.reshape(tokens.shape[0], grid, grid, model.config.hidden_size)
        aligned = []
        cells = (puck_xy * grid).long().clamp(0, grid - 1)
        for item, cell in zip(maps, cells):
            aligned.append(
                torch.roll(
                    item,
                    shifts=(center[1] - int(cell[1]), center[0] - int(cell[0])),
                    dims=(0, 1),
                )
            )
        return torch.stack(aligned).reshape(tokens.shape[0], grid * grid, model.config.hidden_size)

    for batch_index, batch in enumerate(loader):
        video = _video(batch, device)
        with torch.no_grad():
            raw_latents, _ = codec.encode(video)
            latents = normalize_latents(raw_latents.float(), mean, std)
        history = latents[:, :-1]
        actions = batch["actions"].to(device).reshape(-1, 1, temporal)
        puck_xy = batch["state"][:, 0, 4:6].to(device)
        generator = torch.Generator(device=device).manual_seed(dataset_seed + 8_100_011 + batch_index)
        sample = torch.randn(
            history.shape[0],
            1,
            model.config.latent_dim,
            model.config.latent_grid_size,
            model.config.latent_grid_size,
            device=device,
            generator=generator,
        )
        captured: list[torch.Tensor] = []

        def capture(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            leaf = output.detach().requires_grad_(True)
            captured.append(leaf)
            activation_chunks.append(
                align_to_puck(leaf[:, -1].detach(), puck_xy).flatten(1).cpu()
            )
            return leaf

        hook = model.blocks[block_index].register_forward_hook(capture)
        try:
            with torch.enable_grad():
                for start, end in zip(integration_times[:-1], integration_times[1:]):
                    sequence = torch.cat((history, sample), dim=1)
                    clean_sequence = torch.cat((history, sample), dim=1)
                    history_actions = torch.zeros(
                        history.shape[0],
                        history.shape[1],
                        temporal,
                        device=device,
                        dtype=torch.long,
                    )
                    sequence_actions = torch.cat((history_actions, actions), dim=1)
                    times = torch.cat(
                        (
                            torch.ones(history.shape[0], history.shape[1], device=device),
                            start.expand(history.shape[0], 1),
                        ),
                        dim=1,
                    )
                    velocity = model(
                        sequence,
                        sequence_actions,
                        times,
                        model.shifted_clean_past(clean_sequence),
                    )
                    if not isinstance(velocity, torch.Tensor):
                        velocity = velocity[0]
                    sample = (sample + (end - start) * velocity[:, -1:]).clamp(-8.0, 8.0)

                decoded, _ = codec.decode_soft(
                    unnormalize_latents(torch.cat((history, sample), dim=1), mean, std)
                )
                puck_position, _ = _soft_centroid(decoded[:, -temporal:].float(), "puck")
                displacement = (puck_position[:, -1] - puck_position[:, 0]) * codec.config.image_size
                if target == "puck_vx":
                    objective = displacement[:, 0].mean()
                    downstream_target = "decoded puck x displacement in the next two frames"
                elif target == "puck_speed":
                    objective = torch.linalg.vector_norm(displacement, dim=1).mean()
                    downstream_target = "decoded puck speed in the next two frames"
                else:
                    raise ValueError(f"Unknown downstream lens target {target!r}")
                gradients = torch.autograd.grad(objective, captured)
        finally:
            hook.remove()

        for gradient in gradients:
            gradient_sum += align_to_puck(gradient[:, -1], puck_xy).sum(dim=0)
            gradient_calls += gradient.shape[0]

    average_gradient = gradient_sum / max(gradient_calls, 1)
    direction = average_gradient / average_gradient.norm().clamp_min(1e-12)
    activations = torch.cat(activation_chunks).to(device)
    projection = activations @ direction.flatten()
    projection_sigma = float(projection.std().clamp_min(1e-4))
    return direction.detach(), projection_sigma, {
        "contexts": samples,
        "sourceLayer": f"block-{block_index + 1}",
        "downstreamTarget": downstream_target,
        "averagedOver": "contexts, spatial tokens, and every flow-solver evaluation",
        "spatialAlignment": "gradients translated so the puck token is centered before averaging",
        "solverEvaluationsPerContext": integration_steps,
        "gradientL2": _round(average_gradient.norm()),
        "projectionSigma": _round(projection_sigma),
    }


@torch.no_grad()
def _persistent_rollout(
    model: CausalLatentDiT,
    context: torch.Tensor,
    actions: torch.Tensor,
    *,
    block_index: int,
    direction: torch.Tensor | None,
    amplitude: float,
    integration_steps: int,
    seed: int,
    record_activations: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    calls = 0

    def intervene(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal calls
        if record_activations is not None and calls % integration_steps == integration_steps - 1:
            record_activations.append(output[:, -1].detach())
        if direction is None or amplitude == 0.0:
            calls += 1
            return output
        edited = output.clone()
        if direction.ndim == 4:
            step = min(calls // integration_steps, direction.shape[1] - 1)
            current_direction = direction[:, step]
        else:
            current_direction = direction[None]
        edited[:, -1] = edited[:, -1] + amplitude * current_direction
        calls += 1
        return edited

    hook = model.blocks[block_index].register_forward_hook(intervene)
    try:
        return FlowMatchingSchedule().sample_autoregressive(
            model,
            context,
            actions,
            rollout_frames=actions.shape[1],
            integration_steps=integration_steps,
            generator=torch.Generator(device=context.device).manual_seed(seed),
        )
    finally:
        hook.remove()


@torch.no_grad()
def _puck_trajectory(
    codec: torch.nn.Module,
    history: torch.Tensor,
    generated: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    rollout_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    decoded = codec.decode(unnormalize_latents(torch.cat((history, generated), dim=1), mean, std))
    position, _ = _soft_centroid(decoded[:, -(rollout_frames + 1) :].float(), "puck")
    displacement = (position[:, 1:] - position[:, :-1]) * codec.config.image_size
    return position * codec.config.image_size, displacement[..., 0], torch.linalg.vector_norm(displacement, dim=-1)


@torch.no_grad()
def run_downstream_averaged_velocity_intervention(
    model: CausalLatentDiT,
    codec: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    ridge: RidgeFit,
    *,
    block_index: int,
    dataset_seed: int,
    dataset_offset: int,
    lens_samples: int = 48,
    intervention_samples: int = 32,
    batch_size: int = 8,
    rollout_frames: int = 12,
    strength: float = 2.0,
    integration_steps: int = 6,
) -> dict[str, Any]:
    with torch.enable_grad():
        jacobian_direction, projection_sigma, lens_protocol = _fit_downstream_velocity_lens(
            model,
            codec,
            mean,
            std,
            block_index=block_index,
            dataset_seed=dataset_seed,
            dataset_offset=dataset_offset,
            samples=lens_samples,
            batch_size=min(batch_size, lens_samples),
            integration_steps=integration_steps,
            target="puck_vx",
        )
        speed_direction, speed_projection_sigma, speed_lens_protocol = _fit_downstream_velocity_lens(
            model,
            codec,
            mean,
            std,
            block_index=block_index,
            dataset_seed=dataset_seed + 101,
            dataset_offset=dataset_offset + lens_samples,
            samples=lens_samples,
            batch_size=min(batch_size, lens_samples),
            integration_steps=integration_steps,
            target="puck_speed",
        )

    locator, locator_protocol = fit_activation_puck_locator(
        model,
        codec,
        mean,
        std,
        block_index=block_index,
        dataset_seed=dataset_seed + 202,
        dataset_offset=dataset_offset + 2 * lens_samples,
        samples=256,
        batch_size=32,
    )

    device = next(model.parameters()).device
    probe_direction = ridge.raw_unit_direction(CONTINUOUS_TARGETS.index("puck_vx")).reshape_as(
        jacobian_direction
    ).to(device)
    probe_direction = probe_direction / probe_direction.norm().clamp_min(1e-12)
    random_generator = torch.Generator(device=device).manual_seed(dataset_seed + 7_777_019)
    random_direction = torch.randn(
        jacobian_direction.shape,
        device=device,
        generator=random_generator,
    )
    random_direction -= (random_direction * jacobian_direction).sum() * jacobian_direction
    random_direction = random_direction / random_direction.norm().clamp_min(1e-12)
    speed_random_direction = torch.randn(
        speed_direction.shape,
        device=device,
        generator=random_generator,
    )
    speed_random_direction -= (speed_random_direction * speed_direction).sum() * speed_direction
    speed_random_direction = speed_random_direction / speed_random_direction.norm().clamp_min(1e-12)
    cosine = float((jacobian_direction * probe_direction).sum())

    temporal = model.config.temporal_downsample
    context_frames = (model.config.max_sequence_latents - 1) * temporal
    dataset = ClipDataset(
        intervention_samples,
        seed=dataset_seed + (dataset_offset + 2 * lens_samples) * 9_973,
        context_frames=context_frames,
        future_frames=rollout_frames,
        image_size=codec.config.image_size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    conditions = {
        "baseline": (None, 0.0),
        "jacobianPlus": (jacobian_direction, strength * projection_sigma),
        "jacobianMinus": (jacobian_direction, -strength * projection_sigma),
        "oraclePlus": (jacobian_direction, strength * projection_sigma),
        "oracleMinus": (jacobian_direction, -strength * projection_sigma),
        "probePlus": (probe_direction, strength * projection_sigma),
        "probeMinus": (probe_direction, -strength * projection_sigma),
        "randomPlus": (random_direction, strength * projection_sigma),
        "randomMinus": (random_direction, -strength * projection_sigma),
        "speedPlus": (speed_direction, strength * speed_projection_sigma),
        "speedMinus": (speed_direction, -strength * speed_projection_sigma),
        "speedRandomPlus": (speed_random_direction, strength * speed_projection_sigma),
        "speedRandomMinus": (speed_random_direction, -strength * speed_projection_sigma),
    }
    trajectories: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"position": [], "vx": [], "speed": []} for name in conditions
    }
    cursor = 0
    grid = model.config.latent_grid_size
    center = (grid // 2, grid // 2)

    def follow_puck(template: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        directions = []
        latent_steps = rollout_frames // temporal
        for sample_index in range(positions.shape[0]):
            sample_directions = []
            for step in range(latent_steps):
                cell = (positions[sample_index, step * temporal] / codec.config.image_size * grid)
                cell = cell.long().clamp(0, grid - 1)
                mapped = template.reshape(grid, grid, model.config.hidden_size)
                mapped = torch.roll(
                    mapped,
                    shifts=(int(cell[1]) - center[1], int(cell[0]) - center[0]),
                    dims=(0, 1),
                )
                sample_directions.append(mapped.reshape(grid * grid, model.config.hidden_size))
            directions.append(torch.stack(sample_directions))
        return torch.stack(directions)

    def follow_cells(template: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
        directions = []
        for sample_index in range(cells.shape[0]):
            sample_directions = []
            for step in range(cells.shape[1]):
                cell = cells[sample_index, step]
                mapped = template.reshape(grid, grid, model.config.hidden_size)
                mapped = torch.roll(
                    mapped,
                    shifts=(int(cell.div(grid, rounding_mode="floor")) - center[1], int(cell.remainder(grid)) - center[0]),
                    dims=(0, 1),
                )
                sample_directions.append(mapped.reshape(grid * grid, model.config.hidden_size))
            directions.append(torch.stack(sample_directions))
        return torch.stack(directions)

    activation_path_agreements: list[torch.Tensor] = []

    for batch in loader:
        video = _video(batch, device)
        raw_latents, _ = codec.encode(video[:, :context_frames])
        history = normalize_latents(raw_latents.float(), mean, std)
        actions = batch["actions"].to(device)
        base_seed = dataset_seed + 9_000_037 + cursor
        recorded_activations: list[torch.Tensor] = []
        baseline_generated = _persistent_rollout(
            model,
            history,
            actions,
            block_index=block_index,
            direction=None,
            amplitude=0.0,
            integration_steps=integration_steps,
            seed=base_seed,
            record_activations=recorded_activations,
        )
        baseline_position, baseline_vx, baseline_speed = _puck_trajectory(
            codec,
            history,
            baseline_generated,
            mean,
            std,
            rollout_frames,
        )
        trajectories["baseline"]["position"].append(baseline_position.cpu())
        trajectories["baseline"]["vx"].append(baseline_vx.cpu())
        trajectories["baseline"]["speed"].append(baseline_speed.cpu())
        activation_cells = torch.stack(
            [locator.cells(activation.to(device)) for activation in recorded_activations],
            dim=1,
        )
        decoded_xy = baseline_position[:, temporal::temporal] / codec.config.image_size
        decoded_cells_xy = (decoded_xy * grid).long().clamp(0, grid - 1)
        decoded_cells = decoded_cells_xy[..., 1] * grid + decoded_cells_xy[..., 0]
        activation_path_agreements.append((activation_cells == decoded_cells).float().cpu())

        for name, (direction, amplitude) in conditions.items():
            if name == "baseline":
                continue
            tracked_direction = (
                follow_puck(direction, baseline_position)
                if name.startswith("oracle")
                else follow_cells(direction, activation_cells)
            )
            generated = _persistent_rollout(
                model,
                history,
                actions,
                block_index=block_index,
                direction=tracked_direction,
                amplitude=amplitude,
                integration_steps=integration_steps,
                seed=base_seed,
            )
            position, vx, speed = _puck_trajectory(
                codec,
                history,
                generated,
                mean,
                std,
                rollout_frames,
            )
            trajectories[name]["position"].append(position.cpu())
            trajectories[name]["vx"].append(vx.cpu())
            trajectories[name]["speed"].append(speed.cpu())
        cursor += history.shape[0]

    merged = {
        condition: {metric: torch.cat(values) for metric, values in metrics.items()}
        for condition, metrics in trajectories.items()
    }

    def curve(condition: str, metric: str) -> list[float]:
        return [_round(value) for value in merged[condition][metric].mean(dim=0)]

    def comparison(prefix: str) -> dict[str, Any]:
        plus = merged[f"{prefix}Plus"]
        minus = merged[f"{prefix}Minus"]
        delta_vx = plus["vx"].mean(dim=1) - minus["vx"].mean(dim=1)
        delta_speed = plus["speed"].mean(dim=1) - minus["speed"].mean(dim=1)
        final_x = plus["position"][:, -1, 0] - minus["position"][:, -1, 0]
        return {
            "meanVxDeltaPxPerFrame": _round(delta_vx.mean()),
            "meanSpeedDeltaPxPerFrame": _round(delta_speed.mean()),
            "finalXDeltaPx": _round(final_x.mean()),
            "expectedSignRate": _round((delta_vx > 0).float().mean()),
        }

    def speed_comparison(prefix: str) -> dict[str, Any]:
        plus = merged[f"{prefix}Plus"]
        minus = merged[f"{prefix}Minus"]
        delta_speed = plus["speed"].mean(dim=1) - minus["speed"].mean(dim=1)
        return {
            "meanSpeedDeltaPxPerFrame": _round(delta_speed.mean()),
            "plusVsBaselinePxPerFrame": _round(
                (plus["speed"].mean(dim=1) - merged["baseline"]["speed"].mean(dim=1)).mean()
            ),
            "expectedSignRate": _round((delta_speed > 0).float().mean()),
        }

    return {
        "lens": lens_protocol,
        "speedLens": speed_lens_protocol,
        "activationLocator": {
            **locator_protocol,
            "rolloutCellAgreementWithDecodedPosition": _round(torch.cat(activation_path_agreements).mean()),
            "usedForCausalWrite": True,
        },
        "write": {
            "layer": f"block-{block_index + 1}",
            "rolloutFrames": rollout_frames,
            "samples": intervention_samples,
            "strengthProjectionSigmas": strength,
            "persistence": "every denoising evaluation of every autoregressive two-frame step",
            "matchedNoise": True,
            "probeJacobianCosine": _round(cosine),
            "velocitySpeedLensCosine": _round((jacobian_direction * speed_direction).sum()),
        },
        "effects": {
            "downstreamJacobian": comparison("jacobian"),
            "coordinateOracleCeiling": comparison("oracle"),
            "linearProbe": comparison("probe"),
            "randomDirection": comparison("random"),
            "downstreamSpeed": speed_comparison("speed"),
            "randomSpeedDirection": speed_comparison("speedRandom"),
        },
        "curves": {
            "baseline": {"vx": curve("baseline", "vx"), "speed": curve("baseline", "speed")},
            "jacobianPlus": {"vx": curve("jacobianPlus", "vx"), "speed": curve("jacobianPlus", "speed")},
            "jacobianMinus": {"vx": curve("jacobianMinus", "vx"), "speed": curve("jacobianMinus", "speed")},
            "probePlus": {"vx": curve("probePlus", "vx"), "speed": curve("probePlus", "speed")},
            "probeMinus": {"vx": curve("probeMinus", "vx"), "speed": curve("probeMinus", "speed")},
            "randomPlus": {"vx": curve("randomPlus", "vx"), "speed": curve("randomPlus", "speed")},
            "randomMinus": {"vx": curve("randomMinus", "vx"), "speed": curve("randomMinus", "speed")},
            "speedPlus": {"vx": curve("speedPlus", "vx"), "speed": curve("speedPlus", "speed")},
            "speedMinus": {"vx": curve("speedMinus", "vx"), "speed": curve("speedMinus", "speed")},
            "speedRandomPlus": {"vx": curve("speedRandomPlus", "vx"), "speed": curve("speedRandomPlus", "speed")},
            "speedRandomMinus": {"vx": curve("speedRandomMinus", "vx"), "speed": curve("speedRandomMinus", "speed")},
        },
    }


def run_latent_interpretability(
    checkpoint_path: Path,
    output_path: Path,
    *,
    samples: int = 768,
    batch_size: int = 32,
    seed: int = 710_003,
    intervention_samples: int = 32,
) -> dict[str, Any]:
    if samples < 120:
        raise ValueError("Interpretability requires at least 120 independent trajectories")
    (
        features,
        continuous,
        binary,
        model,
        codec,
        mean,
        std,
        checkpoint,
    ) = collect_latent_features(
        checkpoint_path,
        samples=samples,
        batch_size=batch_size,
        seed=seed,
    )
    device = next(model.parameters()).device
    train_count = int(samples * 0.6)
    validation_count = int(samples * 0.2)
    test_count = samples - train_count - validation_count
    visible_names = ["codec", "input", *[f"block-{index + 1}" for index in range(model.config.depth)]]
    all_names = [*visible_names, "untrained", "action-only"]
    ridge_fits: dict[str, RidgeFit] = {}
    representations: dict[str, Any] = {}
    for name in all_names:
        ridge = fit_ridge_probe(
            features[name],
            continuous,
            train_count=train_count,
            validation_count=validation_count,
            device=device,
        )
        classifier = fit_linear_classifier(
            features[name],
            binary,
            train_count=train_count,
            validation_count=validation_count,
            device=device,
        )
        ridge_fits[name] = ridge
        representations[name] = _representation_payload(ridge, classifier)

    block_names = [name for name in visible_names if name.startswith("block-")]
    best_velocity = max(block_names, key=lambda name: representations[name]["taskScores"]["velocity"])
    best_collision = max(block_names, key=lambda name: representations[name]["taskScores"]["collision"])

    permutation = torch.randperm(samples, generator=torch.Generator().manual_seed(seed + 91))
    shuffled_ridge = fit_ridge_probe(
        features[best_velocity],
        continuous[permutation],
        train_count=train_count,
        validation_count=validation_count,
        device=device,
    )
    shuffled_classifier = fit_linear_classifier(
        features[best_collision],
        binary[permutation],
        train_count=train_count,
        validation_count=validation_count,
        device=device,
    )
    shuffled_payload = _representation_payload(shuffled_ridge, shuffled_classifier)

    intervention = run_velocity_interventions(
        model,
        codec,
        mean,
        std,
        ridge_fits[best_velocity],
        block_index=int(best_velocity.split("-")[1]) - 1,
        dataset_seed=seed,
        dataset_offset=samples,
        samples=intervention_samples,
        batch_size=min(batch_size, intervention_samples),
    )
    downstream_intervention = run_downstream_averaged_velocity_intervention(
        model,
        codec,
        mean,
        std,
        ridge_fits[best_velocity],
        block_index=int(best_velocity.split("-")[1]) - 1,
        dataset_seed=seed,
        dataset_offset=samples + 10_000,
        intervention_samples=intervention_samples,
        batch_size=min(8, intervention_samples),
    )

    labels = {
        "codec": "Codec latent",
        "input": "DiT input",
        **{f"block-{index + 1}": f"Block {index + 1}" for index in range(model.config.depth)},
    }
    result = {
        "version": 1,
        "checkpointStep": int(checkpoint.get("step", 0)),
        "modelParameters": sum(parameter.numel() for parameter in model.parameters()),
        "protocol": {
            "samples": samples,
            "trainTrajectories": train_count,
            "validationTrajectories": validation_count,
            "testTrajectories": test_count,
            "seed": seed,
            "historyFrames": (model.config.max_sequence_latents - 1) * model.config.temporal_downsample,
            "predictionFrames": model.config.temporal_downsample,
            "probePoint": "first denoising evaluation at flow time 0",
            "split": "disjoint deterministic simulator trajectories",
        },
        "representations": [
            {"id": name, "label": labels[name], **representations[name]}
            for name in visible_names
        ],
        "controls": {
            "untrained": representations["untrained"],
            "actionOnly": representations["action-only"],
            "shuffledLabels": shuffled_payload,
        },
        "bestLayers": {
            "velocity": best_velocity,
            "collision": best_collision,
        },
        "intervention": intervention,
        "downstreamIntervention": downstream_intervention,
        "targetDefinitions": {
            "position": "End-of-next-pair player and puck x/y",
            "velocity": "End-of-next-pair player and puck vx/vy",
            "speed": "End-of-next-pair player and puck speed",
            "polar": "Player-to-puck distance and sin/cos bearing",
            "collision": "Disc impact or wall contact anywhere in the next two frames",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe and intervene on a latent Blocket League world model")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("latent-interpretability.json"))
    parser.add_argument("--samples", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=710_003)
    parser.add_argument("--intervention-samples", type=int, default=32)
    args = parser.parse_args()
    print(
        json.dumps(
            run_latent_interpretability(
                args.checkpoint,
                args.output,
                samples=args.samples,
                batch_size=args.batch_size,
                seed=args.seed,
                intervention_samples=args.intervention_samples,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
