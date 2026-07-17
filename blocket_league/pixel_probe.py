from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import make_passive_clip
from .pixel_direct_model import build_pixel_direct_from_checkpoint
from .train_pixel_direct import frames_to_classes, palette_tensor


PLAYER_CLASSES = (5, 6)


def _visual_centroid(classes: torch.Tensor, values: tuple[int, ...]) -> torch.Tensor:
    """Measure an entity directly from rendered categorical pixels."""

    mask = torch.zeros_like(classes, dtype=torch.float32)
    for value in values:
        mask.add_(classes.eq(value))
    height, width = classes.shape[-2:]
    x = torch.arange(width, device=classes.device, dtype=torch.float32) + 0.5
    y = torch.arange(height, device=classes.device, dtype=torch.float32) + 0.5
    mass = mask.sum(dim=(-2, -1)).clamp_min(1e-6)
    return torch.stack(
        ((mask * x).sum(dim=(-2, -1)) / mass, (mask * y[:, None]).sum(dim=(-2, -1)) / mass),
        dim=-1,
    )


def _soft_centroid(logits: torch.Tensor, values: tuple[int, ...]) -> torch.Tensor:
    probabilities = logits.float().softmax(dim=1)[:, list(values)].sum(dim=1)
    height, width = probabilities.shape[-2:]
    x = torch.arange(width, device=logits.device, dtype=torch.float32) + 0.5
    y = torch.arange(height, device=logits.device, dtype=torch.float32) + 0.5
    mass = probabilities.sum(dim=(-2, -1)).clamp_min(1e-6)
    return torch.stack(
        (
            (probabilities * x).sum(dim=(-2, -1)) / mass,
            (probabilities * y[:, None]).sum(dim=(-2, -1)) / mass,
        ),
        dim=-1,
    )


def _entity_token_mask(model, classes: torch.Tensor) -> torch.Tensor:
    position = _visual_centroid(classes[:, -1], PLAYER_CLASSES)
    patch_x = (position[:, 0] / model.config.patch_size).long().clamp(0, model.config.grid_size - 1)
    patch_y = (position[:, 1] / model.config.patch_size).long().clamp(0, model.config.grid_size - 1)
    token = patch_y * model.config.grid_size + patch_x
    mask = torch.zeros(
        classes.shape[0], classes.shape[1], model.config.grid_size**2,
        device=classes.device,
    )
    mask[torch.arange(classes.shape[0], device=classes.device), -1, token] = 1
    return mask


def _context_batch(
    seeds: list[int],
    model,
    device: torch.device,
    *,
    post_goal: bool = False,
) -> torch.Tensor:
    videos = []
    for seed in seeds:
        if post_goal:
            clip = make_passive_clip(
                seed,
                context_frames=1,
                future_frames=model.config.history_frames + 20,
                image_size=model.config.image_size,
                goal_centered=True,
            )
            kickoff = np.flatnonzero(clip["events"] == 5)
            if not len(kickoff):
                raise RuntimeError(f"Goal-centered clip {seed} did not reach kickoff")
            end = int(kickoff[0]) + 2
            start = end - model.config.history_frames
            frames = clip["frames"][start:end]
        else:
            clip = make_passive_clip(
                seed,
                context_frames=model.config.history_frames,
                future_frames=1,
                image_size=model.config.image_size,
            )
            frames = clip["context"]
        video = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
        videos.append(video.float().div(127.5).sub(1.0))
    return frames_to_classes(torch.stack(videos).to(device), palette_tensor(device))


def _fit_ridge(train_x: torch.Tensor, train_y: torch.Tensor, ridge: float = 1e-2):
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-5)
    x = (train_x - mean) / scale
    x = torch.cat((x, torch.ones(x.shape[0], 1, device=x.device)), dim=1)
    eye = torch.eye(x.shape[1], device=x.device)
    eye[-1, -1] = 0
    weight = torch.linalg.solve(x.T @ x + ridge * eye, x.T @ train_y)
    return mean, scale, weight


def _ridge_r2(fit, x: torch.Tensor, y: torch.Tensor) -> float:
    mean, scale, weight = fit
    normalized = (x - mean) / scale
    prediction = torch.cat((normalized, torch.ones(x.shape[0], 1, device=x.device)), dim=1) @ weight
    residual = (y - prediction).square().sum()
    total = (y - y.mean(0, keepdim=True)).square().sum().clamp_min(1e-8)
    return float(1 - residual / total)


@torch.no_grad()
def _probe_features(model, classes: torch.Tensor):
    _, hidden = model(classes, return_hidden=True)
    mask = _entity_token_mask(model, classes)[:, -1]
    token = mask.argmax(dim=1)
    batch = torch.arange(classes.shape[0], device=classes.device)
    features = [state[batch, -1, token].float() for state in hidden]
    positions = _visual_centroid(classes, PLAYER_CLASSES)
    target = torch.cat((positions[:, -1], positions[:, -1] - positions[:, -2]), dim=1)
    return features, target


def _downstream_directions(model, classes: torch.Tensor, block_index: int) -> torch.Tensor:
    mask = _entity_token_mask(model, classes)
    write = torch.zeros(
        classes.shape[0], model.config.hidden_size,
        device=classes.device,
        requires_grad=True,
    )
    logits = model(
        classes,
        intervention_block=block_index,
        intervention=write,
        intervention_mask=mask,
    )[:, -1]
    position = _soft_centroid(logits, PLAYER_CLASSES)
    directions = []
    for axis in range(2):
        gradient = torch.autograd.grad(position[:, axis].sum(), write, retain_graph=axis == 0)[0]
        direction = gradient.mean(dim=0)
        directions.append(direction / direction.norm().clamp_min(1e-8))
    return torch.stack(directions)


@torch.no_grad()
def _rollout(model, classes: torch.Tensor, frames: int, *, block_index: int | None = None,
             direction: torch.Tensor | None = None, strength: float = 0.0,
             write_frames: int | None = None) -> torch.Tensor:
    history = classes
    generated = []
    for step in range(frames):
        current = history[:, -model.config.history_frames:]
        if direction is None or (write_frames is not None and step >= write_frames):
            logits = model(current)[:, -1]
        else:
            logits = model(
                current,
                intervention_block=block_index,
                intervention=direction * strength,
                intervention_mask=_entity_token_mask(model, current),
            )[:, -1]
        next_frame = logits.argmax(dim=1)
        generated.append(next_frame)
        history = torch.cat((history, next_frame[:, None]), dim=1)
    return torch.stack(generated, dim=1)


def run_pixel_interpretability(
    checkpoint_path: Path,
    output_path: Path,
    *,
    fit_samples: int = 512,
    test_samples: int = 256,
    batch_size: int = 32,
    rollout_frames: int = 12,
    strength: float = 8.0,
    write_frames: int = 4,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_pixel_direct_from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    fit_seeds = [5_000_003 + index * 9_973 for index in range(fit_samples)]
    test_seeds = [9_000_007 + index * 9_973 for index in range(test_samples)]

    fit_features = [[] for _ in model.blocks]
    fit_targets = []
    for start in range(0, fit_samples, batch_size):
        classes = _context_batch(fit_seeds[start:start + batch_size], model, device)
        features, targets = _probe_features(model, classes)
        for layer, values in enumerate(features):
            fit_features[layer].append(values)
        fit_targets.append(targets)
    fit_features = [torch.cat(values) for values in fit_features]
    fit_targets_tensor = torch.cat(fit_targets)
    position_fits = [_fit_ridge(features, fit_targets_tensor[:, :2]) for features in fit_features]
    velocity_fits = [_fit_ridge(features, fit_targets_tensor[:, 2:]) for features in fit_features]

    test_features = [[] for _ in model.blocks]
    test_targets = []
    for start in range(0, test_samples, batch_size):
        classes = _context_batch(test_seeds[start:start + batch_size], model, device)
        features, targets = _probe_features(model, classes)
        for layer, values in enumerate(features):
            test_features[layer].append(values)
        test_targets.append(targets)
    test_features = [torch.cat(values) for values in test_features]
    test_targets_tensor = torch.cat(test_targets)
    probes = [
        {
            "block": layer + 1,
            "position_r2": _ridge_r2(
                position_fits[layer], test_features[layer], test_targets_tensor[:, :2]
            ),
            "velocity_r2": _ridge_r2(
                velocity_fits[layer], test_features[layer], test_targets_tensor[:, 2:]
            ),
        }
        for layer in range(len(model.blocks))
    ]
    best_velocity_r2 = max(probe["velocity_r2"] for probe in probes)
    best_block = min(
        index
        for index, probe in enumerate(probes)
        if probe["velocity_r2"] >= best_velocity_r2 - 0.01
    )

    direction_batches = []
    for start in range(0, fit_samples, batch_size):
        classes = _context_batch(fit_seeds[start:start + batch_size], model, device)
        direction_batches.append(_downstream_directions(model, classes, best_block))
    directions = torch.stack(direction_batches).mean(0)
    directions /= directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
    generator = torch.Generator(device=device).manual_seed(311)
    random_direction = torch.randn(model.config.hidden_size, generator=generator, device=device)
    random_direction /= random_direction.norm().clamp_min(1e-8)

    def measure_effects(
        *,
        post_goal: bool,
        held_write: bool = False,
    ) -> dict[str, dict[str, float | int]]:
        effects = {"x_plus": [], "x_minus": [], "y_plus": [], "y_minus": [], "random": []}
        active_write_frames = rollout_frames if held_write else write_frames
        for start in range(0, test_samples, batch_size):
            classes = _context_batch(
                test_seeds[start:start + batch_size], model, device, post_goal=post_goal,
            )
            baseline = _rollout(model, classes, rollout_frames)
            rollouts = {
                "x_plus": _rollout(model, classes, rollout_frames, block_index=best_block,
                                   direction=directions[0], strength=strength,
                                   write_frames=active_write_frames),
                "x_minus": _rollout(model, classes, rollout_frames, block_index=best_block,
                                    direction=directions[0], strength=-strength,
                                    write_frames=active_write_frames),
                "y_plus": _rollout(model, classes, rollout_frames, block_index=best_block,
                                   direction=directions[1], strength=strength,
                                   write_frames=active_write_frames),
                "y_minus": _rollout(model, classes, rollout_frames, block_index=best_block,
                                    direction=directions[1], strength=-strength,
                                    write_frames=active_write_frames),
                "random": _rollout(model, classes, rollout_frames, block_index=best_block,
                                   direction=random_direction, strength=strength,
                                   write_frames=active_write_frames),
            }
            baseline_position = _visual_centroid(baseline, PLAYER_CLASSES)
            for name, rollout in rollouts.items():
                axis = 1 if name.startswith("y_") else 0
                delta = (
                    _visual_centroid(rollout, PLAYER_CLASSES)[..., axis]
                    - baseline_position[..., axis]
                )
                release_index = min(active_write_frames - 1, rollout_frames - 1)
                effects[name].append(
                    torch.stack((delta[:, release_index], delta[:, -1]), dim=1).cpu()
                )

        summarized: dict[str, dict[str, float | int]] = {}
        for name, chunks in effects.items():
            values = torch.cat(chunks)
            expected_sign = -1 if name.endswith("minus") else 1
            axis = "y" if name.startswith("y_") else "x"
            summarized[name] = {
                f"release_{axis}_delta_px": float(values[:, 0].mean()),
                f"final_{axis}_delta_px": float(values[:, 1].mean()),
                "post_release_growth_px": float((values[:, 1] - values[:, 0]).mean()),
                "samples": int(values.shape[0]),
            }
            if name == "random":
                summarized[name]["positive_fraction"] = float((values[:, 1] > 0).float().mean())
            else:
                summarized[name]["expected_sign_fraction"] = float(
                    (values[:, 1] * expected_sign > 0).float().mean()
                )
        return summarized

    summarized = measure_effects(post_goal=False)
    post_goal_summarized = measure_effects(post_goal=True)
    post_goal_held_summarized = measure_effects(post_goal=True, held_write=True)
    result = {
        "version": 1,
        "modelKind": "passive-direct-pixel-autoregressive",
        "checkpointStep": checkpoint["step"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fitSamples": fit_samples,
        "testSamples": test_samples,
        "labelSource": "rendered pixels only",
        "actionConditioning": False,
        "causalBlockSelection": "earliest block within 0.01 R2 of the best velocity probe",
        "probes": probes,
        "causal": {
            "method": "held-out downstream-averaged gradient write",
            "block": best_block + 1,
            "strength": strength,
            "writeFrames": write_frames,
            "rolloutFrames": rollout_frames,
            "effects": summarized,
            "postGoalEffects": post_goal_summarized,
            "postGoalHeldEffects": post_goal_held_summarized,
            "xDirection": directions[0].cpu().tolist(),
            "yDirection": directions[1].cpu().tolist(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
