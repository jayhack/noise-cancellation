from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ClipDataset
from .direct_model import DirectLatentTransformer, build_direct_pipeline_from_checkpoint
from .latent_model import normalize_latents, unnormalize_latents
from .latent_probe import (
    BINARY_TARGETS,
    CONTINUOUS_TARGETS,
    TASK_TARGETS,
    PuckLocator,
    RidgeFit,
    _representation_payload,
    _round,
    derive_probe_targets,
    fit_linear_classifier,
    fit_ridge_probe,
)
from .metrics import _soft_centroid
from .env import PALETTE


def _visual_puck_cells(frame: torch.Tensor, grid: int) -> torch.Tensor:
    color = torch.as_tensor(PALETTE["puck"], device=frame.device, dtype=frame.dtype).div(127.5).sub(1.0)
    distance = (frame - color[None, :, None, None]).square().sum(dim=1)
    weights = torch.exp(-distance / 0.04)
    return F.adaptive_avg_pool2d(weights[:, None], (grid, grid)).flatten(1).argmax(dim=1)


def _load(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, codec, mean, std = build_direct_pipeline_from_checkpoint(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval().requires_grad_(False)
    codec = codec.to(device).eval().requires_grad_(False)
    return model, codec, mean.to(device), std.to(device), checkpoint, device


@torch.no_grad()
def collect_direct_features(checkpoint_path: Path, *, samples: int, batch_size: int, seed: int):
    model, codec, mean, std, checkpoint, device = _load(checkpoint_path)
    temporal = model.config.temporal_downsample
    context_frames = model.config.history_latents * temporal
    dataset = ClipDataset(samples, seed=seed, context_frames=context_frames, future_frames=temporal,
                          image_size=codec.config.image_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    names = ["codec", "input", *[f"block-{i + 1}" for i in range(model.config.depth)]]
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    chunks.update({"untrained": [], "action-only": []})
    continuous_chunks: list[torch.Tensor] = []
    binary_chunks: list[torch.Tensor] = []
    random_model = DirectLatentTransformer(model.config, model.delta_std.cpu()).to(device).eval().requires_grad_(False)
    for batch in loader:
        context = batch["context"].to(device)
        raw, _ = codec.encode(context)
        history = normalize_latents(raw.float(), mean, std)
        future_actions = batch["actions"].to(device).reshape(-1, 1, temporal)
        past_actions = torch.zeros(history.shape[0], history.shape[1] - 1, temporal,
                                   device=device, dtype=torch.long)
        actions = torch.cat((past_actions, future_actions), dim=1)
        result = model(history, actions, return_hidden=True)
        assert not isinstance(result, torch.Tensor)
        _, hidden = result
        tokens = history.permute(0, 1, 3, 4, 2).reshape(
            history.shape[0], history.shape[1], model.config.latent_grid_size**2, model.config.latent_dim
        )
        action_condition = model.action_projection(model.action_embedding(actions).flatten(2))
        inputs = (model.input_projection(tokens)
                  + action_condition[:, :, None] + model.spatial_position
                  + model.temporal_position[:, :history.shape[1]])
        chunks["codec"].append(history[:, -1].flatten(1).half().cpu())
        chunks["input"].append(inputs[:, -1].flatten(1).half().cpu())
        for index, value in enumerate(hidden):
            chunks[f"block-{index + 1}"].append(value[:, -1].flatten(1).half().cpu())
        random_result = random_model(history, actions, return_hidden=True)
        assert not isinstance(random_result, torch.Tensor)
        chunks["untrained"].append(random_result[1][-1][:, -1].flatten(1).half().cpu())
        chunks["action-only"].append(F.one_hot(future_actions.flatten(1),
                                               num_classes=model.config.action_count).flatten(1).float().cpu())
        continuous, binary = derive_probe_targets(batch["state"], batch["events"])
        continuous_chunks.append(continuous)
        binary_chunks.append(binary)
    return ({name: torch.cat(value).float() for name, value in chunks.items()},
            torch.cat(continuous_chunks), torch.cat(binary_chunks), model, codec, mean, std, checkpoint)


@torch.no_grad()
def fit_activation_puck_locator(model, codec, mean, std, *, block_index: int, seed: int,
                                samples: int = 320, batch_size: int = 32):
    device = next(model.parameters()).device
    temporal, grid = model.config.temporal_downsample, model.config.latent_grid_size
    dataset = ClipDataset(samples, seed=seed, context_frames=model.config.history_latents * temporal,
                          future_frames=temporal, image_size=codec.config.image_size)
    activations, visual_cells, simulator_cells = [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0):
        context = batch["context"].to(device)
        raw, _ = codec.encode(context)
        history = normalize_latents(raw.float(), mean, std)
        future = batch["actions"].to(device).reshape(-1, 1, temporal)
        actions = torch.cat((torch.zeros(history.shape[0], history.shape[1] - 1, temporal,
                                         device=device, dtype=torch.long), future), dim=1)
        result = model(history, actions, return_hidden=True)
        assert not isinstance(result, torch.Tensor)
        activations.append(result[1][block_index][:, -1].cpu())
        visual_cells.append(_visual_puck_cells(context[:, -1], grid).cpu())
        xy = (batch["state"][:, 0, 4:6] * grid).long().clamp(0, grid - 1)
        simulator_cells.append((xy[:, 1] * grid + xy[:, 0]).cpu())
    activations = torch.cat(activations).to(device)
    visual_cells = torch.cat(visual_cells).to(device)
    simulator_cells = torch.cat(simulator_cells).to(device)
    fit_end = int(samples * 0.8)
    mean_activation = activations[:fit_end].flatten(0, 1).mean(0)
    scale = activations[:fit_end].flatten(0, 1).std(0).clamp_min(1e-5)
    x = ((activations[:fit_end] - mean_activation) / scale).flatten(0, 1)
    y = F.one_hot(visual_cells[:fit_end], grid * grid).float().flatten()
    weights = torch.where(y > 0, torch.full_like(y, grid * grid - 1), torch.ones_like(y))
    design = torch.cat((x, torch.ones(x.shape[0], 1, device=device)), dim=1)
    design = design * weights.sqrt()[:, None]
    target = y[:, None] * weights.sqrt()[:, None]
    identity = torch.eye(design.shape[1], device=device)
    solution = torch.linalg.solve(design.T @ design + identity, design.T @ target).flatten()
    locator = PuckLocator(solution[:-1], solution[-1], mean_activation, scale)
    predicted = locator.cells(activations[fit_end:])
    visual_test, simulator_test = visual_cells[fit_end:], simulator_cells[fit_end:]
    rows = predicted.div(grid, rounding_mode="floor") - visual_test.div(grid, rounding_mode="floor")
    cols = predicted.remainder(grid) - visual_test.remainder(grid)
    return locator, {
        "trainingSignal": "puck-colored pixels in observed RGB only; no simulator coordinates",
        "deploymentSignal": "block activations only", "layer": f"block-{block_index + 1}",
        "samples": samples, "trainTrajectories": int(samples * .6),
        "validationTrajectories": int(samples * .2), "testTrajectories": samples - fit_end,
        "visualCellAccuracy": _round((predicted == visual_test).float().mean()),
        "simulatorCellAccuracyEvaluationOnly": _round((predicted == simulator_test).float().mean()),
        "withinOneCellAccuracy": _round(((rows.abs() <= 1) & (cols.abs() <= 1)).float().mean()),
    }


def _fit_downstream_direction(model, codec, mean, std, *, block_index: int, seed: int,
                              samples: int, batch_size: int, target: str):
    device = next(model.parameters()).device
    temporal, grid = model.config.temporal_downsample, model.config.latent_grid_size
    dataset = ClipDataset(samples, seed=seed, context_frames=model.config.history_latents * temporal,
                          future_frames=temporal, image_size=codec.config.image_size)
    gradient_sum = torch.zeros(grid, grid, model.config.hidden_size, device=device)
    activation_chunks, count = [], 0
    center = (grid // 2, grid // 2)
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0):
        context = batch["context"].to(device)
        with torch.no_grad():
            raw, _ = codec.encode(context)
            history = normalize_latents(raw.float(), mean, std)
        future = batch["actions"].to(device).reshape(-1, 1, temporal)
        actions = torch.cat((torch.zeros(history.shape[0], history.shape[1] - 1, temporal,
                                         device=device, dtype=torch.long), future), dim=1)
        visual_cells = _visual_puck_cells(context[:, -1], grid)
        captured: list[torch.Tensor] = []
        def capture(_module, _inputs, output):
            leaf = output.detach().requires_grad_(True)
            captured.append(leaf)
            return leaf
        hook = model.blocks[block_index].register_forward_hook(capture)
        try:
            with torch.enable_grad():
                delta = model(history, actions)
                assert isinstance(delta, torch.Tensor)
                next_latent = history[:, -1] + delta[:, -1] * model.delta_std[None, :, None, None]
                decoded, _ = codec.decode_soft(unnormalize_latents(next_latent[:, None], mean, std))
                position, _ = _soft_centroid(decoded.float(), "puck")
                previous_position, _ = _soft_centroid(context[:, -1:].float(), "puck")
                displacement = (position[:, -1] - previous_position[:, -1]) * codec.config.image_size
                objective = displacement[:, 0].mean() if target == "puck_vx" else torch.linalg.vector_norm(displacement, dim=1).mean()
                gradient = torch.autograd.grad(objective, captured[0])[0][:, -1]
        finally:
            hook.remove()
        for activation, grad, cell in zip(captured[0][:, -1].detach(), gradient, visual_cells):
            row, col = int(cell // grid), int(cell % grid)
            shifts = (center[0] - row, center[1] - col)
            gradient_sum += torch.roll(grad.reshape(grid, grid, -1), shifts=shifts, dims=(0, 1))
            activation_chunks.append(torch.roll(activation.reshape(grid, grid, -1), shifts=shifts, dims=(0, 1)).flatten())
            count += 1
    average = gradient_sum / max(count, 1)
    direction = average / average.norm().clamp_min(1e-12)
    projections = torch.stack(activation_chunks) @ direction.flatten()
    sigma = float(projections.std().clamp_min(1e-4))
    return direction.detach(), sigma, {
        "contexts": samples, "sourceLayer": f"block-{block_index + 1}",
        "downstreamTarget": "decoded puck x displacement in the next two frames" if target == "puck_vx" else "decoded puck speed in the next two frames",
        "averagedOver": "independent contexts and spatially aligned puck neighborhoods",
        "spatialAlignment": "visual puck location centers each gradient before averaging",
        "solverEvaluationsPerContext": 1, "gradientL2": _round(average.norm()),
        "projectionSigma": _round(sigma),
    }


@torch.no_grad()
def _rollout(model, context, actions, *, block_index: int, direction: torch.Tensor | None,
             amplitude: float, cells: torch.Tensor | None = None, record: list[torch.Tensor] | None = None):
    temporal, grid = model.config.temporal_downsample, model.config.latent_grid_size
    pairs = actions.reshape(actions.shape[0], -1, temporal)
    history = context
    history_actions = torch.zeros(context.shape[0], context.shape[1] - 1, temporal,
                                  device=context.device, dtype=actions.dtype)
    generated = []
    step = 0
    def edit(_module, _inputs, output):
        nonlocal step
        if record is not None:
            record.append(output[:, -1].detach())
        if direction is None or cells is None:
            step += 1
            return output
        edited = output.clone()
        template = direction.reshape(grid, grid, -1)
        center = (grid // 2, grid // 2)
        for item in range(output.shape[0]):
            cell = cells[item, min(step, cells.shape[1] - 1)]
            row, col = int(cell // grid), int(cell % grid)
            mapped = torch.roll(template, shifts=(row - center[0], col - center[1]), dims=(0, 1))
            edited[item, -1] += amplitude * mapped.flatten(0, 1)
        step += 1
        return edited
    hook = model.blocks[block_index].register_forward_hook(edit)
    try:
        for pair in pairs.unbind(1):
            current_history = history[:, -model.config.history_latents:]
            current_actions = torch.cat((history_actions, pair[:, None]), dim=1)[:, -current_history.shape[1]:]
            next_latent = model.next_latent(current_history, current_actions).clamp(-8, 8)
            generated.append(next_latent)
            history = torch.cat((history, next_latent[:, None]), dim=1)
            history_actions = torch.cat((history_actions, pair[:, None]), dim=1)
    finally:
        hook.remove()
    return torch.stack(generated, dim=1)


@torch.no_grad()
def _trajectory(codec, history, generated, mean, std, rollout_frames):
    decoded = codec.decode(unnormalize_latents(torch.cat((history[:, -1:], generated), dim=1), mean, std))
    position, _ = _soft_centroid(decoded[:, -(rollout_frames + 1):].float(), "puck")
    displacement = (position[:, 1:] - position[:, :-1]) * codec.config.image_size
    return position * codec.config.image_size, displacement[..., 0], torch.linalg.vector_norm(displacement, dim=-1)


def run_causal_study(model, codec, mean, std, ridge: RidgeFit, *, block_index: int, seed: int,
                     intervention_samples: int, batch_size: int, rollout_frames: int = 12,
                     lens_samples: int = 64, strength: float = 2.0):
    with torch.enable_grad():
        velocity_direction, velocity_sigma, velocity_protocol = _fit_downstream_direction(
            model, codec, mean, std, block_index=block_index, seed=seed + 1001,
            samples=lens_samples, batch_size=min(batch_size, lens_samples), target="puck_vx")
        speed_direction, speed_sigma, speed_protocol = _fit_downstream_direction(
            model, codec, mean, std, block_index=block_index, seed=seed + 2002,
            samples=lens_samples, batch_size=min(batch_size, lens_samples), target="puck_speed")
    locator, locator_protocol = fit_activation_puck_locator(
        model, codec, mean, std, block_index=block_index, seed=seed + 3003)
    device = next(model.parameters()).device
    probe_direction = ridge.raw_unit_direction(CONTINUOUS_TARGETS.index("puck_vx")).reshape_as(velocity_direction).to(device)
    probe_direction /= probe_direction.norm().clamp_min(1e-12)
    generator = torch.Generator(device=device).manual_seed(seed + 4004)
    random_direction = torch.randn(velocity_direction.shape, generator=generator, device=device)
    random_direction -= (random_direction * velocity_direction).sum() * velocity_direction
    random_direction /= random_direction.norm().clamp_min(1e-12)
    speed_random = torch.randn(speed_direction.shape, generator=generator, device=device)
    speed_random -= (speed_random * speed_direction).sum() * speed_direction
    speed_random /= speed_random.norm().clamp_min(1e-12)
    temporal = model.config.temporal_downsample
    dataset = ClipDataset(intervention_samples, seed=seed + 5005,
                          context_frames=model.config.history_latents * temporal,
                          future_frames=rollout_frames, image_size=codec.config.image_size)
    conditions = {
        "baseline": (None, 0.0), "jacobianPlus": (velocity_direction, strength * velocity_sigma),
        "jacobianMinus": (velocity_direction, -strength * velocity_sigma),
        "oraclePlus": (velocity_direction, strength * velocity_sigma),
        "oracleMinus": (velocity_direction, -strength * velocity_sigma),
        "probePlus": (probe_direction, strength * velocity_sigma), "probeMinus": (probe_direction, -strength * velocity_sigma),
        "randomPlus": (random_direction, strength * velocity_sigma), "randomMinus": (random_direction, -strength * velocity_sigma),
        "speedPlus": (speed_direction, strength * speed_sigma), "speedMinus": (speed_direction, -strength * speed_sigma),
        "speedRandomPlus": (speed_random, strength * speed_sigma), "speedRandomMinus": (speed_random, -strength * speed_sigma),
    }
    trajectories = {name: {metric: [] for metric in ("position", "vx", "speed")} for name in conditions}
    agreements = []
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0):
        context_video = batch["context"].to(device)
        raw, _ = codec.encode(context_video)
        history = normalize_latents(raw.float(), mean, std)
        actions = batch["actions"].to(device)
        recorded: list[torch.Tensor] = []
        baseline = _rollout(model, history, actions, block_index=block_index, direction=None,
                            amplitude=0, record=recorded)
        position, vx, speed = _trajectory(codec, history, baseline, mean, std, rollout_frames)
        for metric, value in zip(("position", "vx", "speed"), (position, vx, speed)):
            trajectories["baseline"][metric].append(value.cpu())
        cells = torch.stack([locator.cells(value.to(device)) for value in recorded], dim=1)
        grid = model.config.latent_grid_size
        decoded_xy = position[:, temporal::temporal] / codec.config.image_size
        decoded_cell_xy = (decoded_xy * grid).long().clamp(0, grid - 1)
        decoded_cells = decoded_cell_xy[..., 1] * grid + decoded_cell_xy[..., 0]
        agreements.append((cells == decoded_cells).float().cpu())
        for name, (direction, amplitude) in conditions.items():
            if name == "baseline": continue
            write_cells = decoded_cells if name.startswith("oracle") else cells
            generated = _rollout(model, history, actions, block_index=block_index, direction=direction,
                                 amplitude=amplitude, cells=write_cells)
            values = _trajectory(codec, history, generated, mean, std, rollout_frames)
            for metric, value in zip(("position", "vx", "speed"), values):
                trajectories[name][metric].append(value.cpu())
    merged = {name: {metric: torch.cat(values) for metric, values in payload.items()}
              for name, payload in trajectories.items()}
    def curve(name, metric): return [_round(v) for v in merged[name][metric].mean(0)]
    def comparison(prefix):
        plus, minus = merged[prefix + "Plus"], merged[prefix + "Minus"]
        dvx = plus["vx"].mean(1) - minus["vx"].mean(1)
        return {"meanVxDeltaPxPerFrame": _round(dvx.mean()),
                "meanSpeedDeltaPxPerFrame": _round((plus["speed"].mean(1) - minus["speed"].mean(1)).mean()),
                "finalXDeltaPx": _round((plus["position"][:, -1, 0] - minus["position"][:, -1, 0]).mean()),
                "expectedSignRate": _round((dvx > 0).float().mean())}
    def speed_comparison(prefix):
        plus, minus = merged[prefix + "Plus"], merged[prefix + "Minus"]
        delta = plus["speed"].mean(1) - minus["speed"].mean(1)
        return {"meanSpeedDeltaPxPerFrame": _round(delta.mean()),
                "plusVsBaselinePxPerFrame": _round((plus["speed"].mean(1) - merged["baseline"]["speed"].mean(1)).mean()),
                "expectedSignRate": _round((delta > 0).float().mean())}
    locator_protocol["rolloutCellAgreementWithDecodedPosition"] = _round(torch.cat(agreements).mean())
    locator_protocol["usedForCausalWrite"] = True
    curves = {name: {"vx": curve(name, "vx"), "speed": curve(name, "speed")} for name in conditions}
    return {"lens": velocity_protocol, "speedLens": speed_protocol, "activationLocator": locator_protocol,
            "write": {"layer": f"block-{block_index + 1}", "rolloutFrames": rollout_frames,
                      "samples": intervention_samples, "strengthProjectionSigmas": strength,
                      "persistence": "every autoregressive two-frame transition", "matchedNoise": True,
                      "probeJacobianCosine": _round((velocity_direction * probe_direction).sum()),
                      "velocitySpeedLensCosine": _round((velocity_direction * speed_direction).sum())},
            "effects": {"downstreamJacobian": comparison("jacobian"), "coordinateOracleCeiling": comparison("oracle"),
                        "linearProbe": comparison("probe"), "randomDirection": comparison("random"),
                        "downstreamSpeed": speed_comparison("speed"), "randomSpeedDirection": speed_comparison("speedRandom")},
            "curves": curves}


def run_direct_interpretability(checkpoint_path: Path, output_path: Path, *, samples: int = 768,
                                batch_size: int = 32, seed: int = 810_003,
                                intervention_samples: int = 256) -> dict[str, Any]:
    features, continuous, binary, model, codec, mean, std, checkpoint = collect_direct_features(
        checkpoint_path, samples=samples, batch_size=batch_size, seed=seed)
    device = next(model.parameters()).device
    train_count, validation_count = int(samples * .6), int(samples * .2)
    visible = ["codec", "input", *[f"block-{i + 1}" for i in range(model.config.depth)]]
    representations, ridges = {}, {}
    for name in [*visible, "untrained", "action-only"]:
        ridge = fit_ridge_probe(features[name], continuous, train_count=train_count,
                                validation_count=validation_count, device=device)
        classifier = fit_linear_classifier(features[name], binary, train_count=train_count,
                                           validation_count=validation_count, device=device)
        ridges[name] = ridge
        representations[name] = _representation_payload(ridge, classifier)
    blocks = [name for name in visible if name.startswith("block-")]
    best_velocity = max(blocks, key=lambda name: representations[name]["taskScores"]["velocity"])
    best_collision = max(blocks, key=lambda name: representations[name]["taskScores"]["collision"])
    permutation = torch.randperm(samples, generator=torch.Generator().manual_seed(seed + 91))
    shuffled = _representation_payload(
        fit_ridge_probe(features[best_velocity], continuous[permutation], train_count=train_count,
                        validation_count=validation_count, device=device),
        fit_linear_classifier(features[best_collision], binary[permutation], train_count=train_count,
                              validation_count=validation_count, device=device))
    block_index = int(best_velocity.split("-")[1]) - 1
    causal = run_causal_study(model, codec, mean, std, ridges[best_velocity], block_index=block_index,
                              seed=seed + samples, intervention_samples=intervention_samples,
                              batch_size=min(batch_size, 16))
    labels = {"codec": "Codec latent", "input": "Transformer input",
              **{f"block-{i + 1}": f"Block {i + 1}" for i in range(model.config.depth)}}
    payload = {
        "version": 2, "modelKind": "direct-autoregressive", "checkpointStep": checkpoint["step"],
        "modelParameters": sum(p.numel() for p in model.parameters()),
        "protocol": {"samples": samples, "trainTrajectories": train_count,
                     "validationTrajectories": validation_count, "testTrajectories": samples - train_count - validation_count,
                     "seed": seed, "historyFrames": model.config.history_latents * model.config.temporal_downsample,
                     "predictionFrames": model.config.temporal_downsample,
                     "probePoint": "single direct transition, after each transformer block",
                     "split": "disjoint deterministic simulator trajectories"},
        "representations": [{"id": name, "label": labels[name], **representations[name]} for name in visible],
        "controls": {"untrained": representations["untrained"], "actionOnly": representations["action-only"],
                     "shuffledLabels": shuffled},
        "bestLayers": {"velocity": best_velocity, "collision": best_collision},
        "intervention": {"layer": f"block-{block_index + 1}", "samples": 0,
                         "probeStandardDeviations": 0, "integrationSteps": 1, "targets": {}},
        "downstreamIntervention": causal,
        "targetDefinitions": {"position": "x/y coordinates at the predicted pair",
                              "velocity": "signed x/y velocity at the predicted pair",
                              "polar": "distance and bearing between player and puck",
                              "collision": "disc or wall collision in the predicted pair"},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("direct-interpretability.json"))
    parser.add_argument("--samples", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--intervention-samples", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run_direct_interpretability(args.checkpoint, args.output, samples=args.samples,
                                                 batch_size=args.batch_size,
                                                 intervention_samples=args.intervention_samples), indent=2))
