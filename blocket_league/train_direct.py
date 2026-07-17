from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .codec import load_codec_checkpoint
from .data import ClipDataset
from .direct_model import DirectLatentTransformer, direct_config_for_preset
from .latent_model import normalize_latents, unnormalize_latents
from .metrics import trajectory_metrics
from .train import save_rollout_comparison


@dataclass
class DirectTrainConfig:
    output_dir: str = "blocket_league/outputs/direct-local"
    codec_checkpoint_path: str = ""
    preset: str = "tiny"
    steps: int = 30_000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 47
    workers: int = 6
    log_every: int = 100
    eval_samples: int = 128
    image_size: int = 64
    history_latents: int = 8
    cache_latents: int = 16
    rollout_frames: int = 64
    cache_samples: int = 16_384
    cache_batch_size: int = 64
    corruption_std: float = 0.12
    late_token_weight: float = 2.0
    ema_decay: float = 0.9995
    warmup_steps: int = 500
    min_learning_rate_ratio: float = 0.1

    @property
    def cache_frames(self) -> int:
        return self.cache_latents * 2


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _learning_rate_multiplier(step: int, config: DirectTrainConfig) -> float:
    if step <= config.warmup_steps:
        return step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return config.min_learning_rate_ratio + (1.0 - config.min_learning_rate_ratio) * cosine


@torch.no_grad()
def _update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


@torch.no_grad()
def build_direct_cache(
    codec: torch.nn.Module,
    config: DirectTrainConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    # Two seed frames followed by action-aligned frames gives exactly one action
    # pair for each transition between codec latents.
    dataset = ClipDataset(
        config.cache_samples,
        seed=config.seed,
        context_frames=2,
        future_frames=config.cache_frames - 2,
        image_size=config.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.cache_batch_size,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0,
    )
    latent_batches: list[torch.Tensor] = []
    action_batches: list[torch.Tensor] = []
    started = time.perf_counter()
    for index, batch in enumerate(loader):
        video = torch.cat((batch["context"], batch["target"]), dim=1).to(
            device, non_blocking=True
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            raw_latents, _ = codec.encode(video)
        latent_batches.append(
            normalize_latents(raw_latents.float(), mean, std).to(torch.bfloat16).cpu()
        )
        action_batches.append(batch["actions"].to(torch.int16))
        if index == 0 or (index + 1) % 64 == 0 or index + 1 == len(loader):
            encoded = min((index + 1) * config.cache_batch_size, config.cache_samples)
            print(json.dumps({"stage": "encode_cache", "encoded": encoded, "total": config.cache_samples}), flush=True)
    latents = torch.cat(latent_batches).to(device)
    actions = torch.cat(action_batches).long().to(device)
    if actions.shape[1] != (config.cache_latents - 1) * 2:
        raise RuntimeError("Cached actions do not align with latent transitions")
    deltas = (latents[:, 1:].float() - latents[:, :-1].float())
    delta_std = deltas.square().mean(dim=(0, 1, 3, 4)).sqrt().clamp_min(1e-3)
    return latents, actions, delta_std, time.perf_counter() - started


def direct_training_loss(
    model: DirectLatentTransformer,
    clean_inputs: torch.Tensor,
    clean_targets: torch.Tensor,
    action_pairs: torch.Tensor,
    *,
    corruption_std: float,
    late_token_weight: float,
) -> torch.Tensor:
    batch, frames = clean_inputs.shape[:2]
    # Each example receives a different drift magnitude. Later states are more
    # corrupted, approximating the errors encountered during free-running play.
    severity = torch.rand(batch, 1, 1, 1, 1, device=clean_inputs.device)
    ramp = torch.linspace(0.25, 1.0, frames, device=clean_inputs.device)[None, :, None, None, None]
    corrupted = clean_inputs + torch.randn_like(clean_inputs) * severity * ramp * corruption_std
    expected = (clean_targets - corrupted) / model.delta_std[None, None, :, None, None]
    predicted = model(corrupted, action_pairs)
    if not isinstance(predicted, torch.Tensor):
        predicted = predicted[0]
    error = F.smooth_l1_loss(predicted.float(), expected.float(), reduction="none", beta=0.5)
    error = error.mean(dim=(2, 3, 4))
    weights = torch.linspace(1.0, late_token_weight, frames, device=error.device)
    return (error * weights[None]).sum() / (weights.sum() * batch)


@torch.no_grad()
def rollout_latents(
    model: DirectLatentTransformer,
    context: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    temporal = model.config.temporal_downsample
    if actions.shape[1] % temporal:
        raise ValueError("Actions must align to codec temporal pairs")
    pairs = actions.reshape(actions.shape[0], -1, temporal)
    history = context
    history_actions = torch.zeros(
        context.shape[0],
        max(context.shape[1] - 1, 0),
        temporal,
        device=actions.device,
        dtype=actions.dtype,
    )
    generated: list[torch.Tensor] = []
    for pair in pairs.unbind(dim=1):
        current_actions = torch.cat((history_actions, pair[:, None]), dim=1)
        current_history = history[:, -model.config.history_latents :]
        current_actions = current_actions[:, -current_history.shape[1] :]
        next_latent = model.next_latent(current_history, current_actions).clamp(-8.0, 8.0)
        generated.append(next_latent)
        history = torch.cat((history, next_latent[:, None]), dim=1)
        history_actions = torch.cat((history_actions, pair[:, None]), dim=1)
    return torch.stack(generated, dim=1)


@torch.no_grad()
def predict_video(
    model: DirectLatentTransformer,
    codec: torch.nn.Module,
    context_video: torch.Tensor,
    actions: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    raw_context, _ = codec.encode(context_video)
    context = normalize_latents(raw_context.float(), mean, std)
    generated = rollout_latents(model, context, actions)
    decoded = codec.decode(unnormalize_latents(generated, mean, std).to(context_video.dtype))
    return decoded[:, : actions.shape[1]]


@torch.no_grad()
def evaluate_direct_model(
    model: DirectLatentTransformer,
    codec: torch.nn.Module,
    config: DirectTrainConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    dataset = ClipDataset(
        config.eval_samples,
        seed=config.seed + 2_000_000,
        context_frames=model.config.history_latents * 2,
        future_frames=config.rollout_frames,
        image_size=config.image_size,
    )
    loader = DataLoader(dataset, batch_size=min(config.batch_size, 8), num_workers=config.workers)
    totals: dict[str, float] = {}
    evaluated = 0
    for batch in loader:
        context = batch["context"].to(device)
        target = batch["target"].to(device)
        actions = batch["actions"].to(device)
        states = batch["state"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = predict_video(model, codec, context, actions, mean, std)
        metrics = trajectory_metrics(prediction, states)
        metrics["pixel_mse"] = float((prediction.float() - target.float()).square().mean())
        count = context.shape[0]
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value * count
        evaluated += count
    return {**{name: value / evaluated for name, value in totals.items()}, "samples": float(evaluated)}


def train_direct(config: DirectTrainConfig) -> dict[str, Any]:
    if not config.codec_checkpoint_path:
        raise ValueError("codec_checkpoint_path is required")
    if config.cache_latents <= config.history_latents:
        raise ValueError("cache_latents must exceed history_latents")
    if config.rollout_frames % 2:
        raise ValueError("rollout_frames must be even")
    _seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    codec, codec_checkpoint = load_codec_checkpoint(Path(config.codec_checkpoint_path).resolve())
    codec = codec.to(device).eval().requires_grad_(False)
    mean = torch.as_tensor(codec_checkpoint["latent_mean"], device=device).float()
    std = torch.as_tensor(codec_checkpoint["latent_std"], device=device).float()
    cached_latents, cached_actions, delta_std, cache_seconds = build_direct_cache(
        codec, config, mean, std, device
    )
    model_config = direct_config_for_preset(
        config.preset,
        latent_dim=codec.config.latent_dim,
        latent_grid_size=codec.config.latent_grid_size,
        temporal_downsample=codec.config.temporal_downsample,
        history_latents=config.history_latents,
    )
    model = DirectLatentTransformer(model_config, delta_std).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    started = time.perf_counter()
    losses: list[float] = []
    log_path = output_dir / "train.jsonl"
    max_start = config.cache_latents - config.history_latents - 1
    model.train()
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.steps + 1):
            indices = torch.randint(0, cached_latents.shape[0], (config.batch_size,), device=device)
            starts = torch.randint(0, max_start + 1, (config.batch_size,), device=device)
            offsets = torch.arange(config.history_latents, device=device)[None]
            latent_indices = starts[:, None] + offsets
            samples = cached_latents[indices[:, None], latent_indices].float()
            targets = cached_latents[indices[:, None], latent_indices + 1].float()
            flat_actions = cached_actions[indices]
            action_pairs = flat_actions.reshape(
                config.batch_size, config.cache_latents - 1, model_config.temporal_downsample
            )
            action_pairs = action_pairs[
                torch.arange(config.batch_size, device=device)[:, None], latent_indices
            ]
            learning_rate = config.learning_rate * _learning_rate_multiplier(step, config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = direct_training_loss(
                    model,
                    samples,
                    targets,
                    action_pairs,
                    corruption_std=config.corruption_std,
                    late_token_weight=config.late_token_weight,
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            _update_ema(ema_model, model, min(config.ema_decay, (1.0 + step) / (10.0 + step)))
            losses.append(float(loss.detach()))
            if step == 1 or step % config.log_every == 0 or step == config.steps:
                payload = {
                    "step": step,
                    "loss": losses[-1],
                    "loss_ema_50": float(np.mean(losses[-50:])),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": learning_rate,
                    "examples_per_second": step * config.batch_size / max(time.perf_counter() - started, 1e-6),
                }
                print(json.dumps(payload), flush=True)
                log_file.write(json.dumps(payload) + "\n")
                log_file.flush()

    ema_model.eval()
    preview = ClipDataset(
        1,
        seed=config.seed + 1_000_000,
        context_frames=config.history_latents * 2,
        future_frames=config.rollout_frames,
        image_size=config.image_size,
    )[0]
    preview_context = preview["context"].unsqueeze(0).to(device)
    preview_target = preview["target"].unsqueeze(0).to(device)
    preview_actions = preview["actions"].unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        preview_prediction = predict_video(
            ema_model, codec, preview_context, preview_actions, mean, std
        )
    save_rollout_comparison(
        output_dir / "rollout.png",
        preview_context[0, -6:],
        preview_target[0, :12],
        preview_prediction[0, :12],
    )
    save_rollout_comparison(
        output_dir / "rollout-long.png",
        preview_context[0, -6:],
        preview_target[0, -12:],
        preview_prediction[0, -12:],
    )
    evaluation = evaluate_direct_model(ema_model, codec, config, mean, std, device)
    checkpoint = {
        "kind": "direct_latent_world_model",
        "model": ema_model.state_dict(),
        "model_config": model_config.to_dict(),
        "delta_std": delta_std.cpu(),
        "codec_checkpoint": codec_checkpoint,
        "train_config": asdict(config),
        "step": config.steps,
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    summary: dict[str, Any] = {
        "kind": "direct_latent_world_model",
        "config": asdict(config),
        "model_config": model_config.to_dict(),
        "parameters": sum(parameter.numel() for parameter in ema_model.parameters()),
        "codec_parameters": sum(parameter.numel() for parameter in codec.parameters()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seconds": time.perf_counter() - started,
        "cache_seconds": cache_seconds,
        "final_loss": losses[-1],
        "loss_ema_50": float(np.mean(losses[-50:])),
        "evaluation": evaluation,
        "artifacts": ["checkpoint.pt", "rollout.png", "rollout-long.png", "train.jsonl", "summary.json"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the direct autoregressive latent transformer")
    parser.add_argument("--output-dir", default=DirectTrainConfig.output_dir)
    parser.add_argument("--codec-checkpoint", required=True)
    parser.add_argument("--preset", choices=("micro", "tiny", "small"), default=DirectTrainConfig.preset)
    parser.add_argument("--steps", type=int, default=DirectTrainConfig.steps)
    parser.add_argument("--batch-size", type=int, default=DirectTrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=DirectTrainConfig.learning_rate)
    parser.add_argument("--workers", type=int, default=DirectTrainConfig.workers)
    parser.add_argument("--eval-samples", type=int, default=DirectTrainConfig.eval_samples)
    parser.add_argument("--rollout-frames", type=int, default=DirectTrainConfig.rollout_frames)
    parser.add_argument("--cache-samples", type=int, default=DirectTrainConfig.cache_samples)
    args = parser.parse_args()
    summary = train_direct(
        DirectTrainConfig(
            output_dir=args.output_dir,
            codec_checkpoint_path=args.codec_checkpoint,
            preset=args.preset,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            workers=args.workers,
            eval_samples=args.eval_samples,
            rollout_frames=args.rollout_frames,
            cache_samples=args.cache_samples,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
