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
from torch.utils.data import DataLoader

from .codec import load_codec_checkpoint
from .data import ClipDataset
from .latent_model import (
    CausalLatentDiT,
    FlowMatchingSchedule,
    latent_config_for_preset,
    normalize_latents,
    unnormalize_latents,
)
from .metrics import trajectory_metrics
from .train import save_rollout_comparison


@dataclass
class LatentTrainConfig:
    output_dir: str = "blocket_league/outputs/latent-local"
    codec_checkpoint_path: str = ""
    preset: str = "tiny"
    steps: int = 30_000
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    seed: int = 23
    workers: int = 6
    log_every: int = 100
    eval_samples: int = 128
    integration_steps: int = 10
    image_size: int = 64
    context_frames: int = 6
    future_frames: int = 12
    rollout_frames: int = 24
    cache_samples: int = 16_384
    cache_batch_size: int = 64
    late_frame_weight: float = 2.0
    ema_decay: float = 0.9995
    warmup_steps: int = 500
    min_learning_rate_ratio: float = 0.1


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _learning_rate_multiplier(step: int, config: LatentTrainConfig) -> float:
    if step <= config.warmup_steps:
        return step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return config.min_learning_rate_ratio + (1.0 - config.min_learning_rate_ratio) * cosine


@torch.no_grad()
def _update_ema(ema_model: CausalLatentDiT, model: CausalLatentDiT, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def _video(batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.cat((batch["context"], batch["target"]), dim=1).to(device, non_blocking=True)


def _split_latents(latents: torch.Tensor, context_latents: int, future_latents: int) -> tuple[torch.Tensor, torch.Tensor]:
    return latents[:, :context_latents], latents[:, context_latents : context_latents + future_latents]


@torch.no_grad()
def build_latent_cache(
    codec: torch.nn.Module,
    config: LatentTrainConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Encode a deterministic simulator bank once so optimization is truly latent-only."""

    dataset = ClipDataset(
        config.cache_samples,
        seed=config.seed,
        context_frames=config.context_frames,
        future_frames=config.future_frames,
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
        video = _video(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw_latents, _ = codec.encode(video)
        latent_batches.append(normalize_latents(raw_latents.float(), mean, std).to(torch.bfloat16).cpu())
        action_batches.append(batch["actions"].to(torch.int16))
        if index == 0 or (index + 1) % 64 == 0 or index + 1 == len(loader):
            encoded = min((index + 1) * config.cache_batch_size, config.cache_samples)
            print(json.dumps({"stage": "encode_cache", "encoded": encoded, "total": config.cache_samples}), flush=True)
    elapsed = time.perf_counter() - started
    latents = torch.cat(latent_batches).to(device)
    actions = torch.cat(action_batches).long().to(device)
    return latents, actions, elapsed


@torch.no_grad()
def _predict_video(
    model: CausalLatentDiT,
    codec: torch.nn.Module,
    schedule: FlowMatchingSchedule,
    context_video: torch.Tensor,
    actions: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    rollout_frames: int,
    integration_steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    context_latents_raw, _ = codec.encode(context_video)
    context_latents = normalize_latents(context_latents_raw.float(), mean, std)
    generated = schedule.sample_autoregressive(
        model,
        context_latents,
        actions,
        rollout_frames=rollout_frames,
        integration_steps=integration_steps,
        generator=generator,
    )
    decode_latents = torch.cat((context_latents, generated), dim=1)
    decoded = codec.decode(unnormalize_latents(decode_latents, mean, std).to(context_video.dtype))
    return decoded[:, context_video.shape[1] : context_video.shape[1] + rollout_frames]


def _accumulate_metrics(
    totals: dict[str, float],
    prediction: torch.Tensor,
    target: torch.Tensor,
    state: torch.Tensor,
    count: int,
) -> None:
    metrics = trajectory_metrics(prediction, state)
    metrics["pixel_mse"] = float(torch.mean((prediction.float() - target.float()) ** 2))
    for name, value in metrics.items():
        totals[name] = totals.get(name, 0.0) + value * count


@torch.no_grad()
def evaluate_latent_model(
    model: CausalLatentDiT,
    codec: torch.nn.Module,
    schedule: FlowMatchingSchedule,
    config: LatentTrainConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    future_frames = max(config.future_frames, config.rollout_frames)
    dataset = ClipDataset(
        config.eval_samples,
        seed=config.seed + 2_000_000,
        context_frames=config.context_frames,
        future_frames=future_frames,
        image_size=config.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, 8),
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
    )
    direct_totals: dict[str, float] = {}
    rolled_totals: dict[str, float] = {}
    codec_totals: dict[str, float] = {}
    evaluated = 0
    for batch in loader:
        context = batch["context"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        states = batch["state"].to(device, non_blocking=True)
        count = context.shape[0]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = _predict_video(
                model,
                codec,
                schedule,
                context,
                actions[:, : config.rollout_frames],
                mean,
                std,
                rollout_frames=config.rollout_frames,
                integration_steps=config.integration_steps,
                generator=torch.Generator(device=device).manual_seed(config.seed + 10_000 + evaluated),
            )
            full_video = torch.cat((context, targets[:, : config.future_frames]), dim=1)
            truth_latents, _ = codec.encode(full_video)
            reconstruction = codec.decode(truth_latents)[:, config.context_frames :]
        _accumulate_metrics(
            direct_totals,
            prediction[:, : config.future_frames],
            targets[:, : config.future_frames],
            states[:, : config.future_frames],
            count,
        )
        _accumulate_metrics(
            codec_totals,
            reconstruction,
            targets[:, : config.future_frames],
            states[:, : config.future_frames],
            count,
        )
        if config.rollout_frames > config.future_frames:
            _accumulate_metrics(
                rolled_totals,
                prediction[:, config.future_frames : config.rollout_frames],
                targets[:, config.future_frames : config.rollout_frames],
                states[:, config.future_frames : config.rollout_frames],
                count,
            )
        evaluated += count
    direct = {name: value / evaluated for name, value in direct_totals.items()}
    direct["samples"] = float(evaluated)
    rolled = {name: value / evaluated for name, value in rolled_totals.items()}
    if rolled:
        rolled["samples"] = float(evaluated)
    codec_metrics = {name: value / evaluated for name, value in codec_totals.items()}
    codec_metrics["samples"] = float(evaluated)
    return direct, rolled, codec_metrics


def train_latent(config: LatentTrainConfig) -> dict[str, Any]:
    if not config.codec_checkpoint_path:
        raise ValueError("codec_checkpoint_path is required")
    if config.rollout_frames < config.future_frames:
        raise ValueError("rollout_frames must be at least future_frames")
    _seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    codec_path = Path(config.codec_checkpoint_path).expanduser().resolve()
    codec, codec_checkpoint = load_codec_checkpoint(codec_path)
    codec = codec.to(device).eval()
    codec.requires_grad_(False)
    codec_config = codec.config
    if codec_config.image_size != config.image_size:
        raise ValueError("Codec and latent model image sizes differ")
    if config.context_frames % codec_config.temporal_downsample:
        raise ValueError("context_frames must align to codec temporal compression")
    if config.future_frames % codec_config.temporal_downsample:
        raise ValueError("future_frames must align to codec temporal compression")
    mean = torch.as_tensor(codec_checkpoint["latent_mean"], device=device).float()
    std = torch.as_tensor(codec_checkpoint["latent_std"], device=device).float()

    model_config = latent_config_for_preset(
        config.preset,
        latent_dim=codec_config.latent_dim,
        latent_grid_size=codec_config.latent_grid_size,
        temporal_downsample=codec_config.temporal_downsample,
        context_frames=config.context_frames,
        future_frames=config.future_frames,
        max_sequence_latents=(config.context_frames + config.future_frames)
        // codec_config.temporal_downsample,
    )
    model = CausalLatentDiT(model_config).to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    schedule = FlowMatchingSchedule()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    started = time.perf_counter()
    cached_latents, cached_actions, cache_seconds = build_latent_cache(
        codec,
        config,
        mean,
        std,
        device,
    )
    training_started = time.perf_counter()
    losses: list[float] = []
    log_path = output_dir / "train.jsonl"
    model.train()
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.steps + 1):
            indices = torch.randint(0, cached_latents.shape[0], (config.batch_size,), device=device)
            latents = cached_latents[indices].float()
            actions = cached_actions[indices]
            context, target = _split_latents(
                latents,
                model_config.context_latents,
                model_config.future_latents,
            )
            action_pairs = actions.reshape(
                actions.shape[0],
                model_config.future_latents,
                model_config.temporal_downsample,
            )
            learning_rate = config.learning_rate * _learning_rate_multiplier(step, config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = schedule.training_loss(
                    model,
                    context,
                    target,
                    action_pairs,
                    late_frame_weight=config.late_frame_weight,
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema_decay = min(config.ema_decay, (1.0 + step) / (10.0 + step))
            _update_ema(ema_model, model, ema_decay)
            losses.append(float(loss.detach()))
            if step == 1 or step % config.log_every == 0 or step == config.steps:
                elapsed = time.perf_counter() - training_started
                payload = {
                    "step": step,
                    "loss": losses[-1],
                    "loss_ema_50": float(np.mean(losses[-50:])),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": learning_rate,
                    "examples_per_second": step * config.batch_size / max(elapsed, 1e-6),
                }
                print(json.dumps(payload), flush=True)
                log_file.write(json.dumps(payload) + "\n")
                log_file.flush()

    ema_model.eval()
    preview = ClipDataset(
        1,
        seed=config.seed + 1_000_000,
        context_frames=config.context_frames,
        future_frames=config.rollout_frames,
        image_size=config.image_size,
    )[0]
    preview_context = preview["context"].unsqueeze(0).to(device)
    preview_target = preview["target"].unsqueeze(0).to(device)
    preview_actions = preview["actions"].unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        preview_prediction = _predict_video(
            ema_model,
            codec,
            schedule,
            preview_context,
            preview_actions,
            mean,
            std,
            rollout_frames=config.rollout_frames,
            integration_steps=config.integration_steps,
            generator=torch.Generator(device=device).manual_seed(config.seed + 101),
        )
    save_rollout_comparison(
        output_dir / "rollout.png",
        preview_context[0],
        preview_target[0, : config.future_frames],
        preview_prediction[0, : config.future_frames],
    )
    if config.rollout_frames > config.future_frames:
        save_rollout_comparison(
            output_dir / "rollout-autoregressive.png",
            preview_prediction[0, config.future_frames - config.context_frames : config.future_frames],
            preview_target[0, config.future_frames : config.rollout_frames],
            preview_prediction[0, config.future_frames : config.rollout_frames],
        )

    direct, rolled, codec_metrics = evaluate_latent_model(
        ema_model,
        codec,
        schedule,
        config,
        mean,
        std,
        device,
    )
    checkpoint = {
        "kind": "latent_world_model",
        "model": ema_model.state_dict(),
        "model_config": model_config.to_dict(),
        "codec_checkpoint": codec_checkpoint,
        "train_config": asdict(config),
        "step": config.steps,
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "kind": "latent_world_model",
        "config": asdict(config),
        "model_config": model_config.to_dict(),
        "parameters": sum(parameter.numel() for parameter in ema_model.parameters()),
        "codec_parameters": sum(parameter.numel() for parameter in codec.parameters()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seconds": elapsed,
        "cache_samples": config.cache_samples,
        "cache_seconds": cache_seconds,
        "training_seconds": time.perf_counter() - training_started,
        "final_loss": losses[-1],
        "loss_ema_50": float(np.mean(losses[-50:])),
        "evaluation": {"integration_steps": config.integration_steps, **direct},
        "rollout_evaluation": {"integration_steps": config.integration_steps, **rolled},
        "codec_ceiling": codec_metrics,
        "artifacts": [
            "checkpoint.pt",
            "rollout.png",
            *( ["rollout-autoregressive.png"] if rolled else [] ),
            "train.jsonl",
            "summary.json",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train latent diffusion forcing for Blocket League")
    parser.add_argument("--output-dir", default=LatentTrainConfig.output_dir)
    parser.add_argument("--codec-checkpoint", required=True)
    parser.add_argument("--preset", choices=("micro", "tiny", "small"), default=LatentTrainConfig.preset)
    parser.add_argument("--steps", type=int, default=LatentTrainConfig.steps)
    parser.add_argument("--batch-size", type=int, default=LatentTrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=LatentTrainConfig.learning_rate)
    parser.add_argument("--workers", type=int, default=LatentTrainConfig.workers)
    parser.add_argument("--eval-samples", type=int, default=LatentTrainConfig.eval_samples)
    parser.add_argument("--integration-steps", type=int, default=LatentTrainConfig.integration_steps)
    parser.add_argument("--rollout-frames", type=int, default=LatentTrainConfig.rollout_frames)
    parser.add_argument("--cache-samples", type=int, default=LatentTrainConfig.cache_samples)
    args = parser.parse_args()
    summary = train_latent(
        LatentTrainConfig(
            output_dir=args.output_dir,
            codec_checkpoint_path=args.codec_checkpoint,
            preset=args.preset,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            workers=args.workers,
            eval_samples=args.eval_samples,
            integration_steps=args.integration_steps,
            rollout_frames=args.rollout_frames,
            cache_samples=args.cache_samples,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
