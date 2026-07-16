from __future__ import annotations

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

from .data import PassiveClipDataset, make_passive_clip
from .env import PALETTE
from .metrics import trajectory_metrics
from .pixel_direct_model import DirectPixelTransformer, pixel_direct_config_for_preset
from .train import save_rollout_comparison


@dataclass
class PixelDirectTrainConfig:
    output_dir: str = "blocket_league/outputs/pixel-direct-local"
    preset: str = "tiny"
    steps: int = 30_000
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 59
    workers: int = 8
    log_every: int = 100
    eval_samples: int = 128
    image_size: int = 64
    patch_size: int = 4
    history_frames: int = 8
    cache_frames: int = 24
    rollout_frames: int = 64
    metric_boundary: int = 12
    cache_samples: int = 16_384
    cache_batch_size: int = 64
    goal_centered_fraction: float = 0.35
    corruption_rate: float = 0.06
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


def _learning_rate_multiplier(step: int, config: PixelDirectTrainConfig) -> float:
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


def palette_tensor(device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.stack(tuple(PALETTE.values())), device=device, dtype=torch.float32)


def frames_to_classes(frames: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    rgb = frames.float().add(1).mul(127.5).permute(0, 1, 3, 4, 2)
    return (rgb[:, :, :, :, None] - palette[None, None, None, None]).square().sum(-1).argmin(-1)


def classes_to_video(classes: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    rgb = palette[classes.long()].permute(0, 1, 4, 2, 3)
    return rgb.div(127.5).sub(1.0)


@torch.no_grad()
def build_pixel_cache(config: PixelDirectTrainConfig, device: torch.device):
    dataset = PassiveClipDataset(
        config.cache_samples,
        seed=config.seed,
        frames=config.cache_frames,
        image_size=config.image_size,
        goal_centered_fraction=config.goal_centered_fraction,
    )
    loader = DataLoader(dataset, batch_size=config.cache_batch_size, num_workers=config.workers,
                        pin_memory=device.type == "cuda", persistent_workers=config.workers > 0)
    palette = palette_tensor(device)
    frame_batches = []
    started = time.perf_counter()
    for index, batch in enumerate(loader):
        video = batch.to(device, non_blocking=True)
        frame_batches.append(frames_to_classes(video, palette).byte().cpu())
        if index == 0 or (index + 1) % 64 == 0 or index + 1 == len(loader):
            encoded = min((index + 1) * config.cache_batch_size, config.cache_samples)
            print(json.dumps({"stage": "cache_pixels", "encoded": encoded,
                              "total": config.cache_samples}), flush=True)
    frames = torch.cat(frame_batches).to(device)
    counts = torch.bincount(frames.flatten().long(), minlength=len(PALETTE)).float()
    frequencies = counts / counts.sum()
    class_weights = (frequencies.max() / frequencies.clamp_min(1e-8)).sqrt().clamp(0.25, 12.0)
    class_weights /= class_weights[1].clamp_min(1e-6)
    class_weights = class_weights.clamp(0.25, 12.0)
    return frames, class_weights, time.perf_counter() - started, frequencies


def pixel_training_loss(model, inputs, targets, class_weights, *, corruption_rate, late_frame_weight):
    batch, time = inputs.shape[:2]
    severity = torch.rand(batch, 1, 1, 1, device=inputs.device)
    ramp = torch.linspace(0.25, 1.0, time, device=inputs.device)[None, :, None, None]
    corrupt = torch.rand(inputs.shape, device=inputs.device) < severity * ramp * corruption_rate
    replacements = torch.randint(0, model.config.palette_size, inputs.shape, device=inputs.device)
    corrupted = torch.where(corrupt, replacements, inputs.long())
    logits = model(corrupted)
    if not isinstance(logits, torch.Tensor):
        logits = logits[0]
    error = F.cross_entropy(
        logits.flatten(0, 1).float(), targets.flatten(0, 1).long(),
        weight=class_weights, reduction="none",
    ).reshape(batch, time, model.config.image_size, model.config.image_size).mean(dim=(2, 3))
    weights = torch.linspace(1.0, late_frame_weight, time, device=error.device)
    return (error * weights[None]).sum() / (weights.sum() * batch)


@torch.no_grad()
def rollout_pixel_classes(model, context, rollout_frames):
    history = context.long()
    generated = []
    for _ in range(rollout_frames):
        current = history[:, -model.config.history_frames:]
        next_frame = model.next_frame(current)
        generated.append(next_frame)
        history = torch.cat((history, next_frame[:, None]), dim=1)
    return torch.stack(generated, dim=1)


def held_out_clip(
    seed: int,
    config: PixelDirectTrainConfig,
    device: torch.device,
    *,
    goal_centered: bool = False,
):
    clip = make_passive_clip(
        seed,
        context_frames=config.history_frames,
        future_frames=config.rollout_frames,
        image_size=config.image_size,
        goal_centered=goal_centered,
    )
    first = torch.from_numpy(clip["context"].copy()).permute(0, 3, 1, 2)
    rest = torch.from_numpy(clip["target"].copy()).permute(0, 3, 1, 2)
    video = torch.cat((first, rest), dim=0).float().div(127.5).sub(1.0).to(device)
    states = torch.from_numpy(clip["state"].copy()).float().to(device)
    return {
        "context": video[:config.history_frames],
        "target": video[config.history_frames:],
        "state": states,
    }


@torch.no_grad()
def evaluate_pixel_direct(model, config, device, *, goal_centered: bool = False):
    palette = palette_tensor(device)
    short_totals: dict[str, float] = {}
    long_totals: dict[str, float] = {}
    batch_size = min(config.batch_size, 8)
    for start in range(0, config.eval_samples, batch_size):
        items = [held_out_clip(
                     config.seed + 2_000_000 + index,
                     config,
                     device,
                     goal_centered=goal_centered,
                 )
                 for index in range(start, min(start + batch_size, config.eval_samples))]
        context_video = torch.stack([item["context"] for item in items])
        target = torch.stack([item["target"] for item in items])
        states = torch.stack([item["state"] for item in items])
        context = frames_to_classes(context_video, palette)
        prediction_classes = rollout_pixel_classes(model, context, config.rollout_frames)
        prediction = classes_to_video(prediction_classes, palette)
        boundary = min(config.metric_boundary, config.rollout_frames)
        short = trajectory_metrics(prediction[:, :boundary], states[:, :boundary])
        short["pixel_mse"] = float((prediction[:, :boundary] - target[:, :boundary]).square().mean())
        long = trajectory_metrics(prediction, states)
        long["pixel_mse"] = float((prediction - target).square().mean())
        count = len(items)
        for name, value in short.items(): short_totals[name] = short_totals.get(name, 0.0) + value * count
        for name, value in long.items(): long_totals[name] = long_totals.get(name, 0.0) + value * count
    return {
        "short": {name: value / config.eval_samples for name, value in short_totals.items()},
        "long": {name: value / config.eval_samples for name, value in long_totals.items()},
        "samples": config.eval_samples,
        "short_frames": min(config.metric_boundary, config.rollout_frames),
        "long_frames": config.rollout_frames,
    }


def train_pixel_direct(config: PixelDirectTrainConfig) -> dict[str, Any]:
    if config.cache_frames <= config.history_frames:
        raise ValueError("cache_frames must exceed history_frames")
    if not 0.0 <= config.goal_centered_fraction <= 1.0:
        raise ValueError("goal_centered_fraction must be in [0, 1]")
    _seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.set_float32_matmul_precision("high")
    cached_frames, class_weights, cache_seconds, frequencies = build_pixel_cache(config, device)
    model_config = pixel_direct_config_for_preset(
        config.preset, image_size=config.image_size, patch_size=config.patch_size,
        palette_size=len(PALETTE), history_frames=config.history_frames,
    )
    model = DirectPixelTransformer(model_config).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay, betas=(0.9, 0.95),
                                  fused=device.type == "cuda")
    losses: list[float] = []
    started = time.perf_counter()
    max_start = config.cache_frames - config.history_frames - 1
    offsets = torch.arange(config.history_frames, device=device)[None]
    log_path = output_dir / "train.jsonl"
    model.train()
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.steps + 1):
            indices = torch.randint(0, cached_frames.shape[0], (config.batch_size,), device=device)
            starts = torch.randint(0, max_start + 1, (config.batch_size,), device=device)
            frame_indices = starts[:, None] + offsets
            inputs = cached_frames[indices[:, None], frame_indices].long()
            targets = cached_frames[indices[:, None], frame_indices + 1].long()
            learning_rate = config.learning_rate * _learning_rate_multiplier(step, config)
            for group in optimizer.param_groups: group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = pixel_training_loss(model, inputs, targets, class_weights,
                                           corruption_rate=config.corruption_rate,
                                           late_frame_weight=config.late_frame_weight)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            _update_ema(ema_model, model, min(config.ema_decay, (1 + step) / (10 + step)))
            losses.append(float(loss.detach()))
            if step == 1 or step % config.log_every == 0 or step == config.steps:
                payload = {"step": step, "loss": losses[-1],
                           "loss_ema_50": float(np.mean(losses[-50:])),
                           "gradient_norm": float(gradient_norm), "learning_rate": learning_rate,
                           "examples_per_second": step * config.batch_size / max(time.perf_counter() - started, 1e-6)}
                print(json.dumps(payload), flush=True)
                log_file.write(json.dumps(payload) + "\n"); log_file.flush()
    ema_model.eval()
    preview = held_out_clip(config.seed + 1_000_000, config, device)
    palette = palette_tensor(device)
    preview_context = preview["context"][None]
    preview_target = preview["target"][None]
    preview_classes = frames_to_classes(preview_context, palette)
    prediction_classes = rollout_pixel_classes(ema_model, preview_classes, config.rollout_frames)
    prediction = classes_to_video(prediction_classes, palette)
    save_rollout_comparison(output_dir / "rollout.png", preview_context[0, -6:],
                            preview_target[0, :12], prediction[0, :12])
    save_rollout_comparison(output_dir / "rollout-long.png", preview_context[0, -6:],
                            preview_target[0, -12:], prediction[0, -12:])
    evaluation = evaluate_pixel_direct(ema_model, config, device)
    post_goal_evaluation = evaluate_pixel_direct(ema_model, config, device, goal_centered=True)
    checkpoint = {"kind": "passive_direct_pixel_world_model", "model": ema_model.state_dict(),
                  "model_config": model_config.to_dict(), "train_config": asdict(config),
                  "palette": np.stack(tuple(PALETTE.values())), "step": config.steps}
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    summary = {"kind": checkpoint["kind"], "config": asdict(config),
               "model_config": model_config.to_dict(),
               "parameters": sum(parameter.numel() for parameter in ema_model.parameters()),
               "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "seconds": time.perf_counter() - started, "cache_seconds": cache_seconds,
               "final_loss": losses[-1], "loss_ema_50": float(np.mean(losses[-50:])),
               "class_frequencies": frequencies.cpu().tolist(), "class_weights": class_weights.cpu().tolist(),
               "evaluation": evaluation,
               "post_goal_evaluation": post_goal_evaluation,
               "artifacts": ["checkpoint.pt", "rollout.png", "rollout-long.png", "train.jsonl", "summary.json"]}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
