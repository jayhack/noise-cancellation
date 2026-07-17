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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .data import ClipDataset
from .metrics import trajectory_metrics
from .model import DiffusionSchedule, VideoDiT, config_for_preset, parameter_count


@dataclass
class TrainConfig:
    output_dir: str = "blocket_league/outputs/local"
    preset: str = "micro"
    steps: int = 2_000
    batch_size: int = 12
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    seed: int = 7
    workers: int = 4
    log_every: int = 25
    preview_ddim_steps: int = 8
    eval_samples: int = 64
    eval_ddim_steps: int = 8
    image_size: int = 64
    context_frames: int = 6
    future_frames: int = 8
    prediction_type: str = "x0"
    patch_size: int = 4
    attention_mode: str = "factorized"
    foreground_weight: float = 10.0
    puck_weight: float = 28.0
    terminal_timestep_fraction: float = 0.35
    late_frame_weight: float = 1.75
    ema_decay: float = 0.9995
    warmup_steps: int = 500
    min_learning_rate_ratio: float = 0.1
    init_checkpoint_path: str = ""
    rollout_context_fraction: float = 0.0
    rollout_context_start_step: int = 3_000
    rollout_context_ramp_steps: int = 7_000
    rollout_context_ddim_steps: int = 1


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_uint8(video: torch.Tensor) -> np.ndarray:
    return (
        video.detach()
        .float()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .round()
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def save_rollout_comparison(
    path: Path,
    context: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> None:
    context_frames = _to_uint8(context)
    target_frames = _to_uint8(target)
    prediction_frames = _to_uint8(prediction)
    scale = 3
    tile = context_frames.shape[1] * scale
    label = 22
    gap = 4
    columns = max(len(context_frames), len(target_frames))
    canvas = Image.new("RGB", (columns * (tile + gap) + gap, 3 * (tile + label + gap) + gap), (8, 12, 17))
    draw = ImageDraw.Draw(canvas)
    rows = (("context", context_frames), ("truth", target_frames), ("diffusion", prediction_frames))
    for row, (name, frames) in enumerate(rows):
        top = gap + row * (tile + label + gap)
        draw.text((gap, top + 4), name, fill=(226, 232, 224))
        for column, frame in enumerate(frames):
            image = Image.fromarray(frame).resize((tile, tile), Image.Resampling.NEAREST)
            canvas.paste(image, (gap + column * (tile + gap), top + label))
    canvas.save(path)


def save_checkpoint(path: Path, model: VideoDiT, config: TrainConfig, step: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.config.to_dict(),
            "train_config": asdict(config),
            "step": step,
        },
        path,
    )


def _learning_rate_multiplier(step: int, config: TrainConfig) -> float:
    if step <= config.warmup_steps:
        return step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return config.min_learning_rate_ratio + (1.0 - config.min_learning_rate_ratio) * cosine


def _rollout_context_probability(step: int, config: TrainConfig) -> float:
    if config.rollout_context_fraction <= 0 or step < config.rollout_context_start_step:
        return 0.0
    if config.rollout_context_ramp_steps <= 0:
        return config.rollout_context_fraction
    progress = (step - config.rollout_context_start_step + 1) / config.rollout_context_ramp_steps
    return config.rollout_context_fraction * min(max(progress, 0.0), 1.0)


def initialize_from_checkpoint(model: VideoDiT, checkpoint_path: Path) -> dict[str, Any]:
    """Load compatible weights while allowing the learned frame table to grow."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model"]
    target = model.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    extended_frames = 0
    for name, target_value in target.items():
        if name not in source:
            raise ValueError(f"Initialization checkpoint is missing {name}")
        source_value = source[name]
        if source_value.shape == target_value.shape:
            adapted[name] = source_value
            continue
        if (
            name == "frame_position"
            and source_value.ndim == target_value.ndim == 4
            and source_value.shape[0] == target_value.shape[0] == 1
            and source_value.shape[2:] == target_value.shape[2:]
            and source_value.shape[1] < target_value.shape[1]
        ):
            value = target_value.clone()
            value[:, : source_value.shape[1]] = source_value
            adapted[name] = value
            extended_frames = target_value.shape[1] - source_value.shape[1]
            continue
        raise ValueError(
            f"Cannot initialize {name}: checkpoint shape {tuple(source_value.shape)} "
            f"does not match model shape {tuple(target_value.shape)}"
        )
    model.load_state_dict(adapted, strict=True)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "source_model_config": checkpoint.get("model_config", {}),
        "extended_frame_positions": extended_frames,
    }


@torch.no_grad()
def _update_ema(ema_model: VideoDiT, model: VideoDiT, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def train(config: TrainConfig) -> dict[str, Any]:
    if not 0.0 <= config.rollout_context_fraction <= 1.0:
        raise ValueError("rollout_context_fraction must be in [0, 1]")
    if config.rollout_context_ddim_steps < 1:
        raise ValueError("rollout_context_ddim_steps must be positive")
    _seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model_config = config_for_preset(
        config.preset,
        image_size=config.image_size,
        patch_size=config.patch_size,
        context_frames=config.context_frames,
        future_frames=config.future_frames,
        prediction_type=config.prediction_type,
        attention_mode=config.attention_mode,
    )
    model = VideoDiT(model_config)
    initialization: dict[str, Any] | None = None
    if config.init_checkpoint_path:
        initialization = initialize_from_checkpoint(
            model,
            Path(config.init_checkpoint_path).expanduser().resolve(),
        )
    model = model.to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    schedule = DiffusionSchedule(
        model_config.diffusion_steps,
        model_config.noise_schedule,
        model_config.prediction_type,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    training_future_frames = config.future_frames * (
        2 if config.rollout_context_fraction > 0 else 1
    )
    dataset = ClipDataset(
        max(config.steps * config.batch_size * 2, 1_024),
        seed=config.seed,
        context_frames=config.context_frames,
        future_frames=training_future_frames,
        image_size=config.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0,
        drop_last=True,
    )
    iterator = iter(loader)
    parameter_total = parameter_count(model)
    started = time.perf_counter()
    losses: list[float] = []
    log_path = output_dir / "train.jsonl"

    model.train()
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            context = batch["context"].to(device, non_blocking=True)
            all_targets = batch["target"].to(device, non_blocking=True)
            all_actions = batch["actions"].to(device, non_blocking=True)
            target = all_targets[:, : config.future_frames]
            actions = all_actions[:, : config.future_frames]

            rollout_probability = _rollout_context_probability(step, config)
            rollout_examples = 0
            if rollout_probability > 0:
                rollout_mask = torch.rand(context.shape[0], device=device) < rollout_probability
                rollout_examples = int(rollout_mask.sum().item())
                if rollout_examples > 0:
                    with torch.no_grad(), torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        generated_prefix = schedule.sample(
                            ema_model,
                            context[rollout_mask],
                            all_actions[rollout_mask, : config.future_frames],
                            ddim_steps=config.rollout_context_ddim_steps,
                        )
                    context = context.clone()
                    target = target.clone()
                    actions = actions.clone()
                    context[rollout_mask] = generated_prefix[
                        :, -config.context_frames :
                    ].to(context.dtype)
                    target[rollout_mask] = all_targets[
                        rollout_mask,
                        config.future_frames : config.future_frames * 2,
                    ]
                    actions[rollout_mask] = all_actions[
                        rollout_mask,
                        config.future_frames : config.future_frames * 2,
                    ]

            optimizer.zero_grad(set_to_none=True)
            learning_rate = config.learning_rate * _learning_rate_multiplier(step, config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss = schedule.training_loss(
                    model,
                    target,
                    context,
                    actions,
                    foreground_weight=config.foreground_weight,
                    puck_weight=config.puck_weight,
                    terminal_timestep_fraction=config.terminal_timestep_fraction,
                    late_frame_weight=config.late_frame_weight,
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema_decay = min(config.ema_decay, (1.0 + step) / (10.0 + step))
            _update_ema(ema_model, model, ema_decay)
            losses.append(float(loss.detach()))

            if step == 1 or step % config.log_every == 0 or step == config.steps:
                elapsed = time.perf_counter() - started
                payload = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "loss_ema_50": float(np.mean(losses[-50:])),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": learning_rate,
                    "examples_per_second": step * config.batch_size / max(elapsed, 1e-6),
                    "rollout_context_probability": rollout_probability,
                    "rollout_context_examples": rollout_examples,
                }
                print(json.dumps(payload), flush=True)
                log_file.write(json.dumps(payload) + "\n")
                log_file.flush()

    checkpoint = output_dir / "checkpoint.pt"
    save_checkpoint(checkpoint, ema_model, config, config.steps)

    preview_dataset = ClipDataset(
        1,
        seed=config.seed + 1_000_000,
        context_frames=config.context_frames,
        future_frames=training_future_frames,
        image_size=config.image_size,
    )
    preview = preview_dataset[0]
    context = preview["context"].unsqueeze(0).to(device)
    all_preview_targets = preview["target"].unsqueeze(0).to(device)
    all_preview_actions = preview["actions"].unsqueeze(0).to(device)
    target = all_preview_targets[:, : config.future_frames]
    actions = all_preview_actions[:, : config.future_frames]
    preview_results: dict[str, dict[str, float]] = {}
    prediction = target
    for ddim_steps in sorted({1, config.preview_ddim_steps}):
        prediction = schedule.sample(
            ema_model,
            context,
            actions,
            ddim_steps=ddim_steps,
            generator=torch.Generator(device=device).manual_seed(config.seed + 101),
        )
        rollout_name = "rollout.png" if ddim_steps == config.preview_ddim_steps else f"rollout-ddim-{ddim_steps}.png"
        save_rollout_comparison(output_dir / rollout_name, context[0], target[0], prediction[0])
        preview_results[str(ddim_steps)] = {
            "pixel_mse": float(torch.mean((prediction - target) ** 2)),
            **trajectory_metrics(
                prediction,
                preview["state"][: config.future_frames].unsqueeze(0).to(device),
            ),
        }

    rollout_preview: dict[str, float] = {}
    if config.rollout_context_fraction > 0:
        rollout_target = all_preview_targets[
            :, config.future_frames : config.future_frames * 2
        ]
        rollout_prediction = schedule.sample(
            ema_model,
            prediction[:, -config.context_frames :],
            all_preview_actions[:, config.future_frames : config.future_frames * 2],
            ddim_steps=config.preview_ddim_steps,
            generator=torch.Generator(device=device).manual_seed(config.seed + 202),
        )
        save_rollout_comparison(
            output_dir / "rollout-autoregressive.png",
            prediction[0, -config.context_frames :],
            rollout_target[0],
            rollout_prediction[0],
        )
        rollout_preview = {
            "pixel_mse": float(torch.mean((rollout_prediction - rollout_target) ** 2)),
            **trajectory_metrics(
                rollout_prediction,
                preview["state"][
                    config.future_frames : config.future_frames * 2
                ].unsqueeze(0).to(device),
            ),
        }

    evaluation: dict[str, float] = {}
    rollout_evaluation: dict[str, float] = {}
    if config.eval_samples > 0:
        evaluation_dataset = ClipDataset(
            config.eval_samples,
            seed=config.seed + 2_000_000,
            context_frames=config.context_frames,
            future_frames=training_future_frames,
            image_size=config.image_size,
        )
        evaluation_loader = DataLoader(
            evaluation_dataset,
            batch_size=min(config.batch_size, 16),
            num_workers=config.workers,
            pin_memory=device.type == "cuda",
        )
        totals: dict[str, float] = {}
        rollout_totals: dict[str, float] = {}
        evaluated = 0
        pixel_total = 0.0
        rollout_pixel_total = 0.0
        for batch in evaluation_loader:
            eval_context = batch["context"].to(device, non_blocking=True)
            eval_all_targets = batch["target"].to(device, non_blocking=True)
            eval_all_actions = batch["actions"].to(device, non_blocking=True)
            eval_all_state = batch["state"].to(device, non_blocking=True)
            eval_target = eval_all_targets[:, : config.future_frames]
            eval_actions = eval_all_actions[:, : config.future_frames]
            eval_state = eval_all_state[:, : config.future_frames]
            eval_prediction = schedule.sample(
                ema_model,
                eval_context,
                eval_actions,
                ddim_steps=config.eval_ddim_steps,
                generator=torch.Generator(device=device).manual_seed(config.seed + 10_000 + evaluated),
            )
            count = eval_context.shape[0]
            batch_metrics = trajectory_metrics(eval_prediction, eval_state)
            for name, value in batch_metrics.items():
                totals[name] = totals.get(name, 0.0) + value * count
            pixel_total += float(torch.mean((eval_prediction - eval_target) ** 2)) * count
            if config.rollout_context_fraction > 0:
                eval_rollout_target = eval_all_targets[
                    :, config.future_frames : config.future_frames * 2
                ]
                eval_rollout_state = eval_all_state[
                    :, config.future_frames : config.future_frames * 2
                ]
                eval_rollout_prediction = schedule.sample(
                    ema_model,
                    eval_prediction[:, -config.context_frames :],
                    eval_all_actions[:, config.future_frames : config.future_frames * 2],
                    ddim_steps=config.eval_ddim_steps,
                    generator=torch.Generator(device=device).manual_seed(
                        config.seed + 20_000 + evaluated
                    ),
                )
                batch_rollout_metrics = trajectory_metrics(
                    eval_rollout_prediction,
                    eval_rollout_state,
                )
                for name, value in batch_rollout_metrics.items():
                    rollout_totals[name] = rollout_totals.get(name, 0.0) + value * count
                rollout_pixel_total += float(
                    torch.mean((eval_rollout_prediction - eval_rollout_target) ** 2)
                ) * count
            evaluated += count
        evaluation = {name: value / evaluated for name, value in totals.items()}
        evaluation["pixel_mse"] = pixel_total / evaluated
        evaluation["samples"] = float(evaluated)
        if rollout_totals:
            rollout_evaluation = {
                name: value / evaluated for name, value in rollout_totals.items()
            }
            rollout_evaluation["pixel_mse"] = rollout_pixel_total / evaluated
            rollout_evaluation["samples"] = float(evaluated)
    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "config": asdict(config),
        "model_config": model_config.to_dict(),
        "parameters": parameter_total,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seconds": elapsed,
        "final_loss": losses[-1],
        "loss_ema_50": float(np.mean(losses[-50:])),
        "initialization": initialization,
        "preview": preview_results,
        "rollout_preview": rollout_preview,
        "evaluation": {"ddim_steps": config.eval_ddim_steps, **evaluation},
        "rollout_evaluation": {
            "ddim_steps": config.eval_ddim_steps,
            **rollout_evaluation,
        },
        "terminal_alpha_bar": float(schedule.alpha_bar[-1]),
        "artifacts": [
            "checkpoint.pt",
            "rollout.png",
            "rollout-ddim-1.png",
            *(["rollout-autoregressive.png"] if rollout_preview else []),
            "train.jsonl",
            "summary.json",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Blocket League video diffusion model")
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--preset", choices=("micro", "tiny", "small"), default=TrainConfig.preset)
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--workers", type=int, default=TrainConfig.workers)
    parser.add_argument("--preview-ddim-steps", type=int, default=TrainConfig.preview_ddim_steps)
    parser.add_argument("--eval-samples", type=int, default=TrainConfig.eval_samples)
    parser.add_argument("--eval-ddim-steps", type=int, default=TrainConfig.eval_ddim_steps)
    parser.add_argument("--context-frames", type=int, default=TrainConfig.context_frames)
    parser.add_argument("--future-frames", type=int, default=TrainConfig.future_frames)
    parser.add_argument("--patch-size", type=int, default=TrainConfig.patch_size)
    parser.add_argument("--attention-mode", choices=("full", "factorized"), default=TrainConfig.attention_mode)
    parser.add_argument("--foreground-weight", type=float, default=TrainConfig.foreground_weight)
    parser.add_argument("--puck-weight", type=float, default=TrainConfig.puck_weight)
    parser.add_argument("--terminal-timestep-fraction", type=float, default=TrainConfig.terminal_timestep_fraction)
    parser.add_argument("--late-frame-weight", type=float, default=TrainConfig.late_frame_weight)
    parser.add_argument("--init-checkpoint", default=TrainConfig.init_checkpoint_path)
    parser.add_argument(
        "--rollout-context-fraction",
        type=float,
        default=TrainConfig.rollout_context_fraction,
    )
    parser.add_argument(
        "--rollout-context-start-step",
        type=int,
        default=TrainConfig.rollout_context_start_step,
    )
    parser.add_argument(
        "--rollout-context-ramp-steps",
        type=int,
        default=TrainConfig.rollout_context_ramp_steps,
    )
    parser.add_argument(
        "--rollout-context-ddim-steps",
        type=int,
        default=TrainConfig.rollout_context_ddim_steps,
    )
    parser.add_argument(
        "--prediction-type",
        choices=("x0", "v", "epsilon"),
        default=TrainConfig.prediction_type,
    )
    args = parser.parse_args()
    summary = train(
        TrainConfig(
            output_dir=args.output_dir,
            preset=args.preset,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            workers=args.workers,
            preview_ddim_steps=args.preview_ddim_steps,
            eval_samples=args.eval_samples,
            eval_ddim_steps=args.eval_ddim_steps,
            context_frames=args.context_frames,
            future_frames=args.future_frames,
            prediction_type=args.prediction_type,
            patch_size=args.patch_size,
            attention_mode=args.attention_mode,
            foreground_weight=args.foreground_weight,
            puck_weight=args.puck_weight,
            terminal_timestep_fraction=args.terminal_timestep_fraction,
            late_frame_weight=args.late_frame_weight,
            init_checkpoint_path=args.init_checkpoint,
            rollout_context_fraction=args.rollout_context_fraction,
            rollout_context_start_step=args.rollout_context_start_step,
            rollout_context_ramp_steps=args.rollout_context_ramp_steps,
            rollout_context_ddim_steps=args.rollout_context_ddim_steps,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
