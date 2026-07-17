from __future__ import annotations

import argparse
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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .codec import CODEC_PALETTE, RepresentationCodec, RepresentationCodecConfig, load_pretrained_dinov2
from .data import ClipDataset
from .metrics import trajectory_metrics


@dataclass
class CodecTrainConfig:
    output_dir: str = "blocket_league/outputs/codec-local"
    steps: int = 8_000
    batch_size: int = 12
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    seed: int = 17
    workers: int = 6
    log_every: int = 50
    eval_samples: int = 128
    image_size: int = 64
    context_frames: int = 6
    future_frames: int = 12
    feature_weight: float = 1.0
    categorical_weight: float = 0.2
    foreground_weight: float = 10.0
    puck_weight: float = 28.0
    warmup_steps: int = 500
    min_learning_rate_ratio: float = 0.1
    dino_model: str = "facebook/dinov2-small"
    latent_dim: int = 32
    decoder_width: int = 160
    decoder_depth: int = 5
    decoder_heads: int = 5
    init_checkpoint_path: str = ""


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _learning_rate_multiplier(step: int, config: CodecTrainConfig) -> float:
    if step <= config.warmup_steps:
        return step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return config.min_learning_rate_ratio + (1.0 - config.min_learning_rate_ratio) * cosine


def _palette_weights(target: torch.Tensor, foreground_weight: float, puck_weight: float) -> torch.Tensor:
    palette = torch.tensor(
        ((50, 213, 173), (11, 57, 52), (239, 242, 233), (150, 161, 153)),
        device=target.device,
        dtype=target.dtype,
    ).div(127.5).sub(1.0)
    target_hwc = target.permute(0, 1, 3, 4, 2)
    distance = (target_hwc.unsqueeze(-2) - palette).square().sum(dim=-1)
    player = distance[..., :2].min(dim=-1).values < 1e-5
    puck = distance[..., 2:].min(dim=-1).values < 1e-5
    return (
        torch.ones_like(player, dtype=target.dtype)
        + player.to(target.dtype) * (foreground_weight - 1.0)
        + puck.to(target.dtype) * (puck_weight - 1.0)
    )


def codec_loss(
    codec: RepresentationCodec,
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    target_features: torch.Tensor,
    logits: torch.Tensor,
    *,
    feature_weight: float,
    categorical_weight: float,
    foreground_weight: float,
    puck_weight: float,
) -> dict[str, torch.Tensor]:
    pixel_error = (reconstruction.float() - target.float()).abs().mean(dim=2)
    weights = _palette_weights(target.float(), foreground_weight, puck_weight)
    loss_l1 = (pixel_error * weights).sum() / weights.sum().clamp_min(1.0)
    palette = torch.tensor(CODEC_PALETTE, device=target.device, dtype=target.dtype).div(127.5).sub(1.0)
    target_hwc = target.permute(0, 1, 3, 4, 2)
    classes = (target_hwc.unsqueeze(-2) - palette).square().sum(dim=-1).argmin(dim=-1)
    batch, frames = target.shape[:2]
    categorical = F.cross_entropy(
        logits.reshape(batch * frames, len(CODEC_PALETTE), target.shape[-2], target.shape[-1]).float(),
        classes.reshape(batch * frames, target.shape[-2], target.shape[-1]),
        reduction="none",
    ).reshape_as(weights)
    loss_categorical = (categorical * weights).sum() / weights.sum().clamp_min(1.0)
    anchor_loss = loss_l1 + categorical_weight * loss_categorical

    loss_feature = target.new_zeros(())
    feature_auto_weight = target.new_ones(())
    if feature_weight > 0:
        frame_count = target.shape[1]
        selected = torch.linspace(
            0,
            frame_count - 1,
            max(1, frame_count // 4),
            device=target.device,
        ).round().long()
        predicted_features = codec.encoder.features(
            reconstruction[:, selected],
            input_grad=True,
        )
        predicted_features = F.normalize(predicted_features.float(), dim=2, eps=1e-6)
        expected_features = F.normalize(
            target_features[:, selected].detach().float(),
            dim=2,
            eps=1e-6,
        )
        loss_feature = F.mse_loss(predicted_features, expected_features)
        if torch.is_grad_enabled() and loss_feature.requires_grad:
            anchor_gradient = torch.autograd.grad(
                anchor_loss,
                codec.decoder.last_layer_weight,
                retain_graph=True,
            )[0]
            feature_gradient = torch.autograd.grad(
                loss_feature,
                codec.decoder.last_layer_weight,
                retain_graph=True,
            )[0]
            feature_auto_weight = (
                anchor_gradient.norm() / (feature_gradient.norm() + 1e-6)
            ).clamp(0.0, 1e4).detach()
    return {
        "loss_total": anchor_loss + feature_weight * feature_auto_weight * loss_feature,
        "loss_l1": loss_l1,
        "loss_categorical": loss_categorical,
        "loss_feature": loss_feature,
        "feature_auto_weight": feature_auto_weight,
    }


def _to_uint8(video: torch.Tensor) -> np.ndarray:
    return (
        video.detach().float().clamp(-1, 1).add(1).mul(127.5).round().byte()
        .permute(0, 2, 3, 1).cpu().numpy()
    )


def save_reconstruction(path: Path, target: torch.Tensor, reconstruction: torch.Tensor) -> None:
    truth = _to_uint8(target)
    decoded = _to_uint8(reconstruction)
    tile = target.shape[-1] * 3
    gap = 4
    label = 22
    columns = len(truth)
    canvas = Image.new("RGB", (columns * (tile + gap) + gap, 2 * (tile + label + gap) + gap), (8, 12, 17))
    draw = ImageDraw.Draw(canvas)
    for row, (name, frames) in enumerate((("pixels", truth), ("RAE decode", decoded))):
        top = gap + row * (tile + label + gap)
        draw.text((gap, top + 4), name, fill=(226, 232, 224))
        for column, frame in enumerate(frames):
            image = Image.fromarray(frame).resize((tile, tile), Image.Resampling.NEAREST)
            canvas.paste(image, (gap + column * (tile + gap), top + label))
    canvas.save(path)


def _sequence(batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.cat((batch["context"], batch["target"]), dim=1).to(device, non_blocking=True)


@torch.no_grad()
def evaluate_codec(
    codec: RepresentationCodec,
    config: CodecTrainConfig,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    dataset = ClipDataset(
        config.eval_samples,
        seed=config.seed + 2_000_000,
        context_frames=config.context_frames,
        future_frames=config.future_frames,
        image_size=config.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, 16),
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
    )
    totals: dict[str, float] = {}
    latent_sum = torch.zeros(config.latent_dim, device=device)
    latent_square_sum = torch.zeros(config.latent_dim, device=device)
    latent_count = 0
    evaluated = 0
    for batch in loader:
        video = _sequence(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            latents, _ = codec.encode(video)
            reconstruction = codec.decode(latents)
        future = reconstruction[:, config.context_frames :]
        truth = batch["target"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        count = video.shape[0]
        metrics = trajectory_metrics(future, state)
        metrics["pixel_mse"] = float(torch.mean((future.float() - truth.float()) ** 2))
        metrics["pixel_l1"] = float(torch.mean((future.float() - truth.float()).abs()))
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value * count
        channel_values = latents.float().permute(2, 0, 1, 3, 4).reshape(config.latent_dim, -1)
        latent_sum += channel_values.sum(dim=1)
        latent_square_sum += channel_values.square().sum(dim=1)
        latent_count += channel_values.shape[1]
        evaluated += count
    mean = latent_sum / latent_count
    variance = latent_square_sum / latent_count - mean.square()
    std = variance.clamp_min(1e-6).sqrt()
    return ({name: value / evaluated for name, value in totals.items()}, mean, std)


def train_codec(config: CodecTrainConfig) -> dict[str, Any]:
    if (config.context_frames + config.future_frames) % 2:
        raise ValueError("The RAEv2 codec needs an even number of total RGB frames")
    _seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    backbone, backbone_config = load_pretrained_dinov2(config.dino_model)
    codec_config = RepresentationCodecConfig(
        image_size=config.image_size,
        latent_dim=config.latent_dim,
        decoder_width=config.decoder_width,
        decoder_depth=config.decoder_depth,
        decoder_heads=config.decoder_heads,
    )
    codec = RepresentationCodec(codec_config, backbone).to(device)
    initialization: dict[str, Any] | None = None
    if config.init_checkpoint_path:
        source_checkpoint = torch.load(
            Path(config.init_checkpoint_path).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        source = source_checkpoint["model"]
        target = codec.state_dict()
        compatible = {
            name: value
            for name, value in source.items()
            if name in target and value.shape == target[name].shape
        }
        target.update(compatible)
        codec.load_state_dict(target)
        initialization = {
            "checkpoint_step": int(source_checkpoint.get("step", 0)),
            "loaded_tensors": len(compatible),
            "total_tensors": len(target),
        }
    trainable = [parameter for parameter in codec.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    dataset = ClipDataset(
        max(config.steps * config.batch_size * 2, 1_024),
        seed=config.seed,
        context_frames=config.context_frames,
        future_frames=config.future_frames,
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
    started = time.perf_counter()
    losses: list[float] = []
    log_path = output_dir / "train.jsonl"
    codec.train()
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, config.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            video = _sequence(batch, device)
            learning_rate = config.learning_rate * _learning_rate_multiplier(step, config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                reconstruction, _, target_features, logits = codec(video)
                terms = codec_loss(
                    codec,
                    video,
                    reconstruction,
                    target_features,
                    logits,
                    feature_weight=config.feature_weight,
                    categorical_weight=config.categorical_weight,
                    foreground_weight=config.foreground_weight,
                    puck_weight=config.puck_weight,
                )
            terms["loss_total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(terms["loss_total"].detach()))
            if step == 1 or step % config.log_every == 0 or step == config.steps:
                elapsed = time.perf_counter() - started
                payload = {
                    "step": step,
                    "loss": losses[-1],
                    "loss_ema_50": float(np.mean(losses[-50:])),
                    "loss_l1": float(terms["loss_l1"].detach()),
                    "loss_feature": float(terms["loss_feature"].detach()),
                    "loss_categorical": float(terms["loss_categorical"].detach()),
                    "feature_auto_weight": float(terms["feature_auto_weight"].detach()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": learning_rate,
                    "examples_per_second": step * config.batch_size / max(elapsed, 1e-6),
                }
                print(json.dumps(payload), flush=True)
                log_file.write(json.dumps(payload) + "\n")
                log_file.flush()

    codec.eval()
    evaluation, latent_mean, latent_std = evaluate_codec(codec, config, device)
    preview = ClipDataset(
        1,
        seed=config.seed + 1_000_000,
        context_frames=config.context_frames,
        future_frames=config.future_frames,
        image_size=config.image_size,
    )[0]
    preview_video = torch.cat((preview["context"], preview["target"]), dim=0).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        preview_latents, _ = codec.encode(preview_video)
        preview_reconstruction = codec.decode(preview_latents)
    save_reconstruction(
        output_dir / "reconstruction.png",
        preview_video[0],
        preview_reconstruction[0],
    )

    checkpoint = {
        "kind": "representation_codec",
        "model": codec.state_dict(),
        "codec_config": codec_config.to_dict(),
        "backbone_config": backbone_config,
        "backbone_name": config.dino_model,
        "latent_mean": latent_mean.cpu(),
        "latent_std": latent_std.cpu(),
        "train_config": asdict(config),
        "step": config.steps,
    }
    torch.save(checkpoint, output_dir / "codec.pt")
    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "kind": "representation_codec",
        "config": asdict(config),
        "codec_config": codec_config.to_dict(),
        "parameters_total": sum(parameter.numel() for parameter in codec.parameters()),
        "parameters_trainable": sum(parameter.numel() for parameter in trainable),
        "compression": {
            "rgb_scalars_per_two_frames": 2 * 3 * config.image_size * config.image_size,
            "latent_scalars_per_two_frames": config.latent_dim * codec_config.latent_grid_size**2,
            "ratio": (2 * 3 * config.image_size * config.image_size)
            / (config.latent_dim * codec_config.latent_grid_size**2),
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seconds": elapsed,
        "final_loss": losses[-1],
        "loss_ema_50": float(np.mean(losses[-50:])),
        "initialization": initialization,
        "latent_mean": latent_mean.tolist(),
        "latent_std": latent_std.tolist(),
        "evaluation": evaluation,
        "artifacts": ["codec.pt", "reconstruction.png", "train.jsonl", "summary.json"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Blocket League representation codec")
    parser.add_argument("--output-dir", default=CodecTrainConfig.output_dir)
    parser.add_argument("--steps", type=int, default=CodecTrainConfig.steps)
    parser.add_argument("--batch-size", type=int, default=CodecTrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=CodecTrainConfig.learning_rate)
    parser.add_argument("--workers", type=int, default=CodecTrainConfig.workers)
    parser.add_argument("--eval-samples", type=int, default=CodecTrainConfig.eval_samples)
    parser.add_argument("--feature-weight", type=float, default=CodecTrainConfig.feature_weight)
    parser.add_argument("--latent-dim", type=int, default=CodecTrainConfig.latent_dim)
    args = parser.parse_args()
    summary = train_codec(
        CodecTrainConfig(
            output_dir=args.output_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            workers=args.workers,
            eval_samples=args.eval_samples,
            feature_weight=args.feature_weight,
            latent_dim=args.latent_dim,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
