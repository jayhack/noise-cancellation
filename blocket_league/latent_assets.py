from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data import make_clip
from .env import ACTION_NAMES
from .latent_model import (
    FlowMatchingSchedule,
    build_latent_pipeline_from_checkpoint,
    normalize_latents,
    unnormalize_latents,
)
from .metrics import trajectory_metrics
from .trajectory_assets import EVENT_NAMES, SCENARIO_COPY, _video_to_uint8


def _video_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value.copy()).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)


def _inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def render_latent_trajectory_atlases(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    scenario_seeds: list[int],
    samples_per_scenario: int = 3,
    integration_steps: int = 10,
    rollout_frames: int = 24,
    asset_url_prefix: str = "/blocket-league/trajectories-latent",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, codec, mean, std = build_latent_pipeline_from_checkpoint(checkpoint)
    device = _inference_device()
    model = model.to(device).eval()
    codec = codec.to(device).eval()
    mean = mean.to(device)
    std = std.to(device)
    schedule = FlowMatchingSchedule()
    temporal = model.config.temporal_downsample
    if rollout_frames % temporal:
        raise ValueError(f"rollout_frames must be divisible by {temporal}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "version": 3,
        "frameSize": codec.config.image_size,
        "contextFrames": model.config.context_frames,
        "futureFrames": rollout_frames,
        "modelFutureFrames": rollout_frames,
        "rolloutBoundaries": [],
        "sourceFps": 20,
        "playbackFps": 6,
        "ddimSteps": integration_steps,
        "samplerLabel": f"{integration_steps}-step flow",
        "generationLabel": "LATENT AR",
        "latentStepFrames": temporal,
        "metricBoundary": min(12, rollout_frames),
        "checkpointStep": int(checkpoint.get("step", 0)),
        "latentGrid": codec.config.latent_grid_size,
        "latentChannels": codec.config.latent_dim,
        "compressionRatio": (
            temporal * 3 * codec.config.image_size**2
            / (codec.config.latent_dim * codec.config.latent_grid_size**2)
        ),
        "scenarios": [],
    }

    for scenario_index, seed in enumerate(scenario_seeds):
        print(
            json.dumps(
                {
                    "stage": "render_latent_trajectory",
                    "scenario": scenario_index + 1,
                    "scenarios": len(scenario_seeds),
                    "seed": seed,
                    "device": str(device),
                }
            ),
            flush=True,
        )
        clip = make_clip(
            seed,
            context_frames=model.config.context_frames,
            future_frames=rollout_frames,
            image_size=codec.config.image_size,
        )
        context = _video_tensor(clip["context"]).unsqueeze(0).to(device)
        target = _video_tensor(clip["target"]).unsqueeze(0).to(device)
        actions = torch.from_numpy(clip["actions"].copy()).long().unsqueeze(0).to(device)
        state = torch.from_numpy(clip["state"].copy()).float().unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw_context_latents, _ = codec.encode(context)
            context_latents = normalize_latents(raw_context_latents.float(), mean, std)
            sample_latents = schedule.sample_autoregressive(
                model,
                context_latents.repeat(samples_per_scenario, 1, 1, 1, 1),
                actions.repeat(samples_per_scenario, 1),
                rollout_frames=rollout_frames,
                integration_steps=integration_steps,
                generator=torch.Generator(device=device).manual_seed(91_003 + seed),
            )
            decode_context = context_latents.repeat(samples_per_scenario, 1, 1, 1, 1)
            decode_latents = torch.cat((decode_context, sample_latents), dim=1)
            if decode_latents.shape[1] <= codec.config.max_latent_frames:
                decoded = codec.decode(unnormalize_latents(decode_latents, mean, std))
                sample_batch = decoded[
                    :, model.config.context_frames : model.config.context_frames + rollout_frames
                ]
            else:
                # The 64-frame exhibit fills the codec's entire 32-latent decode window.
                # Decode the frozen world-model samples directly; the displayed context is
                # still the exact shared simulator input stored separately in the atlas.
                sample_batch = codec.decode(unnormalize_latents(sample_latents, mean, std))[
                    :, :rollout_frames
                ]

        context_frames = _video_to_uint8(context[0])
        futures = [_video_to_uint8(target[0])]
        futures.extend(_video_to_uint8(sample_batch[index]) for index in range(samples_per_scenario))
        frame_size = codec.config.image_size
        total_frames = model.config.context_frames + rollout_frames
        atlas = np.empty((len(futures) * frame_size, total_frames * frame_size, 3), dtype=np.uint8)
        atlas[:] = (7, 11, 16)
        for row, future in enumerate(futures):
            sequence = np.concatenate((context_frames, future), axis=0)
            for column, frame in enumerate(sequence):
                atlas[
                    row * frame_size : (row + 1) * frame_size,
                    column * frame_size : (column + 1) * frame_size,
                ] = frame

        scenario_id, title, description = SCENARIO_COPY.get(
            seed,
            (f"scenario-{scenario_index + 1}", f"Scenario {scenario_index + 1}", "Held-out trajectory."),
        )
        atlas_name = f"{scenario_id}.png"
        Image.fromarray(atlas).save(output_dir / atlas_name, optimize=True)
        lanes: list[dict[str, Any]] = [
            {
                "id": "truth",
                "label": "Ground truth",
                "kind": "truth",
                "playerErrorPx": 0.0,
                "puckErrorPx": 0.0,
            }
        ]
        split = min(12, rollout_frames)
        for index in range(samples_per_scenario):
            overall = trajectory_metrics(sample_batch[index : index + 1], state)
            first = trajectory_metrics(sample_batch[index : index + 1, :split], state[:, :split])
            second = (
                trajectory_metrics(sample_batch[index : index + 1, split:], state[:, split:])
                if split < rollout_frames
                else first
            )
            lanes.append(
                {
                    "id": f"sample-{index + 1}",
                    "label": f"Hallucination {chr(65 + index)}",
                    "kind": "sample",
                    "playerErrorPx": round(overall["player_position_error_px"], 2),
                    "puckErrorPx": round(overall["puck_position_error_px"], 2),
                    "directPlayerErrorPx": round(first["player_position_error_px"], 2),
                    "directPuckErrorPx": round(first["puck_position_error_px"], 2),
                    "rolledPlayerErrorPx": round(second["player_position_error_px"], 2),
                    "rolledPuckErrorPx": round(second["puck_position_error_px"], 2),
                }
            )
        manifest["scenarios"].append(
            {
                "id": scenario_id,
                "title": title,
                "description": description,
                "seed": seed,
                "atlas": f"{asset_url_prefix.rstrip('/')}/{atlas_name}",
                "actions": [ACTION_NAMES[int(value)] for value in clip["actions"][:rollout_frames]],
                "events": [EVENT_NAMES[int(value)] for value in clip["events"][:rollout_frames]],
                "lanes": lanes,
            }
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@torch.no_grad()
def render_codec_atlases(
    codec_checkpoint_path: Path,
    output_dir: Path,
    *,
    scenario_seeds: list[int],
    asset_url_prefix: str = "/blocket-league/codec",
) -> dict[str, Any]:
    from .codec import load_codec_checkpoint

    codec, checkpoint = load_codec_checkpoint(codec_checkpoint_path)
    device = _inference_device()
    codec = codec.to(device).eval()
    mean = torch.as_tensor(checkpoint["latent_mean"], device=device).float()
    std = torch.as_tensor(checkpoint["latent_std"], device=device).float()
    frames = 18
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "version": 1,
        "frameSize": codec.config.image_size,
        "totalFrames": frames,
        "playbackFps": 6,
        "temporalDownsample": codec.config.temporal_downsample,
        "latentGrid": codec.config.latent_grid_size,
        "latentChannels": codec.config.latent_dim,
        "compressionRatio": (
            codec.config.temporal_downsample * 3 * codec.config.image_size**2
            / (codec.config.latent_dim * codec.config.latent_grid_size**2)
        ),
        "checkpointStep": int(checkpoint.get("step", 0)),
        "scenarios": [],
    }
    for scenario_index, seed in enumerate(scenario_seeds):
        clip = make_clip(seed, context_frames=6, future_frames=12, image_size=codec.config.image_size)
        video = _video_tensor(clip["frames"]).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw_latents, _ = codec.encode(video)
            reconstruction = codec.decode(raw_latents)
        truth = _video_to_uint8(video[0])
        decoded = _video_to_uint8(reconstruction[0])

        normalized = normalize_latents(raw_latents.float(), mean, std)[0]
        points = normalized.permute(0, 2, 3, 1).reshape(-1, codec.config.latent_dim)
        centered = points - points.mean(dim=0, keepdim=True)
        _, _, components = torch.pca_lowrank(centered, q=3)
        colors = (centered @ components[:, :3]).reshape(
            normalized.shape[0], codec.config.latent_grid_size, codec.config.latent_grid_size, 3
        )
        low = torch.quantile(colors.reshape(-1, 3), 0.02, dim=0)
        high = torch.quantile(colors.reshape(-1, 3), 0.98, dim=0)
        colors = ((colors - low) / (high - low).clamp_min(1e-5)).clamp(0, 1)
        latent_images = (
            colors.mul(255).byte().repeat_interleave(codec.config.temporal_downsample, dim=0).cpu().numpy()
        )[:frames]
        latent_images = np.stack(
            [
                np.asarray(
                    Image.fromarray(frame).resize(
                        (codec.config.image_size, codec.config.image_size),
                        Image.Resampling.NEAREST,
                    )
                )
                for frame in latent_images
            ]
        )
        rows = (truth, decoded, latent_images)
        size = codec.config.image_size
        atlas = np.empty((3 * size, frames * size, 3), dtype=np.uint8)
        for row, row_frames in enumerate(rows):
            for column, frame in enumerate(row_frames):
                atlas[row * size : (row + 1) * size, column * size : (column + 1) * size] = frame
        scenario_id, title, description = SCENARIO_COPY.get(
            seed,
            (f"scenario-{scenario_index + 1}", f"Scenario {scenario_index + 1}", "Held-out clip."),
        )
        atlas_name = f"{scenario_id}.png"
        Image.fromarray(atlas).save(output_dir / atlas_name, optimize=True)
        manifest["scenarios"].append(
            {
                "id": scenario_id,
                "title": title,
                "description": description,
                "atlas": f"{asset_url_prefix.rstrip('/')}/{atlas_name}",
            }
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render representation-codec or latent-world atlases")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seeds", default="2000007,2009980,2129656")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--integration-steps", type=int, default=10)
    parser.add_argument("--rollout-frames", type=int, default=24)
    parser.add_argument("--asset-url-prefix", default="/blocket-league/trajectories-latent")
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("kind") == "latent_world_model":
        manifest = render_latent_trajectory_atlases(
            args.checkpoint,
            args.output_dir,
            scenario_seeds=seeds,
            samples_per_scenario=args.samples,
            integration_steps=args.integration_steps,
            rollout_frames=args.rollout_frames,
            asset_url_prefix=args.asset_url_prefix,
        )
    elif payload.get("kind") == "representation_codec":
        manifest = render_codec_atlases(
            args.checkpoint,
            args.output_dir,
            scenario_seeds=seeds,
            asset_url_prefix=args.asset_url_prefix,
        )
    else:
        raise ValueError(f"Unsupported checkpoint kind: {payload.get('kind')!r}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
