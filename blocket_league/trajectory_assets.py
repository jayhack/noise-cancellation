from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data import make_clip
from .env import ACTION_NAMES
from .metrics import trajectory_metrics
from .model import DiffusionSchedule, VideoDiT, VideoDiTConfig


SCENARIO_COPY = {
    2_000_007: ("open-ice", "Open ice", "A clean acceleration arc with changing actions."),
    2_009_980: ("bank-shot", "Bank shot", "The puck meets the wall, then the player."),
    2_129_656: ("goal-reset", "Goal + reset", "A scored goal crosses an abrupt state reset."),
}
EVENT_NAMES = ("coast", "thrust", "impact", "wall", "goal", "kickoff")


def _video_to_uint8(video: torch.Tensor) -> np.ndarray:
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


@torch.no_grad()
def sample_autoregressive(
    schedule: DiffusionSchedule,
    model: VideoDiT,
    context: torch.Tensor,
    actions: torch.Tensor,
    *,
    rollout_frames: int,
    ddim_steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Roll a fixed-horizon checkpoint forward by feeding predictions back as context."""

    if rollout_frames < 1:
        raise ValueError("rollout_frames must be positive")
    window = model.config.future_frames
    required_actions = ceil(rollout_frames / window) * window
    if actions.shape != (context.shape[0], required_actions):
        raise ValueError(
            f"Expected actions [B, {required_actions}] for a {rollout_frames}-frame rollout, "
            f"got {tuple(actions.shape)}"
        )

    generated: list[torch.Tensor] = []
    current_context = context
    cursor = 0
    while cursor < rollout_frames:
        prediction = schedule.sample(
            model,
            current_context,
            actions[:, cursor : cursor + window],
            ddim_steps=ddim_steps,
            generator=generator,
        )
        take = min(window, rollout_frames - cursor)
        generated.append(prediction[:, :take])
        cursor += take
        if cursor < rollout_frames:
            current_context = torch.cat((current_context, prediction), dim=1)[
                :, -model.config.context_frames :
            ]
    return torch.cat(generated, dim=1)


@torch.no_grad()
def render_trajectory_atlases(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    scenario_seeds: list[int],
    samples_per_scenario: int = 3,
    ddim_steps: int = 8,
    rollout_frames: int = 12,
    asset_url_prefix: str = "/blocket-league/trajectories",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = VideoDiTConfig(**checkpoint["model_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoDiT(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = DiffusionSchedule(
        model_config.diffusion_steps,
        model_config.noise_schedule,
        model_config.prediction_type,
    ).to(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    required_future_frames = ceil(rollout_frames / model_config.future_frames) * model_config.future_frames
    rollout_boundaries = [
        model_config.context_frames + offset
        for offset in range(model_config.future_frames, rollout_frames, model_config.future_frames)
    ]

    manifest: dict[str, Any] = {
        "version": 2,
        "frameSize": model_config.image_size,
        "contextFrames": model_config.context_frames,
        "futureFrames": rollout_frames,
        "modelFutureFrames": model_config.future_frames,
        "rolloutBoundaries": rollout_boundaries,
        "sourceFps": 20,
        "playbackFps": 6,
        "ddimSteps": ddim_steps,
        "checkpointStep": int(checkpoint.get("step", 0)),
        "scenarios": [],
    }

    for scenario_index, seed in enumerate(scenario_seeds):
        clip = make_clip(
            seed,
            context_frames=model_config.context_frames,
            future_frames=required_future_frames,
            image_size=model_config.image_size,
        )

        def video_tensor(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(value.copy()).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)

        context = video_tensor(clip["context"]).unsqueeze(0).to(device)
        target = video_tensor(clip["target"][:rollout_frames]).unsqueeze(0).to(device)
        actions = torch.from_numpy(clip["actions"].copy()).long().unsqueeze(0).to(device)
        state = torch.from_numpy(clip["state"][:rollout_frames].copy()).float().unsqueeze(0).to(device)
        sample_batch = sample_autoregressive(
            schedule,
            model,
            context.repeat(samples_per_scenario, 1, 1, 1, 1),
            actions.repeat(samples_per_scenario, 1),
            rollout_frames=rollout_frames,
            ddim_steps=ddim_steps,
            generator=torch.Generator(device=device).manual_seed(71_003 + seed),
        )

        context_frames = _video_to_uint8(context[0])
        futures = [_video_to_uint8(target[0])]
        futures.extend(_video_to_uint8(sample_batch[index]) for index in range(samples_per_scenario))
        frame_size = model_config.image_size
        total_frames = model_config.context_frames + rollout_frames
        atlas = np.empty((len(futures) * frame_size, total_frames * frame_size, 3), dtype=np.uint8)
        atlas[:] = (7, 11, 16)
        for row, future in enumerate(futures):
            sequence = np.concatenate((context_frames, future), axis=0)
            for column, frame in enumerate(sequence):
                top = row * frame_size
                left = column * frame_size
                atlas[top : top + frame_size, left : left + frame_size] = frame

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
        for index in range(samples_per_scenario):
            metrics = trajectory_metrics(sample_batch[index : index + 1], state)
            direct_end = min(model_config.future_frames, rollout_frames)
            direct_metrics = trajectory_metrics(
                sample_batch[index : index + 1, :direct_end],
                state[:, :direct_end],
            )
            rolled_metrics = (
                trajectory_metrics(
                    sample_batch[index : index + 1, direct_end:],
                    state[:, direct_end:],
                )
                if direct_end < rollout_frames
                else direct_metrics
            )
            lanes.append(
                {
                    "id": f"sample-{index + 1}",
                    "label": f"Hallucination {chr(65 + index)}",
                    "kind": "sample",
                    "playerErrorPx": round(metrics["player_position_error_px"], 2),
                    "puckErrorPx": round(metrics["puck_position_error_px"], 2),
                    "directPlayerErrorPx": round(direct_metrics["player_position_error_px"], 2),
                    "directPuckErrorPx": round(direct_metrics["puck_position_error_px"], 2),
                    "rolledPlayerErrorPx": round(rolled_metrics["player_position_error_px"], 2),
                    "rolledPuckErrorPx": round(rolled_metrics["puck_position_error_px"], 2),
                }
            )
        manifest["scenarios"].append(
            {
                "id": scenario_id,
                "title": title,
                "description": description,
                "seed": seed,
                "atlas": f"{asset_url_prefix.rstrip('/')}/{atlas_name}",
                "actions": [ACTION_NAMES[int(action)] for action in clip["actions"][:rollout_frames]],
                "events": [EVENT_NAMES[int(event)] for event in clip["events"][:rollout_frames]],
                "lanes": lanes,
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
