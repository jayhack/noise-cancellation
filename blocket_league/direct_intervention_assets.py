from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data import make_clip
from .direct_model import build_direct_pipeline_from_checkpoint
from .direct_probe import _fit_downstream_direction, _rollout, fit_activation_puck_locator
from .latent_model import normalize_latents, unnormalize_latents
from .metrics import _soft_centroid
from .trajectory_assets import _video_to_uint8


def _video_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value.copy()).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)


@torch.no_grad()
def render_direct_intervention_atlases(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    candidate_seeds: list[int],
    scenarios: int = 3,
    rollout_frames: int = 12,
    strength: float = 2.0,
    asset_url_prefix: str = "/blocket-league/interventions",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, codec, mean, std = build_direct_pipeline_from_checkpoint(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval().requires_grad_(False)
    codec = codec.to(device).eval().requires_grad_(False)
    mean, std = mean.to(device), std.to(device)
    block_index = 3
    with torch.enable_grad():
        direction, sigma, lens = _fit_downstream_direction(
            model, codec, mean, std, block_index=block_index, seed=811_772,
            samples=64, batch_size=16, target="puck_vx",
        )
    locator, locator_protocol = fit_activation_puck_locator(
        model, codec, mean, std, block_index=block_index, seed=813_774,
    )
    temporal = model.config.temporal_downsample
    context_frames = model.config.history_latents * temporal
    candidates: list[dict[str, Any]] = []
    for seed in candidate_seeds:
        clip = make_clip(seed, context_frames=context_frames, future_frames=rollout_frames,
                         image_size=codec.config.image_size)
        context_video = _video_tensor(clip["context"]).unsqueeze(0).to(device)
        raw, _ = codec.encode(context_video)
        history = normalize_latents(raw.float(), mean, std)
        coast_actions = torch.zeros(1, rollout_frames, device=device, dtype=torch.long)
        recorded: list[torch.Tensor] = []
        baseline_latents = _rollout(model, history, coast_actions, block_index=block_index,
                                    direction=None, amplitude=0.0, record=recorded)
        cells = torch.stack([locator.cells(value) for value in recorded], dim=1)
        activated_latents = _rollout(
            model, history, coast_actions, block_index=block_index, direction=direction,
            amplitude=strength * sigma, cells=cells,
        )
        baseline_future = codec.decode(unnormalize_latents(baseline_latents, mean, std))[:, :rollout_frames]
        activated_future = codec.decode(unnormalize_latents(activated_latents, mean, std))[:, :rollout_frames]
        shared = context_video[:, -1:]
        baseline_video = torch.cat((shared, baseline_future), dim=1)[0]
        activated_video = torch.cat((shared, activated_future), dim=1)[0]
        baseline_position, _ = _soft_centroid(baseline_video[None].float(), "puck")
        activated_position, _ = _soft_centroid(activated_video[None].float(), "puck")
        baseline_px = baseline_position[0] * codec.config.image_size
        activated_px = activated_position[0] * codec.config.image_size
        separation = activated_px[:, 0] - baseline_px[:, 0]
        baseline_step = baseline_px[1:] - baseline_px[:-1]
        activated_step = activated_px[1:] - activated_px[:-1]
        candidates.append({
            "seed": seed,
            "baseline": baseline_video,
            "activated": activated_video,
            "baselinePosition": baseline_px.cpu(),
            "activatedPosition": activated_px.cpu(),
            "finalSeparation": float(separation[-1]),
            "meanVelocityDelta": float((activated_step[:, 0] - baseline_step[:, 0]).mean()),
            "score": float(separation[-1] + 0.3 * separation.abs().max()),
        })

    selected = sorted(candidates, key=lambda item: item["score"], reverse=True)[:scenarios]
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_payloads: list[dict[str, Any]] = []
    frame_size = codec.config.image_size
    for index, item in enumerate(selected):
        rows = [_video_to_uint8(item["baseline"]), _video_to_uint8(item["activated"])]
        atlas = np.full((2 * frame_size, (rollout_frames + 1) * frame_size, 3), (7, 11, 16), dtype=np.uint8)
        for row, video in enumerate(rows):
            for frame, image in enumerate(video):
                atlas[row * frame_size:(row + 1) * frame_size,
                      frame * frame_size:(frame + 1) * frame_size] = image
        atlas_name = f"phantom-force-{index + 1}.png"
        Image.fromarray(atlas).save(output_dir / atlas_name, optimize=True)
        scenario_payloads.append({
            "id": f"phantom-force-{index + 1}",
            "title": f"Held-out world {index + 1}",
            "seed": item["seed"],
            "atlas": f"{asset_url_prefix.rstrip('/')}/{atlas_name}",
            "finalSeparationPx": round(item["finalSeparation"], 3),
            "meanVelocityDeltaPxPerFrame": round(item["meanVelocityDelta"], 3),
            "baselinePuckXY": [[round(float(x), 2), round(float(y), 2)] for x, y in item["baselinePosition"]],
            "activatedPuckXY": [[round(float(x), 2), round(float(y), 2)] for x, y in item["activatedPosition"]],
        })
    manifest = {
        "version": 1,
        "checkpointStep": int(checkpoint["step"]),
        "frameSize": frame_size,
        "frames": rollout_frames + 1,
        "playbackFps": 5,
        "action": "coast for every generated frame",
        "sourceLayer": "block-4",
        "writeStrengthSigmas": strength,
        "writePersistence": "once per autoregressive two-frame transition",
        "directionContexts": lens["contexts"],
        "locatorAccuracy": locator_protocol["visualCellAccuracy"],
        "scenarios": scenario_payloads,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
