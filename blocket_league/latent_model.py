from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .codec import RepresentationCodec, build_codec_from_checkpoint


@dataclass(frozen=True)
class LatentWorldModelConfig:
    latent_dim: int = 32
    latent_grid_size: int = 4
    temporal_downsample: int = 2
    context_frames: int = 6
    future_frames: int = 12
    hidden_size: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    action_count: int = 9
    max_sequence_latents: int = 9

    @property
    def context_latents(self) -> int:
        return self.context_frames // self.temporal_downsample

    @property
    def future_latents(self) -> int:
        return self.future_frames // self.temporal_downsample

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


LATENT_MODEL_PRESETS: dict[str, dict[str, int]] = {
    "micro": {"hidden_size": 128, "depth": 4, "heads": 4},
    "tiny": {"hidden_size": 192, "depth": 6, "heads": 6},
    "small": {"hidden_size": 256, "depth": 8, "heads": 8},
}


def latent_config_for_preset(name: str, **overrides: object) -> LatentWorldModelConfig:
    if name not in LATENT_MODEL_PRESETS:
        raise ValueError(f"Unknown latent model preset {name!r}")
    return LatentWorldModelConfig(**{**LATENT_MODEL_PRESETS[name], **overrides})


def flow_time_embedding(times: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -log(10_000) * torch.arange(half, device=times.device, dtype=torch.float32) / max(half - 1, 1)
    )
    phases = times.float()[:, :, None] * 1_000.0 * frequencies[None, None]
    embedding = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _modulate(tokens: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return tokens * (1.0 + scale) + shift


class CausalAdaFactorizedBlock(nn.Module):
    """MIRA-style per-frame conditioning with factorized causal attention."""

    def __init__(self, hidden_size: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.spatial_attention = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.temporal_attention = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        inner = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner, hidden_size),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 9),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch, frames, patches, hidden = tokens.shape
        values = self.modulation(condition).chunk(9, dim=-1)
        spatial_shift, spatial_scale, spatial_gate = values[:3]
        temporal_shift, temporal_scale, temporal_gate = values[3:6]
        mlp_shift, mlp_scale, mlp_gate = values[6:]

        spatial = _modulate(
            self.spatial_norm(tokens),
            spatial_shift[:, :, None],
            spatial_scale[:, :, None],
        ).reshape(batch * frames, patches, hidden)
        spatial = self.spatial_attention(spatial, spatial, spatial, need_weights=False)[0]
        tokens = tokens + spatial.reshape(batch, frames, patches, hidden) * spatial_gate[:, :, None]

        temporal = self.temporal_norm(tokens).permute(0, 2, 1, 3)
        temporal = temporal.reshape(batch * patches, frames, hidden)
        temporal_shift = temporal_shift[:, None].expand(batch, patches, frames, hidden).reshape(
            batch * patches, frames, hidden
        )
        temporal_scale = temporal_scale[:, None].expand(batch, patches, frames, hidden).reshape(
            batch * patches, frames, hidden
        )
        temporal = _modulate(temporal, temporal_shift, temporal_scale)
        causal_mask = torch.ones(frames, frames, device=tokens.device, dtype=torch.bool).triu(1)
        temporal = self.temporal_attention(
            temporal,
            temporal,
            temporal,
            attn_mask=causal_mask,
            need_weights=False,
        )[0]
        temporal = temporal.reshape(batch, patches, frames, hidden).permute(0, 2, 1, 3)
        tokens = tokens + temporal * temporal_gate[:, :, None]

        feedforward = _modulate(
            self.mlp_norm(tokens),
            mlp_shift[:, :, None],
            mlp_scale[:, :, None],
        )
        return tokens + self.mlp(feedforward) * mlp_gate[:, :, None]


class CausalLatentDiT(nn.Module):
    """Action-conditioned causal flow transformer over normalized codec latents."""

    def __init__(self, config: LatentWorldModelConfig) -> None:
        super().__init__()
        if config.context_frames % config.temporal_downsample:
            raise ValueError("context_frames must be divisible by temporal_downsample")
        if config.future_frames % config.temporal_downsample:
            raise ValueError("future_frames must be divisible by temporal_downsample")
        if config.hidden_size % config.heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.config = config
        self.input_projection = nn.Linear(config.latent_dim, config.hidden_size)
        self.past_projection = nn.Linear(config.latent_dim, config.hidden_size)
        self.spatial_position = nn.Parameter(
            torch.randn(1, 1, config.latent_grid_size**2, config.hidden_size) * 0.02
        )
        self.temporal_position = nn.Parameter(
            torch.randn(1, config.max_sequence_latents, 1, config.hidden_size) * 0.02
        )
        self.action_embedding = nn.Embedding(config.action_count, config.hidden_size)
        self.action_projection = nn.Linear(config.hidden_size * config.temporal_downsample, config.hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
        )
        self.blocks = nn.ModuleList(
            CausalAdaFactorizedBlock(config.hidden_size, config.heads, config.mlp_ratio)
            for _ in range(config.depth)
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, config.latent_dim)
        self.bos = nn.Parameter(
            torch.randn(1, 1, config.latent_dim, config.latent_grid_size, config.latent_grid_size) * 0.02
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def shifted_clean_past(self, clean_sequence: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.bos.expand(clean_sequence.shape[0], -1, -1, -1, -1), clean_sequence[:, :-1]), dim=1)

    def forward(
        self,
        noisy_sequence: torch.Tensor,
        action_pairs: torch.Tensor,
        times: torch.Tensor,
        clean_past: torch.Tensor,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        batch, frames, channels, height, width = noisy_sequence.shape
        config = self.config
        expected = (config.latent_dim, config.latent_grid_size, config.latent_grid_size)
        if (channels, height, width) != expected:
            raise ValueError(f"Expected latent tail {expected}, got {(channels, height, width)}")
        if frames > config.max_sequence_latents:
            raise ValueError(f"Sequence has {frames} latents; maximum is {config.max_sequence_latents}")
        if action_pairs.shape != (batch, frames, config.temporal_downsample):
            raise ValueError(
                f"Expected actions {(batch, frames, config.temporal_downsample)}, got {tuple(action_pairs.shape)}"
            )
        if times.shape != (batch, frames):
            raise ValueError(f"Expected times {(batch, frames)}, got {tuple(times.shape)}")
        if clean_past.shape != noisy_sequence.shape:
            raise ValueError("clean_past must match the latent sequence shape")

        tokens = noisy_sequence.permute(0, 1, 3, 4, 2).reshape(
            batch, frames, height * width, channels
        )
        past = clean_past.permute(0, 1, 3, 4, 2).reshape(
            batch, frames, height * width, channels
        )
        tokens = (
            self.input_projection(tokens)
            + self.past_projection(past)
            + self.spatial_position
            + self.temporal_position[:, :frames]
        )
        actions = self.action_embedding(action_pairs).reshape(
            batch,
            frames,
            config.temporal_downsample * config.hidden_size,
        )
        condition = self.action_projection(actions) + self.time_mlp(
            flow_time_embedding(times, config.hidden_size)
        )

        hidden_states: list[torch.Tensor] = []
        for block in self.blocks:
            tokens = block(tokens, condition)
            if return_hidden:
                hidden_states.append(tokens.reshape(batch, frames * height * width, config.hidden_size))
        velocity = self.output_projection(self.output_norm(tokens))
        velocity = velocity.reshape(batch, frames, height, width, channels).permute(0, 1, 4, 2, 3)
        if return_hidden:
            return velocity, hidden_states
        return velocity


class FlowMatchingSchedule:
    @staticmethod
    def inference_times(
        integration_steps: int,
        device: torch.device,
        *,
        schedule_type: str = "linear_quadratic",
    ) -> torch.Tensor:
        if integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        if schedule_type == "linear":
            return torch.linspace(0.0, 1.0, integration_steps + 1, device=device)
        if schedule_type != "linear_quadratic":
            raise ValueError(f"Unknown inference schedule {schedule_type!r}")
        if integration_steps < 2:
            return torch.tensor((0.0, 1.0), device=device)
        linear_steps = integration_steps // 2
        quadratic_steps = integration_steps - linear_steps
        linear = torch.linspace(0.0, 0.1, linear_steps + 1, device=device)
        quadratic = torch.linspace(linear[-1].sqrt(), 1.0, quadratic_steps + 1, device=device).square()
        return torch.cat((linear[:-1], quadratic))

    def training_loss(
        self,
        model: CausalLatentDiT,
        context: torch.Tensor,
        target: torch.Tensor,
        action_pairs: torch.Tensor,
        *,
        late_frame_weight: float = 1.0,
    ) -> torch.Tensor:
        batch, future_latents = target.shape[:2]
        noise = torch.randn_like(target)
        times = torch.rand(batch, future_latents, device=target.device, dtype=target.dtype)
        noisy_target = times[:, :, None, None, None] * target + (
            1.0 - times[:, :, None, None, None]
        ) * noise
        clean_sequence = torch.cat((context, target), dim=1)
        noisy_sequence = torch.cat((context, noisy_target), dim=1)
        sequence_times = torch.cat(
            (torch.ones(batch, context.shape[1], device=target.device, dtype=target.dtype), times),
            dim=1,
        )
        context_actions = torch.zeros(
            batch,
            context.shape[1],
            model.config.temporal_downsample,
            device=action_pairs.device,
            dtype=action_pairs.dtype,
        )
        sequence_actions = torch.cat((context_actions, action_pairs), dim=1)
        prediction = model(
            noisy_sequence,
            sequence_actions,
            sequence_times,
            model.shifted_clean_past(clean_sequence),
        )
        if not isinstance(prediction, torch.Tensor):
            prediction = prediction[0]
        error = (prediction[:, -future_latents:] - (target - noise)).float().square()
        error = error.mean(dim=(2, 3, 4))
        weights = torch.linspace(
            1.0,
            late_frame_weight,
            future_latents,
            device=target.device,
            dtype=error.dtype,
        )
        return (error * weights[None]).sum() / (weights.sum() * batch)

    @torch.no_grad()
    def sample_autoregressive(
        self,
        model: CausalLatentDiT,
        context: torch.Tensor,
        actions: torch.Tensor,
        *,
        rollout_frames: int,
        integration_steps: int = 8,
        history_noise_level: float | None = 0.2,
        schedule_type: str = "linear_quadratic",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        config = model.config
        temporal = config.temporal_downsample
        if rollout_frames < 1 or rollout_frames % temporal:
            raise ValueError(f"rollout_frames must be a positive multiple of {temporal}")
        latent_steps = rollout_frames // temporal
        if actions.shape != (context.shape[0], rollout_frames):
            raise ValueError(f"Expected actions {(context.shape[0], rollout_frames)}, got {tuple(actions.shape)}")
        if history_noise_level is not None and not 0.0 <= history_noise_level < 1.0:
            raise ValueError("history_noise_level must be in [0, 1) or None")
        action_pairs = actions.reshape(context.shape[0], latent_steps, temporal)
        clean_history = context
        noisy_history = context
        history_actions = torch.zeros(
            context.shape[0],
            context.shape[1],
            temporal,
            device=actions.device,
            dtype=actions.dtype,
        )
        history_times = torch.ones(
            context.shape[0],
            context.shape[1],
            device=context.device,
            dtype=context.dtype,
        )
        generated: list[torch.Tensor] = []
        integration_times = self.inference_times(
            integration_steps,
            context.device,
            schedule_type=schedule_type,
        )
        for index in range(latent_steps):
            keep = config.max_sequence_latents - 1
            current_clean_history = clean_history[:, -keep:]
            current_noisy_history = noisy_history[:, -keep:]
            current_history_actions = history_actions[:, -keep:]
            current_history_times = history_times[:, -keep:]
            sample = torch.randn(
                context.shape[0],
                1,
                config.latent_dim,
                config.latent_grid_size,
                config.latent_grid_size,
                device=context.device,
                dtype=context.dtype,
                generator=generator,
            )
            current_action = action_pairs[:, index : index + 1]
            for start, end in zip(integration_times[:-1], integration_times[1:]):
                sequence = torch.cat((current_noisy_history, sample), dim=1)
                clean_sequence = torch.cat((current_clean_history, sample), dim=1)
                sequence_actions = torch.cat((current_history_actions, current_action), dim=1)
                times = torch.cat(
                    (
                        current_history_times,
                        start.to(dtype=context.dtype).expand(sequence.shape[0], 1),
                    ),
                    dim=1,
                )
                clean_past = model.shifted_clean_past(clean_sequence)
                velocity = model(sequence, sequence_actions, times, clean_past)
                if not isinstance(velocity, torch.Tensor):
                    velocity = velocity[0]
                sample = sample + (end - start) * velocity[:, -1:]
                sample = sample.clamp(-8.0, 8.0)
            generated.append(sample)
            clean_history = torch.cat((clean_history, sample), dim=1)
            if history_noise_level is None:
                cached_sample = sample
                cached_time = 1.0
            else:
                cached_time = 1.0 - history_noise_level
                cache_noise = torch.randn(
                    sample.shape,
                    device=sample.device,
                    dtype=sample.dtype,
                    generator=generator,
                )
                cached_sample = cached_time * sample + history_noise_level * cache_noise
            noisy_history = torch.cat((noisy_history, cached_sample), dim=1)
            history_actions = torch.cat((history_actions, current_action), dim=1)
            history_times = torch.cat(
                (
                    history_times,
                    torch.full(
                        (history_times.shape[0], 1),
                        cached_time,
                        device=history_times.device,
                        dtype=history_times.dtype,
                    ),
                ),
                dim=1,
            )
        return torch.cat(generated, dim=1)


def normalize_latents(latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (latents - mean[None, None, :, None, None]) / std[None, None, :, None, None]


def unnormalize_latents(latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return latents * std[None, None, :, None, None] + mean[None, None, :, None, None]


def build_latent_pipeline_from_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[CausalLatentDiT, RepresentationCodec, torch.Tensor, torch.Tensor]:
    if checkpoint.get("kind") != "latent_world_model":
        raise ValueError("Checkpoint is not a latent world model")
    raw_config = dict(checkpoint["model_config"])
    model = CausalLatentDiT(LatentWorldModelConfig(**raw_config))
    model.load_state_dict(checkpoint["model"])
    codec_checkpoint = checkpoint["codec_checkpoint"]
    codec = build_codec_from_checkpoint(codec_checkpoint)
    mean = torch.as_tensor(codec_checkpoint["latent_mean"]).float()
    std = torch.as_tensor(codec_checkpoint["latent_std"]).float()
    return model, codec, mean, std


def load_latent_world_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[CausalLatentDiT, RepresentationCodec, torch.Tensor, torch.Tensor, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model, codec, mean, std = build_latent_pipeline_from_checkpoint(checkpoint)
    return model, codec, mean, std, checkpoint
