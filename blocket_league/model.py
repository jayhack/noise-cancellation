from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VideoDiTConfig:
    image_size: int = 64
    patch_size: int = 8
    context_frames: int = 4
    future_frames: int = 8
    hidden_size: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    action_count: int = 9
    diffusion_steps: int = 100
    noise_schedule: str = "cosine"
    prediction_type: str = "x0"
    # "full" preserves compatibility with the first checkpoints. New runs
    # explicitly request factorized attention so 4x4 patches are affordable.
    attention_mode: str = "full"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MODEL_PRESETS: dict[str, dict[str, int]] = {
    "micro": {"hidden_size": 128, "depth": 4, "heads": 4},
    "tiny": {"hidden_size": 192, "depth": 6, "heads": 6},
    "small": {"hidden_size": 256, "depth": 8, "heads": 8},
}


def config_for_preset(name: str, **overrides: object) -> VideoDiTConfig:
    if name not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset {name!r}; choose from {sorted(MODEL_PRESETS)}")
    values = {**MODEL_PRESETS[name], **overrides}
    return VideoDiTConfig(**values)


def timestep_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -log(10_000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    phases = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(hidden_size)
        inner = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner, hidden_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attention(tokens)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + attended
        return tokens + self.mlp(self.norm_mlp(tokens))


class FactorizedTransformerBlock(nn.Module):
    """Alternating spatial and temporal attention for dense video tokens."""

    def __init__(self, hidden_size: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_spatial = nn.LayerNorm(hidden_size)
        self.spatial_attention = nn.MultiheadAttention(
            hidden_size,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_temporal = nn.LayerNorm(hidden_size)
        self.temporal_attention = nn.MultiheadAttention(
            hidden_size,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(hidden_size)
        inner = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner, hidden_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, frames, patches, hidden = tokens.shape

        spatial = self.norm_spatial(tokens).reshape(batch * frames, patches, hidden)
        spatial = self.spatial_attention(spatial, spatial, spatial, need_weights=False)[0]
        tokens = tokens + spatial.reshape(batch, frames, patches, hidden)

        temporal = self.norm_temporal(tokens).permute(0, 2, 1, 3)
        temporal = temporal.reshape(batch * patches, frames, hidden)
        temporal = self.temporal_attention(temporal, temporal, temporal, need_weights=False)[0]
        temporal = temporal.reshape(batch, patches, frames, hidden).permute(0, 2, 1, 3)
        tokens = tokens + temporal
        return tokens + self.mlp(self.norm_mlp(tokens))


class VideoDiT(nn.Module):
    """A deliberately plain, hook-friendly video diffusion transformer."""

    def __init__(self, config: VideoDiTConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if config.hidden_size % config.heads:
            raise ValueError("hidden_size must be divisible by heads")
        if config.attention_mode not in {"full", "factorized"}:
            raise ValueError("attention_mode must be 'full' or 'factorized'")
        self.config = config
        self.grid_size = config.image_size // config.patch_size
        self.tokens_per_frame = self.grid_size**2

        self.patch_embedding = nn.Conv2d(
            3,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.spatial_position = nn.Parameter(
            torch.empty(1, 1, self.tokens_per_frame, config.hidden_size)
        )
        self.frame_position = nn.Parameter(
            torch.empty(
                1,
                config.context_frames + config.future_frames,
                1,
                config.hidden_size,
            )
        )
        self.segment_embedding = nn.Parameter(torch.empty(2, config.hidden_size))
        self.action_embedding = nn.Embedding(config.action_count, config.hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
        )
        block_type = TransformerBlock if config.attention_mode == "full" else FactorizedTransformerBlock
        self.blocks = nn.ModuleList(
            block_type(config.hidden_size, config.heads, config.mlp_ratio)
            for _ in range(config.depth)
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_projection = nn.Linear(
            config.hidden_size,
            3 * config.patch_size * config.patch_size,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        nn.init.trunc_normal_(self.frame_position, std=0.02)
        nn.init.trunc_normal_(self.segment_embedding, std=0.02)
        nn.init.normal_(self.action_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.patch_embedding.weight)
        nn.init.zeros_(self.patch_embedding.bias)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)

    def _patchify(self, video: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = video.shape
        if channels != 3 or height != self.config.image_size or width != self.config.image_size:
            raise ValueError(
                f"Expected video [B, T, 3, {self.config.image_size}, {self.config.image_size}], got {video.shape}"
            )
        patches = self.patch_embedding(video.reshape(batch * frames, channels, height, width))
        patches = patches.flatten(2).transpose(1, 2)
        return patches.reshape(batch, frames, self.tokens_per_frame, self.config.hidden_size)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch, frames, _, _ = patches.shape
        patch = self.config.patch_size
        grid = self.grid_size
        pixels = self.output_projection(patches)
        pixels = pixels.reshape(batch, frames, grid, grid, 3, patch, patch)
        return pixels.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch,
            frames,
            3,
            self.config.image_size,
            self.config.image_size,
        )

    def forward(
        self,
        noisy_future: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        config = self.config
        if context.shape[1] != config.context_frames:
            raise ValueError(f"Expected {config.context_frames} context frames")
        if noisy_future.shape[1] != config.future_frames:
            raise ValueError(f"Expected {config.future_frames} future frames")
        if actions.shape != (noisy_future.shape[0], config.future_frames):
            raise ValueError(
                f"Expected actions [B, {config.future_frames}], got {tuple(actions.shape)}"
            )

        context_tokens = self._patchify(context)
        future_tokens = self._patchify(noisy_future)
        positions = self.spatial_position + self.frame_position
        context_tokens = (
            context_tokens
            + positions[:, : config.context_frames]
            + self.segment_embedding[0][None, None, None]
        )
        time_condition = self.time_mlp(timestep_embedding(timesteps, config.hidden_size))
        future_tokens = (
            future_tokens
            + positions[:, config.context_frames :]
            + self.segment_embedding[1][None, None, None]
            + self.action_embedding(actions)[:, :, None, :]
            + time_condition[:, None, None, :]
        )

        batch = noisy_future.shape[0]
        tokens = torch.cat((context_tokens, future_tokens), dim=1)
        if config.attention_mode == "full":
            tokens = tokens.reshape(batch, -1, config.hidden_size)
        hidden_states: list[torch.Tensor] = []
        for block in self.blocks:
            tokens = block(tokens)
            if return_hidden:
                hidden_states.append(tokens.reshape(batch, -1, config.hidden_size))

        if config.attention_mode == "full":
            future_start = config.context_frames * self.tokens_per_frame
            future_tokens_out = tokens[:, future_start:].reshape(
                batch,
                config.future_frames,
                self.tokens_per_frame,
                config.hidden_size,
            )
        else:
            future_tokens_out = tokens[:, config.context_frames :]
        future_output = self.output_norm(future_tokens_out)
        prediction = self._unpatchify(future_output)
        if return_hidden:
            return prediction, hidden_states
        return prediction


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        steps: int = 100,
        schedule: str = "cosine",
        prediction_type: str = "x0",
    ) -> None:
        super().__init__()
        if prediction_type not in {"epsilon", "v", "x0"}:
            raise ValueError("prediction_type must be 'epsilon', 'v', or 'x0'")
        if schedule == "cosine":
            offset = 0.008
            phase = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
            alpha_bar_points = torch.cos(
                ((phase + offset) / (1 + offset)) * torch.pi * 0.5
            ).square()
            alpha_bar_points = alpha_bar_points / alpha_bar_points[0].clone()
            beta = 1.0 - alpha_bar_points[1:] / alpha_bar_points[:-1]
            beta = beta.clamp(1e-5, 0.999).float()
        elif schedule == "linear":
            beta = torch.linspace(1e-4, 0.02, steps, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown noise schedule: {schedule}")
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.steps = steps
        self.schedule = schedule
        self.prediction_type = prediction_type
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", alpha_bar)

    def add_noise(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise
        alpha_bar = self.alpha_bar[timesteps].view(-1, 1, 1, 1, 1)
        noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
        return noisy, noise

    def training_loss(
        self,
        model: VideoDiT,
        target: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
        *,
        foreground_weight: float = 1.0,
        puck_weight: float | None = None,
        terminal_timestep_fraction: float = 0.0,
        late_frame_weight: float = 1.0,
    ) -> torch.Tensor:
        timesteps = torch.randint(0, self.steps, (target.shape[0],), device=target.device)
        if terminal_timestep_fraction > 0:
            terminal = torch.rand(target.shape[0], device=target.device) < terminal_timestep_fraction
            timesteps = torch.where(terminal, self.steps - 1, timesteps)
        noisy, noise = self.add_noise(target, timesteps)
        prediction = model(noisy, context, actions, timesteps)
        if not isinstance(prediction, torch.Tensor):
            prediction = prediction[0]
        if self.prediction_type == "epsilon":
            objective = noise
        elif self.prediction_type == "v":
            alpha_bar = self.alpha_bar[timesteps].view(-1, 1, 1, 1, 1)
            objective = alpha_bar.sqrt() * noise - (1.0 - alpha_bar).sqrt() * target
        else:
            objective = target
        squared_error = (prediction - objective).square().mean(dim=2)
        weights = torch.ones_like(squared_error)
        if foreground_weight > 1 or (puck_weight is not None and puck_weight > 1):
            # Exact palette colors make it possible to balance moving objects
            # using only RGB targets; privileged simulator state stays hidden.
            palette = torch.tensor(
                (
                    (50, 213, 173),
                    (11, 57, 52),
                    (239, 242, 233),
                    (150, 161, 153),
                ),
                device=target.device,
                dtype=target.dtype,
            ).div(127.5).sub(1.0)
            target_hwc = target.permute(0, 1, 3, 4, 2)
            distance = (target_hwc.unsqueeze(-2) - palette).square().sum(dim=-1)
            player = distance[..., :2].min(dim=-1).values < 1e-5
            puck = distance[..., 2:].min(dim=-1).values < 1e-5
            weights = weights + player.to(weights.dtype) * (foreground_weight - 1.0)
            effective_puck_weight = foreground_weight if puck_weight is None else puck_weight
            weights = weights + puck.to(weights.dtype) * (effective_puck_weight - 1.0)
        if late_frame_weight != 1:
            frame_weights = torch.linspace(
                1.0,
                late_frame_weight,
                target.shape[1],
                device=target.device,
                dtype=target.dtype,
            )
            weights = weights * frame_weights[None, :, None, None]
        return (squared_error * weights).sum() / weights.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(
        self,
        model: VideoDiT,
        context: torch.Tensor,
        actions: torch.Tensor,
        *,
        ddim_steps: int = 20,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch = context.shape[0]
        shape = (
            batch,
            model.config.future_frames,
            3,
            model.config.image_size,
            model.config.image_size,
        )
        sample = torch.randn(shape, device=context.device, generator=generator)
        indices = torch.linspace(self.steps - 1, 0, min(ddim_steps, self.steps), device=context.device)
        indices = torch.unique_consecutive(indices.round().long())

        for index, timestep_value in enumerate(indices):
            timesteps = torch.full(
                (batch,),
                int(timestep_value.item()),
                device=context.device,
                dtype=torch.long,
            )
            prediction = model(sample, context, actions, timesteps)
            if not isinstance(prediction, torch.Tensor):
                prediction = prediction[0]
            alpha_bar = self.alpha_bar[timestep_value]
            if self.prediction_type == "epsilon":
                predicted_noise = prediction
                predicted_clean = (
                    sample - (1.0 - alpha_bar).sqrt() * predicted_noise
                ) / alpha_bar.sqrt()
            elif self.prediction_type == "v":
                predicted_clean = alpha_bar.sqrt() * sample - (1.0 - alpha_bar).sqrt() * prediction
                predicted_noise = (1.0 - alpha_bar).sqrt() * sample + alpha_bar.sqrt() * prediction
            else:
                predicted_clean = prediction
                predicted_noise = (
                    sample - alpha_bar.sqrt() * predicted_clean
                ) / (1.0 - alpha_bar).sqrt().clamp_min(1e-6)
            predicted_clean = predicted_clean.clamp(-1.0, 1.0)
            if index == len(indices) - 1:
                sample = predicted_clean
                break
            next_alpha_bar = self.alpha_bar[indices[index + 1]]
            sample = (
                next_alpha_bar.sqrt() * predicted_clean
                + (1.0 - next_alpha_bar).sqrt() * predicted_noise
            )
        return sample


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
