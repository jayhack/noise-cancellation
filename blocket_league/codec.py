from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


CODEC_PALETTE = (
    (7, 11, 16),
    (11, 28, 31),
    (31, 64, 65),
    (81, 103, 105),
    (238, 181, 62),
    (50, 213, 173),
    (11, 57, 52),
    (239, 242, 233),
    (150, 161, 153),
)


@dataclass(frozen=True)
class RepresentationCodecConfig:
    """A Blocket-scale version of MIRA's RAEv2 video codec."""

    image_size: int = 64
    # Upsampling the 64px game frame to an 8x8 DINO patch grid preserves the
    # three-pixel puck. MIRA can downsample more aggressively because its
    # rendered objects occupy many more source pixels.
    dino_input_size: int = 112
    dino_patch_size: int = 14
    dino_hidden_size: int = 384
    dino_layers: tuple[int, ...] = (3, 6, 9, 12)
    latent_dim: int = 32
    temporal_downsample: int = 2
    decoder_width: int = 160
    decoder_depth: int = 5
    decoder_heads: int = 5
    max_latent_frames: int = 32

    @property
    def latent_grid_size(self) -> int:
        return self.dino_input_size // self.dino_patch_size

    @property
    def spatial_downsample(self) -> int:
        return self.image_size // self.latent_grid_size

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DinoRepresentationEncoder(nn.Module):
    """Frozen DINOv2 features followed by a learned temporal bottleneck."""

    def __init__(self, config: RepresentationCodecConfig, backbone: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.projection = nn.Conv3d(
            config.dino_hidden_size,
            config.latent_dim,
            kernel_size=(config.temporal_downsample, 1, 1),
            stride=(config.temporal_downsample, 1, 1),
        )
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406))[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225))[None, :, None, None],
            persistent=False,
        )

    def train(self, mode: bool = True) -> DinoRepresentationEncoder:
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, video: torch.Tensor, *, input_grad: bool = False) -> torch.Tensor:
        """Return aggregated DINO patch features as [B,T,C,H,W]."""

        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(f"Expected [B,T,3,H,W] video, got {tuple(video.shape)}")
        batch, frames = video.shape[:2]
        pixels = video.reshape(batch * frames, 3, video.shape[-2], video.shape[-1])
        pixels = F.interpolate(
            pixels.float().add(1).mul(0.5),
            size=(self.config.dino_input_size, self.config.dino_input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        pixels = (pixels - self.image_mean) / self.image_std

        context = torch.enable_grad() if input_grad else torch.no_grad()
        with context:
            outputs = self.backbone(
                pixel_values=pixels,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("DINO backbone did not return hidden states")
            selected = [hidden_states[index] for index in self.config.dino_layers]
            aggregate = torch.stack(selected).mean(dim=0) + selected[-1]

        grid = self.config.latent_grid_size
        patch_count = grid * grid
        patches = aggregate[:, 1 : patch_count + 1]
        patches = patches.transpose(1, 2).reshape(
            batch,
            frames,
            self.config.dino_hidden_size,
            grid,
            grid,
        )
        return patches

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(video, input_grad=False)
        projected = self.projection(features.permute(0, 2, 1, 3, 4))
        return projected.permute(0, 2, 1, 3, 4), features


class CodecFactorizedBlock(nn.Module):
    """Bidirectional spatial attention and causal temporal attention."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(width)
        self.spatial_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(width * 4, width),
        )
        self.spatial_scale = nn.Parameter(torch.full((width,), 1e-3))
        self.temporal_scale = nn.Parameter(torch.full((width,), 1e-3))
        self.mlp_scale = nn.Parameter(torch.full((width,), 1e-3))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, frames, patches, width = tokens.shape
        spatial = self.spatial_norm(tokens).reshape(batch * frames, patches, width)
        spatial = self.spatial_attention(spatial, spatial, spatial, need_weights=False)[0]
        tokens = tokens + spatial.reshape(batch, frames, patches, width) * self.spatial_scale

        temporal = self.temporal_norm(tokens).permute(0, 2, 1, 3)
        temporal = temporal.reshape(batch * patches, frames, width)
        causal_mask = torch.ones(frames, frames, device=tokens.device, dtype=torch.bool).triu(1)
        temporal = self.temporal_attention(
            temporal,
            temporal,
            temporal,
            attn_mask=causal_mask,
            need_weights=False,
        )[0]
        temporal = temporal.reshape(batch, patches, frames, width).permute(0, 2, 1, 3)
        tokens = tokens + temporal * self.temporal_scale
        return tokens + self.mlp(self.mlp_norm(tokens)) * self.mlp_scale


class CausalVideoDecoder(nn.Module):
    """Small ViT decoder that expands one latent into two RGB frames."""

    def __init__(self, config: RepresentationCodecConfig) -> None:
        super().__init__()
        self.config = config
        grid = config.latent_grid_size
        self.input_projection = nn.Conv2d(config.latent_dim, config.decoder_width, kernel_size=1)
        self.spatial_position = nn.Parameter(
            torch.randn(1, 1, grid * grid, config.decoder_width) * 0.02
        )
        self.temporal_position = nn.Parameter(
            torch.randn(1, config.max_latent_frames, 1, config.decoder_width) * 0.02
        )
        self.blocks = nn.ModuleList(
            CodecFactorizedBlock(config.decoder_width, config.decoder_heads)
            for _ in range(config.decoder_depth)
        )
        self.output_norm = nn.LayerNorm(config.decoder_width)
        feature_channels = 64
        self.to_feature = nn.Linear(
            config.decoder_width,
            config.temporal_downsample * feature_channels,
        )
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(feature_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, len(CODEC_PALETTE), kernel_size=3, padding=1),
        )
        self.register_buffer(
            "palette",
            torch.tensor(CODEC_PALETTE, dtype=torch.float32).div(127.5).sub(1.0),
        )

    @property
    def last_layer_weight(self) -> torch.Tensor:
        return self.upsampler[-1].weight

    def logits(self, latents: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = latents.shape
        grid = self.config.latent_grid_size
        if channels != self.config.latent_dim or height != grid or width != grid:
            raise ValueError(
                f"Expected [B,T,{self.config.latent_dim},{grid},{grid}] latents, "
                f"got {tuple(latents.shape)}"
            )
        if frames > self.config.max_latent_frames:
            raise ValueError(f"Decoder supports at most {self.config.max_latent_frames} latent frames")

        tokens = self.input_projection(latents.reshape(batch * frames, channels, grid, grid))
        tokens = tokens.flatten(2).transpose(1, 2).reshape(
            batch,
            frames,
            grid * grid,
            self.config.decoder_width,
        )
        tokens = tokens + self.spatial_position + self.temporal_position[:, :frames]
        for block in self.blocks:
            tokens = block(tokens)
        features = self.to_feature(self.output_norm(tokens))
        temporal = self.config.temporal_downsample
        features = features.reshape(batch, frames, grid, grid, temporal, 64)
        features = features.permute(0, 1, 4, 5, 2, 3).reshape(
            batch * frames * temporal,
            64,
            grid,
            grid,
        )
        logits = self.upsampler(features)
        return logits.reshape(
            batch,
            frames * temporal,
            len(CODEC_PALETTE),
            self.config.image_size,
            self.config.image_size,
        )

    def forward(self, latents: torch.Tensor, *, hard: bool = True) -> torch.Tensor:
        logits = self.logits(latents)
        if hard:
            classes = logits.argmax(dim=2)
            return self.palette[classes].permute(0, 1, 4, 2, 3)
        probabilities = logits.float().softmax(dim=2)
        return torch.einsum("btphw,pc->btchw", probabilities, self.palette)


class RepresentationCodec(nn.Module):
    def __init__(self, config: RepresentationCodecConfig, backbone: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.encoder = DinoRepresentationEncoder(config, backbone)
        self.decoder = CausalVideoDecoder(config)

    def encode(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(video)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents, hard=True)

    def decode_soft(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.decoder.logits(latents)
        probabilities = logits.float().softmax(dim=2)
        reconstruction = torch.einsum("btphw,pc->btchw", probabilities, self.decoder.palette)
        return reconstruction, logits

    def forward(
        self,
        video: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latents, features = self.encode(video)
        reconstruction, logits = self.decode_soft(latents)
        return reconstruction, latents, features, logits


def load_pretrained_dinov2(model_name: str = "facebook/dinov2-small") -> tuple[nn.Module, dict[str, Any]]:
    from transformers import Dinov2Model

    backbone = Dinov2Model.from_pretrained(model_name)
    return backbone, backbone.config.to_dict()


def build_codec_from_checkpoint(checkpoint: dict[str, Any]) -> RepresentationCodec:
    from transformers import Dinov2Config, Dinov2Model

    raw_config = dict(checkpoint["codec_config"])
    raw_config["dino_layers"] = tuple(raw_config["dino_layers"])
    config = RepresentationCodecConfig(**raw_config)
    backbone_config = Dinov2Config.from_dict(checkpoint["backbone_config"])
    pretrained_backbone = checkpoint.get("pretrained_backbone")
    backbone = (
        Dinov2Model.from_pretrained(str(pretrained_backbone))
        if pretrained_backbone
        else Dinov2Model(backbone_config)
    )
    codec = RepresentationCodec(config, backbone)
    incompatible = codec.load_state_dict(checkpoint["model"], strict=not pretrained_backbone)
    if pretrained_backbone:
        invalid_missing = [
            key for key in incompatible.missing_keys if not key.startswith("encoder.backbone.")
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise ValueError(
                "Slim codec checkpoint has incompatible learned weights: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
    return codec


def load_codec_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> tuple[RepresentationCodec, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("kind") != "representation_codec":
        raise ValueError(f"Not a representation codec checkpoint: {path}")
    return build_codec_from_checkpoint(checkpoint), checkpoint


def fake_backbone_outputs(hidden_states: tuple[torch.Tensor, ...]) -> SimpleNamespace:
    """Tiny helper used by unit tests without importing transformers."""

    return SimpleNamespace(hidden_states=hidden_states)
