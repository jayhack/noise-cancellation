from __future__ import annotations

import torch

from .env import PALETTE


def _soft_centroid(video: torch.Tensor, color_name: str, temperature: float = 0.04) -> tuple[torch.Tensor, torch.Tensor]:
    """Return color-weighted XY centroids and mass for [B,T,C,H,W] videos."""

    color = torch.as_tensor(
        PALETTE[color_name],
        device=video.device,
        dtype=video.dtype,
    ).div(127.5).sub(1.0)
    distance = (video - color[None, None, :, None, None]).square().sum(dim=2)
    weights = torch.exp(-distance / temperature)
    height, width = video.shape[-2:]
    x = (torch.arange(width, device=video.device, dtype=video.dtype) + 0.5) / width
    y = (torch.arange(height, device=video.device, dtype=video.dtype) + 0.5) / height
    mass = weights.sum(dim=(-2, -1)).clamp_min(1e-6)
    centroid_x = (weights * x[None, None, None, :]).sum(dim=(-2, -1)) / mass
    centroid_y = (weights * y[None, None, :, None]).sum(dim=(-2, -1)) / mass
    return torch.stack((centroid_x, centroid_y), dim=-1), mass


@torch.no_grad()
def trajectory_metrics(prediction: torch.Tensor, state: torch.Tensor) -> dict[str, float]:
    """Measure soft RGB entity trajectories against privileged evaluation state."""

    image_size = prediction.shape[-1]
    player_position, player_mass = _soft_centroid(prediction.float(), "player")
    puck_position, puck_mass = _soft_centroid(prediction.float(), "puck")
    player_error = torch.linalg.vector_norm(player_position - state[..., :2], dim=-1) * image_size
    puck_error = torch.linalg.vector_norm(puck_position - state[..., 4:6], dim=-1) * image_size
    return {
        "player_position_error_px": float(player_error.mean()),
        "puck_position_error_px": float(puck_error.mean()),
        "final_player_error_px": float(player_error[:, -1].mean()),
        "final_puck_error_px": float(puck_error[:, -1].mean()),
        "mean_position_error_px": float(torch.cat((player_error, puck_error), dim=0).mean()),
        "player_color_mass": float(player_mass.mean()),
        "puck_color_mass": float(puck_mass.mean()),
    }
