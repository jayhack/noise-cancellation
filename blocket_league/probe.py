from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import ClipDataset, STATE_NAMES
from .model import DiffusionSchedule, VideoDiT, VideoDiTConfig


@torch.no_grad()
def collect_layer_features(
    checkpoint_path: Path,
    *,
    samples: int = 512,
    batch_size: int = 16,
    seed: int = 90_001,
) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, object]]:
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
    dataset = ClipDataset(
        samples,
        seed=seed,
        context_frames=model_config.context_frames,
        future_frames=model_config.future_frames,
        image_size=model_config.image_size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    features: list[list[torch.Tensor]] = [[] for _ in range(model_config.depth)]
    targets: list[torch.Tensor] = []
    for batch in loader:
        context = batch["context"].to(device)
        target = batch["target"].to(device)
        actions = batch["actions"].to(device)
        probe_timestep = model_config.diffusion_steps - 1
        timesteps = torch.full(
            (context.shape[0],),
            probe_timestep,
            device=device,
            dtype=torch.long,
        )
        noisy, _ = schedule.add_noise(target, timesteps)
        result = model(noisy, context, actions, timesteps, return_hidden=True)
        if isinstance(result, torch.Tensor):
            raise RuntimeError("Model did not return hidden states")
        _, hidden_states = result
        for layer, hidden in enumerate(hidden_states):
            # Preserve the spatial layout: mean-pooling makes an object's
            # location nearly undecodable by construction. The final future
            # frame is the one paired with the privileged state target below.
            final_frame = hidden[:, -model.tokens_per_frame :]
            features[layer].append(final_frame.flatten(1).float().cpu())
        targets.append(batch["state"][:, -1, :8].float())

    return (
        [torch.cat(layer_features) for layer_features in features],
        torch.cat(targets),
        {
            "checkpoint": str(checkpoint_path),
            "samples": samples,
            "probe_timestep": probe_timestep,
            "probe_features": "flattened final-future-frame tokens",
            "state_names": list(STATE_NAMES[:8]),
        },
    )


def ridge_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    train_fraction: float = 0.75,
    ridge: float = 1e-2,
) -> dict[str, object]:
    split = max(2, int(len(features) * train_fraction))
    x_train, x_test = features[:split], features[split:]
    y_train, y_test = targets[:split], targets[split:]
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_scale = x_train.std(dim=0, keepdim=True).clamp_min(1e-5)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_scale = y_train.std(dim=0, keepdim=True).clamp_min(1e-5)
    x_train = (x_train - x_mean) / x_scale
    x_test = (x_test - x_mean) / x_scale
    y_train_normalized = (y_train - y_mean) / y_scale
    if x_train.shape[1] > x_train.shape[0]:
        # The spatially faithful representation is wide (64 tokens × hidden
        # width), so solve ridge regression in sample space instead of forming
        # an 8k × 8k matrix.
        identity = torch.eye(x_train.shape[0])
        dual_weights = torch.linalg.solve(
            x_train @ x_train.T + ridge * identity,
            y_train_normalized,
        )
        prediction_normalized = (x_test @ x_train.T) @ dual_weights
    else:
        identity = torch.eye(x_train.shape[1])
        weights = torch.linalg.solve(
            x_train.T @ x_train + ridge * identity,
            x_train.T @ y_train_normalized,
        )
        prediction_normalized = x_test @ weights
    prediction = prediction_normalized * y_scale + y_mean
    residual = ((y_test - prediction) ** 2).sum(dim=0)
    total = ((y_test - y_test.mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp_min(1e-8)
    r2 = 1.0 - residual / total
    return {
        "mean_r2": float(r2.mean()),
        "r2": {name: float(value) for name, value in zip(STATE_NAMES[:8], r2)},
    }


def run_probes(
    checkpoint_path: Path,
    output_path: Path,
    *,
    samples: int = 512,
    batch_size: int = 16,
) -> dict[str, object]:
    features, targets, metadata = collect_layer_features(
        checkpoint_path,
        samples=samples,
        batch_size=batch_size,
    )
    layers = [ridge_probe(layer, targets) for layer in features]
    result = {**metadata, "layers": layers}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit layerwise linear probes to Blocket League state")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("probe-results.json"))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(
        json.dumps(
            run_probes(
                args.checkpoint,
                args.output,
                samples=args.samples,
                batch_size=args.batch_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
