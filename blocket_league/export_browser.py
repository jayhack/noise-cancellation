from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from .data import make_clip
from .latent_model import (
    CausalLatentDiT,
    normalize_latents,
    unnormalize_latents,
    build_latent_pipeline_from_checkpoint,
)


class BrowserDynamics(nn.Module):
    """Fixed-window world-model graph with only the newest velocity exposed."""

    def __init__(self, model: CausalLatentDiT) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        noisy_sequence: torch.Tensor,
        actions: torch.Tensor,
        times: torch.Tensor,
        clean_sequence: torch.Tensor,
    ) -> torch.Tensor:
        clean_past = self.model.shifted_clean_past(clean_sequence)
        velocity = self.model(noisy_sequence, actions, times, clean_past)
        if not isinstance(velocity, torch.Tensor):
            velocity = velocity[0]
        return velocity[:, -1]


class BrowserDecoder(nn.Module):
    """Decode the newest latent pair and leave palette argmax to the browser."""

    def __init__(
        self,
        decoder: nn.Module,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.register_buffer("latent_mean", mean[None, None, :, None, None])
        self.register_buffer("latent_std", std[None, None, :, None, None])

    def forward(self, normalized_sequence: torch.Tensor) -> torch.Tensor:
        raw_sequence = unnormalize_latents(
            normalized_sequence,
            self.latent_mean[0, 0, :, 0, 0],
            self.latent_std[0, 0, :, 0, 0],
        )
        return self.decoder.logits(raw_sequence)[:, -2:]


def _video_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value.copy()).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)


def _export_graphs(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    seed: int,
    opset: int,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, codec, mean, std = build_latent_pipeline_from_checkpoint(checkpoint)
    model.eval()
    codec.eval()

    config = model.config
    window = config.max_sequence_latents
    if window != 9:
        raise ValueError(f"The browser player expects a nine-latent window, got {window}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dynamics_path = output_dir / "dynamics.onnx"
    decoder_path = output_dir / "decoder.onnx"

    dynamics = BrowserDynamics(model).eval()
    noisy = torch.randn(1, window, config.latent_dim, config.latent_grid_size, config.latent_grid_size)
    actions = torch.zeros(1, window, config.temporal_downsample, dtype=torch.long)
    times = torch.ones(1, window)
    clean = torch.randn_like(noisy)
    torch.onnx.export(
        dynamics,
        (noisy, actions, times, clean),
        dynamics_path,
        input_names=("noisy_sequence", "actions", "times", "clean_sequence"),
        output_names=("velocity",),
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    decoder = BrowserDecoder(codec.decoder, mean, std).eval()
    normalized = torch.randn_like(noisy)
    torch.onnx.export(
        decoder,
        (normalized,),
        decoder_path,
        input_names=("normalized_sequence",),
        output_names=("logits",),
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    context_frames = (window - 1) * config.temporal_downsample
    clip = make_clip(
        seed,
        context_frames=context_frames,
        future_frames=0,
        image_size=codec.config.image_size,
    )
    context_video = _video_tensor(clip["context"]).unsqueeze(0)
    with torch.inference_mode():
        raw_context, _ = codec.encode(context_video)
        context_latents = normalize_latents(raw_context.float(), mean, std)
    expected_context_shape = (
        1,
        window - 1,
        config.latent_dim,
        config.latent_grid_size,
        config.latent_grid_size,
    )
    if tuple(context_latents.shape) != expected_context_shape:
        raise RuntimeError(
            f"Expected starter context {expected_context_shape}, got {tuple(context_latents.shape)}"
        )
    context_latents.numpy().astype("<f4", copy=False).tofile(output_dir / "starter-context.bin")
    Image.fromarray(clip["context"][-1]).save(output_dir / "starter-frame.png", optimize=True)

    manifest: dict[str, object] = {
        "version": 1,
        "checkpointStep": int(checkpoint.get("step", 0)),
        "seed": seed,
        "sourceFps": 20,
        "frameSize": codec.config.image_size,
        "latentChannels": config.latent_dim,
        "latentGrid": config.latent_grid_size,
        "historyLatents": window - 1,
        "sequenceLatents": window,
        "temporalDownsample": config.temporal_downsample,
        "historyNoiseLevel": 0.2,
        "defaultIntegrationSteps": 6,
        "modelParameters": sum(parameter.numel() for parameter in model.parameters()),
        "decoderParameters": sum(parameter.numel() for parameter in codec.decoder.parameters()),
        "modelBytes": dynamics_path.stat().st_size,
        "decoderBytes": decoder_path.stat().st_size,
        "palette": [list(color) for color in codec.decoder.palette.mul(127.5).add(127.5).round().byte().tolist()],
        "assets": {
            "dynamics": "/blocket-league/live/dynamics.onnx",
            "decoder": "/blocket-league/live/decoder.onnx",
            "starterContext": "/blocket-league/live/starter-context.bin",
            "starterFrame": "/blocket-league/live/starter-frame.png",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _validate(output_dir: Path, checkpoint_path: Path) -> dict[str, float]:
    import onnxruntime as ort

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, codec, mean, std = build_latent_pipeline_from_checkpoint(checkpoint)
    dynamics = BrowserDynamics(model.eval()).eval()
    decoder = BrowserDecoder(codec.decoder, mean, std).eval()
    generator = torch.Generator().manual_seed(81_311)
    noisy = torch.randn(1, 9, 32, 8, 8, generator=generator)
    actions = torch.randint(0, 9, (1, 9, 2), generator=generator)
    times = torch.rand(1, 9, generator=generator)
    clean = torch.randn(1, 9, 32, 8, 8, generator=generator)

    dynamics_session = ort.InferenceSession(
        str(output_dir / "dynamics.onnx"), providers=("CPUExecutionProvider",)
    )
    dynamics_inputs = {
        "noisy_sequence": noisy.numpy(),
        "actions": actions.numpy(),
        "times": times.numpy(),
        "clean_sequence": clean.numpy(),
    }
    with torch.inference_mode():
        expected_velocity = dynamics(noisy, actions, times, clean).numpy()
    actual_velocity = dynamics_session.run(None, dynamics_inputs)[0]

    decoder_session = ort.InferenceSession(
        str(output_dir / "decoder.onnx"), providers=("CPUExecutionProvider",)
    )
    with torch.inference_mode():
        expected_logits = decoder(clean).numpy()
    actual_logits = decoder_session.run(None, {"normalized_sequence": clean.numpy()})[0]
    return {
        "dynamicsMaxAbsoluteError": float(np.max(np.abs(expected_velocity - actual_velocity))),
        "dynamicsMeanAbsoluteError": float(np.mean(np.abs(expected_velocity - actual_velocity))),
        "decoderMaxAbsoluteError": float(np.max(np.abs(expected_logits - actual_logits))),
        "decoderMeanAbsoluteError": float(np.mean(np.abs(expected_logits - actual_logits))),
        "decoderClassAgreement": float(
            np.mean(expected_logits.argmax(axis=2) == actual_logits.argmax(axis=2))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the frozen latent model for browser play")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=23_117)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    manifest = _export_graphs(
        args.checkpoint.expanduser().resolve(),
        args.output.expanduser().resolve(),
        seed=args.seed,
        opset=args.opset,
    )
    if not args.skip_validation:
        manifest["validation"] = _validate(
            args.output.expanduser().resolve(), args.checkpoint.expanduser().resolve()
        )
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
