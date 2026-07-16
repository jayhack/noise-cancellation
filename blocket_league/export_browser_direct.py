from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from .data import make_clip
from .direct_model import DirectLatentTransformer, build_direct_pipeline_from_checkpoint
from .export_browser import BrowserDecoder
from .latent_model import normalize_latents


class BrowserDirectDynamics(nn.Module):
    def __init__(self, model: DirectLatentTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, latent_history: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.model.next_latent(latent_history, actions).clamp(-8.0, 8.0)


def _video_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value.copy()).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)


def export_direct_graphs(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    seed: int,
    opset: int,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, codec, mean, std = build_direct_pipeline_from_checkpoint(checkpoint)
    model.eval()
    codec.eval()
    config = model.config
    output_dir.mkdir(parents=True, exist_ok=True)
    dynamics_path = output_dir / "dynamics.onnx"
    decoder_path = output_dir / "decoder.onnx"

    dynamics = BrowserDirectDynamics(model).eval()
    history = torch.randn(
        1,
        config.history_latents,
        config.latent_dim,
        config.latent_grid_size,
        config.latent_grid_size,
    )
    actions = torch.zeros(
        1,
        config.history_latents,
        config.temporal_downsample,
        dtype=torch.long,
    )
    torch.onnx.export(
        dynamics,
        (history, actions),
        dynamics_path,
        input_names=("latent_history", "actions"),
        output_names=("next_latent",),
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    decode_latents = config.history_latents + 1
    decoder = BrowserDecoder(codec.decoder, mean, std).eval()
    decode_input = torch.randn(
        1,
        decode_latents,
        config.latent_dim,
        config.latent_grid_size,
        config.latent_grid_size,
    )
    torch.onnx.export(
        decoder,
        (decode_input,),
        decoder_path,
        input_names=("normalized_sequence",),
        output_names=("logits",),
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # Generate 16 starter frames while retaining the seven action pairs that
    # connect the eight compressed latents. The eighth pair is supplied live.
    clip = make_clip(seed, context_frames=2, future_frames=config.history_latents * 2 - 2)
    video = _video_tensor(np.concatenate((clip["context"], clip["target"]))).unsqueeze(0)
    with torch.inference_mode():
        raw_context, _ = codec.encode(video)
        context_latents = normalize_latents(raw_context.float(), mean, std)
    context_latents.numpy().astype("<f4", copy=False).tofile(output_dir / "starter-context.bin")
    clip["actions"].astype("<i8", copy=False).tofile(output_dir / "starter-actions.bin")
    Image.fromarray(clip["frames"][-1]).save(output_dir / "starter-frame.png", optimize=True)

    manifest: dict[str, object] = {
        "version": 2,
        "modelKind": "direct-autoregressive",
        "checkpointStep": int(checkpoint.get("step", 0)),
        "seed": seed,
        "sourceFps": 20,
        "frameSize": codec.config.image_size,
        "latentChannels": config.latent_dim,
        "latentGrid": config.latent_grid_size,
        "historyLatents": config.history_latents,
        "sequenceLatents": decode_latents,
        "temporalDownsample": config.temporal_downsample,
        "historyNoiseLevel": 0.0,
        "defaultIntegrationSteps": 1,
        "modelParameters": sum(parameter.numel() for parameter in model.parameters()),
        "decoderParameters": sum(parameter.numel() for parameter in codec.decoder.parameters()),
        "modelBytes": dynamics_path.stat().st_size,
        "decoderBytes": decoder_path.stat().st_size,
        "palette": [
            list(color)
            for color in codec.decoder.palette.mul(127.5).add(127.5).round().byte().tolist()
        ],
        "assets": {
            "dynamics": "/blocket-league/live/dynamics.onnx",
            "decoder": "/blocket-league/live/decoder.onnx",
            "starterContext": "/blocket-league/live/starter-context.bin",
            "starterActions": "/blocket-league/live/starter-actions.bin",
            "starterFrame": "/blocket-league/live/starter-frame.png",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_direct_export(output_dir: Path, checkpoint_path: Path) -> dict[str, float]:
    import onnxruntime as ort

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, codec, mean, std = build_direct_pipeline_from_checkpoint(checkpoint)
    model.eval()
    decoder = BrowserDecoder(codec.decoder.eval(), mean, std).eval()
    generator = torch.Generator().manual_seed(81_311)
    history = torch.randn(1, model.config.history_latents, 32, 8, 8, generator=generator)
    actions = torch.randint(0, 9, (1, model.config.history_latents, 2), generator=generator)
    sequence = torch.randn(1, model.config.history_latents + 1, 32, 8, 8, generator=generator)
    dynamics_session = ort.InferenceSession(
        str(output_dir / "dynamics.onnx"), providers=("CPUExecutionProvider",)
    )
    decoder_session = ort.InferenceSession(
        str(output_dir / "decoder.onnx"), providers=("CPUExecutionProvider",)
    )
    with torch.inference_mode():
        expected_latent = model.next_latent(history, actions).clamp(-8.0, 8.0).numpy()
        expected_logits = decoder(sequence).numpy()
    actual_latent = dynamics_session.run(
        None, {"latent_history": history.numpy(), "actions": actions.numpy()}
    )[0]
    actual_logits = decoder_session.run(None, {"normalized_sequence": sequence.numpy()})[0]
    return {
        "dynamicsMaxAbsoluteError": float(np.max(np.abs(expected_latent - actual_latent))),
        "dynamicsMeanAbsoluteError": float(np.mean(np.abs(expected_latent - actual_latent))),
        "decoderMaxAbsoluteError": float(np.max(np.abs(expected_logits - actual_logits))),
        "decoderMeanAbsoluteError": float(np.mean(np.abs(expected_logits - actual_logits))),
        "decoderClassAgreement": float(
            np.mean(expected_logits.argmax(axis=2) == actual_logits.argmax(axis=2))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the direct latent transformer for browser play")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=23_117)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = export_direct_graphs(checkpoint, output, seed=args.seed, opset=args.opset)
    if not args.skip_validation:
        manifest["validation"] = validate_direct_export(output, checkpoint)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
