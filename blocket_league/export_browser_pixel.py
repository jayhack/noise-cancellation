from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from .data import make_passive_clip
from .pixel_direct_model import DirectPixelTransformer, build_pixel_direct_from_checkpoint
from .train_pixel_direct import frames_to_classes, palette_tensor


class BrowserPixelDynamics(nn.Module):
    def __init__(self, model: DirectPixelTransformer, block_index: int) -> None:
        super().__init__()
        self.model = model
        self.block_index = block_index

    def forward(
        self,
        pixel_history: torch.Tensor,
        intervention: torch.Tensor,
        intervention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            pixel_history,
            intervention_block=self.block_index,
            intervention=intervention,
            intervention_mask=intervention_mask,
        )[:, -1]


def export_pixel_graph(
    checkpoint_path: Path,
    interpretation_path: Path,
    output_dir: Path,
    *,
    seed: int,
    opset: int,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_pixel_direct_from_checkpoint(checkpoint).eval()
    interpretation = json.loads(interpretation_path.read_text(encoding="utf-8"))
    causal = interpretation["causal"]
    block_index = int(causal["block"]) - 1
    config = model.config
    output_dir.mkdir(parents=True, exist_ok=True)
    dynamics_path = output_dir / "pixel-dynamics.onnx"

    wrapper = BrowserPixelDynamics(model, block_index).eval()
    history = torch.randint(
        0,
        config.palette_size,
        (1, config.history_frames, config.image_size, config.image_size),
        dtype=torch.long,
    ).float()
    intervention = torch.zeros(1, config.hidden_size)
    mask = torch.zeros(1, config.history_frames, config.grid_size**2)
    torch.onnx.export(
        wrapper,
        (history, intervention, mask),
        dynamics_path,
        input_names=("pixel_history", "intervention", "intervention_mask"),
        output_names=("next_logits",),
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    clip = make_passive_clip(
        seed,
        context_frames=config.history_frames,
        future_frames=1,
        image_size=config.image_size,
    )
    video = torch.from_numpy(clip["context"].copy()).permute(0, 3, 1, 2).float()
    classes = frames_to_classes(video[None].div(127.5).sub(1.0), palette_tensor(torch.device("cpu")))
    classes.numpy().astype("<f4", copy=False).tofile(output_dir / "starter-context.bin")
    Image.fromarray(clip["context"][-1]).save(output_dir / "starter-frame.png", optimize=True)

    palette = np.asarray(checkpoint["palette"], dtype=np.uint8)
    manifest: dict[str, object] = {
        "version": 3,
        "modelKind": "passive-direct-pixel-autoregressive",
        "checkpointStep": int(checkpoint["step"]),
        "seed": seed,
        "sourceFps": 20,
        "frameSize": config.image_size,
        "patchSize": config.patch_size,
        "gridSize": config.grid_size,
        "historyFrames": config.history_frames,
        "hiddenSize": config.hidden_size,
        "interventionBlock": block_index + 1,
        "interventionStrength": float(causal["strength"]),
        "modelParameters": sum(parameter.numel() for parameter in model.parameters()),
        "modelBytes": dynamics_path.stat().st_size,
        "palette": palette.tolist(),
        "directions": {
            "x": causal["xDirection"],
            "y": causal["yDirection"],
        },
        "assets": {
            "dynamics": "/blocket-league/live/pixel-dynamics.onnx",
            "starterContext": "/blocket-league/live/starter-context.bin",
            "starterFrame": "/blocket-league/live/starter-frame.png",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_pixel_export(output_dir: Path, checkpoint_path: Path, manifest: dict[str, object]) -> dict[str, float]:
    import onnxruntime as ort

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_pixel_direct_from_checkpoint(checkpoint).eval()
    block_index = int(manifest["interventionBlock"]) - 1
    wrapper = BrowserPixelDynamics(model, block_index).eval()
    generator = torch.Generator().manual_seed(81_311)
    history = torch.randint(
        0,
        model.config.palette_size,
        (1, model.config.history_frames, model.config.image_size, model.config.image_size),
        generator=generator,
    ).float()
    intervention = torch.randn(1, model.config.hidden_size, generator=generator)
    mask = torch.zeros(1, model.config.history_frames, model.config.grid_size**2)
    mask[:, -1, 19] = 1
    session = ort.InferenceSession(
        str(output_dir / "pixel-dynamics.onnx"), providers=("CPUExecutionProvider",)
    )
    with torch.inference_mode():
        expected = wrapper(history, intervention, mask).numpy()
    actual = session.run(
        None,
        {
            "pixel_history": history.numpy().astype(np.float32),
            "intervention": intervention.numpy(),
            "intervention_mask": mask.numpy(),
        },
    )[0]
    return {
        "maxAbsoluteError": float(np.max(np.abs(expected - actual))),
        "meanAbsoluteError": float(np.mean(np.abs(expected - actual))),
        "classAgreement": float(np.mean(expected.argmax(axis=1) == actual.argmax(axis=1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the passive pixel transformer for browser steering")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("interpretation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=23_117)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    interpretation = args.interpretation.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = export_pixel_graph(
        checkpoint,
        interpretation,
        output,
        seed=args.seed,
        opset=args.opset,
    )
    if not args.skip_validation:
        manifest["validation"] = validate_pixel_export(output, checkpoint, manifest)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
