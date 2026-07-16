from __future__ import annotations

import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import modal


APP_NAME = "blocket-league-world-model"
HERE = Path(__file__).resolve().parent
REMOTE_PROJECT = "/opt/blocket"
REMOTE_RESULTS = "/results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "numpy==2.2.6",
        "pillow==11.2.1",
        "torch==2.7.1",
        "transformers==4.53.0",
    )
    .env({"PYTHONPATH": REMOTE_PROJECT})
    .add_local_dir(HERE, f"{REMOTE_PROJECT}/blocket_league", copy=True)
)
results = modal.Volume.from_name("blocket-league-results", create_if_missing=True)
app = modal.App(APP_NAME)


def _run_training(
    job_id: str,
    train_config: dict[str, object],
    probe_samples: int,
    init_checkpoint_bytes: bytes | None,
) -> dict[str, object]:
    sys.path.insert(0, REMOTE_PROJECT)
    from blocket_league.probe import run_probes
    from blocket_league.train import TrainConfig, train

    output_dir = Path(REMOTE_RESULTS) / job_id
    if init_checkpoint_bytes:
        init_checkpoint = Path("/tmp/init-checkpoint.pt")
        init_checkpoint.write_bytes(init_checkpoint_bytes)
        train_config = {**train_config, "init_checkpoint_path": str(init_checkpoint)}
    config = TrainConfig(output_dir=str(output_dir), **train_config)
    summary = train(config)
    if probe_samples > 0:
        summary["probes"] = run_probes(
            output_dir / "checkpoint.pt",
            output_dir / "probes.json",
            samples=probe_samples,
            batch_size=min(config.batch_size, 16),
        )
        summary["artifacts"].append("probes.json")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    results.commit()
    return summary


def _run_codec_training(
    job_id: str,
    train_config: dict[str, object],
    init_checkpoint_bytes: bytes | None,
) -> dict[str, object]:
    sys.path.insert(0, REMOTE_PROJECT)
    from blocket_league.train_codec import CodecTrainConfig, train_codec

    output_dir = Path(REMOTE_RESULTS) / job_id
    if init_checkpoint_bytes:
        init_checkpoint = Path("/tmp/init-codec.pt")
        init_checkpoint.write_bytes(init_checkpoint_bytes)
        train_config = {**train_config, "init_checkpoint_path": str(init_checkpoint)}
    summary = train_codec(CodecTrainConfig(output_dir=str(output_dir), **train_config))
    results.commit()
    return summary


def _run_latent_training(
    job_id: str,
    train_config: dict[str, object],
    codec_checkpoint_bytes: bytes,
) -> dict[str, object]:
    sys.path.insert(0, REMOTE_PROJECT)
    from blocket_league.train_latent import LatentTrainConfig, train_latent

    codec_checkpoint = Path("/tmp/codec.pt")
    codec_checkpoint.write_bytes(codec_checkpoint_bytes)
    output_dir = Path(REMOTE_RESULTS) / job_id
    summary = train_latent(
        LatentTrainConfig(
            output_dir=str(output_dir),
            codec_checkpoint_path=str(codec_checkpoint),
            **train_config,
        )
    )
    results.commit()
    return summary


def _run_direct_training(
    job_id: str,
    train_config: dict[str, object],
    codec_checkpoint_bytes: bytes,
) -> dict[str, object]:
    sys.path.insert(0, REMOTE_PROJECT)
    from blocket_league.train_direct import DirectTrainConfig, train_direct

    codec_checkpoint = Path("/tmp/codec.pt")
    codec_checkpoint.write_bytes(codec_checkpoint_bytes)
    output_dir = Path(REMOTE_RESULTS) / job_id
    summary = train_direct(
        DirectTrainConfig(
            output_dir=str(output_dir),
            codec_checkpoint_path=str(codec_checkpoint),
            **train_config,
        )
    )
    results.commit()
    return summary


def _run_pixel_direct_training(job_id: str, train_config: dict[str, object]) -> dict[str, object]:
    sys.path.insert(0, REMOTE_PROJECT)
    from blocket_league.train_pixel_direct import PixelDirectTrainConfig, train_pixel_direct

    output_dir = Path(REMOTE_RESULTS) / job_id
    summary = train_pixel_direct(PixelDirectTrainConfig(output_dir=str(output_dir), **train_config))
    results.commit()
    return summary


@app.function(
    image=image,
    gpu="L4",
    cpu=8.0,
    memory=24_576,
    timeout=4 * 60 * 60,
    single_use_containers=True,
    volumes={REMOTE_RESULTS: results},
)
def train_remote_l4(
    job_id: str,
    train_config: dict[str, object],
    probe_samples: int,
    init_checkpoint_bytes: bytes | None,
) -> dict[str, object]:
    return _run_training(job_id, train_config, probe_samples, init_checkpoint_bytes)


@app.function(
    image=image,
    gpu="A100",
    cpu=12.0,
    memory=49_152,
    timeout=4 * 60 * 60,
    single_use_containers=True,
    volumes={REMOTE_RESULTS: results},
)
def train_remote_a100(
    job_id: str,
    train_config: dict[str, object],
    probe_samples: int,
    init_checkpoint_bytes: bytes | None,
) -> dict[str, object]:
    return _run_training(job_id, train_config, probe_samples, init_checkpoint_bytes)


@app.function(
    image=image,
    gpu="H100",
    cpu=16.0,
    memory=65_536,
    timeout=4 * 60 * 60,
    single_use_containers=True,
    volumes={REMOTE_RESULTS: results},
)
def train_remote_h100(
    job_id: str,
    train_config: dict[str, object],
    probe_samples: int,
    init_checkpoint_bytes: bytes | None,
) -> dict[str, object]:
    return _run_training(job_id, train_config, probe_samples, init_checkpoint_bytes)


@app.function(
    image=image,
    gpu="H100",
    cpu=16.0,
    memory=65_536,
    timeout=4 * 60 * 60,
    single_use_containers=True,
    volumes={REMOTE_RESULTS: results},
)
def train_codec_remote_h100(
    job_id: str,
    train_config: dict[str, object],
    init_checkpoint_bytes: bytes | None,
) -> dict[str, object]:
    return _run_codec_training(job_id, train_config, init_checkpoint_bytes)


@app.function(
    image=image,
    gpu="H100",
    cpu=16.0,
    memory=65_536,
    timeout=4 * 60 * 60,
    single_use_containers=True,
    volumes={REMOTE_RESULTS: results},
)
def train_latent_remote_h100(
    job_id: str,
    train_config: dict[str, object],
    codec_checkpoint_bytes: bytes,
) -> dict[str, object]:
    return _run_latent_training(job_id, train_config, codec_checkpoint_bytes)


@app.function(
    image=image,
    gpu="H100",
    cpu=16.0,
    memory=65_536,
    timeout=4 * 60 * 60,
    single_use_containers=True,
    volumes={REMOTE_RESULTS: results},
)
def train_direct_remote_h100(
    job_id: str,
    train_config: dict[str, object],
    codec_checkpoint_bytes: bytes,
) -> dict[str, object]:
    return _run_direct_training(job_id, train_config, codec_checkpoint_bytes)


@app.function(
    image=image,
    gpu="H100",
    cpu=16.0,
    memory=65_536,
    timeout=4 * 60 * 60,
    single_use_containers=True,
    volumes={REMOTE_RESULTS: results},
)
def train_pixel_direct_remote_h100(
    job_id: str,
    train_config: dict[str, object],
) -> dict[str, object]:
    return _run_pixel_direct_training(job_id, train_config)


@app.function(
    image=image,
    gpu="L4",
    cpu=8.0,
    memory=24_576,
    timeout=30 * 60,
    single_use_containers=True,
)
def probe_checkpoint_remote(checkpoint_bytes: bytes, probe_samples: int) -> dict[str, object]:
    sys.path.insert(0, REMOTE_PROJECT)
    from blocket_league.probe import run_probes

    checkpoint = Path("/tmp/checkpoint.pt")
    output = Path("/tmp/probes.json")
    checkpoint.write_bytes(checkpoint_bytes)
    return run_probes(
        checkpoint,
        output,
        samples=probe_samples,
        batch_size=16,
    )


@app.function(
    image=image,
    gpu="A100",
    cpu=12.0,
    memory=49_152,
    timeout=45 * 60,
    single_use_containers=True,
)
def interpret_latent_checkpoint_remote(
    checkpoint_bytes: bytes,
    samples: int,
    batch_size: int,
    intervention_samples: int,
) -> dict[str, object]:
    sys.path.insert(0, REMOTE_PROJECT)
    from blocket_league.latent_probe import run_latent_interpretability

    checkpoint = Path("/tmp/latent-checkpoint.pt")
    output = Path("/tmp/latent-interpretability.json")
    checkpoint.write_bytes(checkpoint_bytes)
    return run_latent_interpretability(
        checkpoint,
        output,
        samples=samples,
        batch_size=batch_size,
        intervention_samples=intervention_samples,
    )


@app.function(
    image=image,
    gpu="A100",
    cpu=12.0,
    memory=49_152,
    timeout=60 * 60,
    single_use_containers=True,
)
def interpret_direct_checkpoint_remote(
    checkpoint_bytes: bytes,
    samples: int,
    batch_size: int,
    intervention_samples: int,
) -> dict[str, object]:
    sys.path.insert(0, REMOTE_PROJECT)
    from blocket_league.direct_probe import run_direct_interpretability

    checkpoint = Path("/tmp/direct-checkpoint.pt")
    output = Path("/tmp/direct-interpretability.json")
    checkpoint.write_bytes(checkpoint_bytes)
    return run_direct_interpretability(
        checkpoint,
        output,
        samples=samples,
        batch_size=batch_size,
        intervention_samples=intervention_samples,
    )


@app.function(
    image=image,
    gpu="H100",
    cpu=16.0,
    memory=65_536,
    timeout=60 * 60,
    single_use_containers=True,
)
def interpret_pixel_checkpoint_remote(
    checkpoint_bytes: bytes,
    fit_samples: int,
    test_samples: int,
    batch_size: int,
    strength: float,
) -> dict[str, object]:
    sys.path.insert(0, REMOTE_PROJECT)
    from blocket_league.pixel_probe import run_pixel_interpretability

    checkpoint = Path("/tmp/passive-pixel-checkpoint.pt")
    output = Path("/tmp/passive-pixel-interpretability.json")
    checkpoint.write_bytes(checkpoint_bytes)
    return run_pixel_interpretability(
        checkpoint,
        output,
        fit_samples=fit_samples,
        test_samples=test_samples,
        batch_size=batch_size,
        rollout_frames=12,
        strength=strength,
    )


@app.function(
    image=image,
    gpu="A100",
    cpu=8.0,
    memory=32_768,
    timeout=30 * 60,
    single_use_containers=True,
    volumes={REMOTE_RESULTS: results},
)
def render_trajectories_remote(
    job_id: str,
    checkpoint_bytes: bytes,
    scenario_seeds: list[int],
    samples_per_scenario: int,
    ddim_steps: int,
    rollout_frames: int,
    asset_url_prefix: str,
) -> dict[str, object]:
    sys.path.insert(0, REMOTE_PROJECT)

    checkpoint = Path("/tmp/checkpoint.pt")
    checkpoint.write_bytes(checkpoint_bytes)
    output_dir = Path(REMOTE_RESULTS) / job_id
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("kind") == "latent_world_model":
        from blocket_league.latent_assets import render_latent_trajectory_atlases

        manifest = render_latent_trajectory_atlases(
            checkpoint,
            output_dir,
            scenario_seeds=scenario_seeds,
            samples_per_scenario=samples_per_scenario,
            integration_steps=ddim_steps,
            rollout_frames=rollout_frames,
            asset_url_prefix=asset_url_prefix,
        )
    elif payload.get("kind") == "representation_codec":
        from blocket_league.latent_assets import render_codec_atlases

        manifest = render_codec_atlases(
            checkpoint,
            output_dir,
            scenario_seeds=scenario_seeds,
            asset_url_prefix=asset_url_prefix,
        )
    else:
        from blocket_league.trajectory_assets import render_trajectory_atlases

        manifest = render_trajectory_atlases(
            checkpoint,
            output_dir,
            scenario_seeds=scenario_seeds,
            samples_per_scenario=samples_per_scenario,
            ddim_steps=ddim_steps,
            rollout_frames=rollout_frames,
            asset_url_prefix=asset_url_prefix,
        )
    results.commit()
    return manifest


@app.function(
    image=image,
    gpu="A100",
    cpu=12.0,
    memory=49_152,
    timeout=45 * 60,
    single_use_containers=True,
    volumes={REMOTE_RESULTS: results},
)
def render_direct_interventions_remote(
    job_id: str,
    checkpoint_bytes: bytes,
    candidate_seeds: list[int],
    scenarios: int,
    rollout_frames: int,
    strength: float,
    asset_url_prefix: str,
) -> dict[str, object]:
    sys.path.insert(0, REMOTE_PROJECT)
    from blocket_league.direct_intervention_assets import render_direct_intervention_atlases

    checkpoint = Path("/tmp/direct-checkpoint.pt")
    checkpoint.write_bytes(checkpoint_bytes)
    output_dir = Path(REMOTE_RESULTS) / job_id
    manifest = render_direct_intervention_atlases(
        checkpoint,
        output_dir,
        candidate_seeds=candidate_seeds,
        scenarios=scenarios,
        rollout_frames=rollout_frames,
        strength=strength,
        asset_url_prefix=asset_url_prefix,
    )
    results.commit()
    return manifest


def _download(remote_dir: str, local_dir: Path) -> int:
    from modal.types import FileEntryType

    local_dir.mkdir(parents=True, exist_ok=True)
    normalized = remote_dir.strip("/")
    count = 0
    for entry in results.iterdir(normalized, recursive=True):
        if entry.type is not FileEntryType.FILE:
            continue
        remote_path = entry.path.strip("/")
        relative = Path(remote_path).relative_to(normalized)
        destination = local_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in results.read_file(remote_path):
                handle.write(chunk)
        count += 1
        print(f"downloaded {relative} ({entry.size / 1024 / 1024:.2f} MiB)")
    return count


@app.local_entrypoint()
def main(
    stage: str = "pixel",
    output_dir: str = "",
    probe_checkpoint: str = "",
    interpret_checkpoint: str = "",
    interpret_direct_checkpoint: str = "",
    interpret_pixel_checkpoint: str = "",
    interpret_samples: int = 768,
    interpret_batch_size: int = 32,
    intervention_samples: int = 32,
    trajectory_checkpoint: str = "",
    trajectory_seeds: str = "2000007,2009980,2129656",
    trajectory_samples: int = 3,
    trajectory_ddim_steps: int = 8,
    trajectory_rollout_frames: int = 12,
    trajectory_asset_prefix: str = "/blocket-league/trajectories",
    intervention_asset_checkpoint: str = "",
    intervention_asset_candidates: int = 24,
    intervention_asset_scenarios: int = 3,
    intervention_asset_strength: float = 2.0,
    init_checkpoint: str = "",
    codec_checkpoint: str = "",
    preset: str = "micro",
    steps: int = 300,
    batch_size: int = 12,
    learning_rate: float = 2e-4,
    seed: int = 7,
    workers: int = 6,
    preview_ddim_steps: int = 12,
    eval_samples: int = 64,
    eval_ddim_steps: int = 8,
    context_frames: int = 6,
    future_frames: int = 8,
    prediction_type: str = "x0",
    patch_size: int = 4,
    attention_mode: str = "factorized",
    foreground_weight: float = 10.0,
    puck_weight: float = 28.0,
    terminal_timestep_fraction: float = 0.35,
    late_frame_weight: float = 1.75,
    ema_decay: float = 0.9995,
    warmup_steps: int = 500,
    log_every: int = 100,
    rollout_context_fraction: float = 0.0,
    rollout_context_start_step: int = 3_000,
    rollout_context_ramp_steps: int = 7_000,
    rollout_context_ddim_steps: int = 1,
    latent_rollout_frames: int = 24,
    latent_cache_samples: int = 16_384,
    pixel_history_frames: int = 8,
    integration_steps: int = 10,
    codec_feature_weight: float = 1.0,
    latent_dim: int = 32,
    decoder_width: int = 160,
    decoder_depth: int = 5,
    decoder_heads: int = 5,
    gpu: str = "A100",
    probe_samples: int = 0,
    keep_remote: bool = False,
) -> None:
    if stage not in {"pixel", "codec", "latent", "direct", "pixel-direct"}:
        raise ValueError("stage must be pixel, codec, latent, direct, or pixel-direct")
    if preset not in {"micro", "tiny", "small"}:
        raise ValueError("preset must be micro, tiny, or small")
    if prediction_type not in {"x0", "v", "epsilon"}:
        raise ValueError("prediction_type must be x0, v, or epsilon")
    if attention_mode not in {"full", "factorized"}:
        raise ValueError("attention_mode must be full or factorized")
    if gpu not in {"L4", "A100", "H100"}:
        raise ValueError("gpu must be L4, A100, or H100")
    if not 0.0 <= rollout_context_fraction <= 1.0:
        raise ValueError("rollout_context_fraction must be in [0, 1]")
    if stage in {"codec", "latent", "direct", "pixel-direct"} and gpu != "H100":
        raise ValueError("The representation-codec stages currently run on H100")
    if probe_checkpoint:
        checkpoint = Path(probe_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        sample_count = probe_samples or 512
        probe_result = probe_checkpoint_remote.remote(checkpoint.read_bytes(), sample_count)
        destination = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else checkpoint.with_name("probes-high-noise.json")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(probe_result, indent=2), encoding="utf-8")
        print(json.dumps(probe_result, indent=2))
        print(f"Probe results downloaded to: {destination}")
        return
    if interpret_checkpoint:
        checkpoint = Path(interpret_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        interpretation = interpret_latent_checkpoint_remote.remote(
            checkpoint.read_bytes(),
            interpret_samples,
            interpret_batch_size,
            intervention_samples,
        )
        destination_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else HERE.parent / "public" / "blocket-league" / "interpretability"
        )
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / "manifest.json"
        destination.write_text(json.dumps(interpretation, indent=2), encoding="utf-8")
        print(json.dumps(interpretation, indent=2))
        print(f"Interpretability results downloaded to: {destination}")
        return
    if interpret_direct_checkpoint:
        checkpoint = Path(interpret_direct_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        interpretation = interpret_direct_checkpoint_remote.remote(
            checkpoint.read_bytes(),
            interpret_samples,
            interpret_batch_size,
            intervention_samples,
        )
        destination_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else HERE.parent / "public" / "blocket-league" / "interpretability"
        )
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / "direct-manifest.json"
        destination.write_text(json.dumps(interpretation, indent=2), encoding="utf-8")
        print(json.dumps(interpretation, indent=2))
        print(f"Direct interpretability results downloaded to: {destination}")
        return
    if interpret_pixel_checkpoint:
        checkpoint = Path(interpret_pixel_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        fit_samples = interpret_samples
        test_samples = max(intervention_samples, 64)
        interpretation = interpret_pixel_checkpoint_remote.remote(
            checkpoint.read_bytes(),
            fit_samples,
            test_samples,
            interpret_batch_size,
            intervention_asset_strength,
        )
        destination_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else HERE.parent / "public" / "blocket-league" / "interpretability"
        )
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / "passive-pixel-manifest.json"
        destination.write_text(json.dumps(interpretation, indent=2), encoding="utf-8")
        print(json.dumps(interpretation, indent=2))
        print(f"Passive pixel interpretability results downloaded to: {destination}")
        return
    if trajectory_checkpoint:
        checkpoint = Path(trajectory_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if trajectory_rollout_frames < 1:
            raise ValueError("trajectory_rollout_frames must be positive")
        seeds = [int(value.strip()) for value in trajectory_seeds.split(",") if value.strip()]
        if not seeds:
            raise ValueError("trajectory_seeds must include at least one integer seed")
        job_id = f"trajectories-{uuid.uuid4().hex[:8]}"
        destination = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else HERE / "outputs" / job_id
        )
        manifest = render_trajectories_remote.remote(
            job_id,
            checkpoint.read_bytes(),
            seeds,
            trajectory_samples,
            trajectory_ddim_steps,
            trajectory_rollout_frames,
            trajectory_asset_prefix,
        )
        count = _download(job_id, destination)
        if count == 0:
            raise RuntimeError(f"No trajectory assets found for Modal job {job_id}")
        try:
            results.remove_file(job_id, recursive=True)
        except Exception as error:
            print(f"Warning: could not remove remote artifacts: {error}")
        print(json.dumps(manifest, indent=2))
        print(f"Trajectory assets downloaded to: {destination}")
        return
    if intervention_asset_checkpoint:
        checkpoint = Path(intervention_asset_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        job_id = f"interventions-{uuid.uuid4().hex[:8]}"
        destination = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else HERE.parent / "public" / "blocket-league" / "interventions"
        )
        seeds = [9_100_003 + index * 9_973 for index in range(intervention_asset_candidates)]
        manifest = render_direct_interventions_remote.remote(
            job_id,
            checkpoint.read_bytes(),
            seeds,
            intervention_asset_scenarios,
            12,
            intervention_asset_strength,
            "/blocket-league/interventions",
        )
        count = _download(job_id, destination)
        if count == 0:
            raise RuntimeError(f"No intervention assets found for Modal job {job_id}")
        try:
            results.remove_file(job_id, recursive=True)
        except Exception as error:
            print(f"Warning: could not remove remote artifacts: {error}")
        print(json.dumps(manifest, indent=2))
        print(f"Intervention assets downloaded to: {destination}")
        return
    if stage == "codec":
        job_id = f"codec-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        destination = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else HERE / "outputs" / job_id
        )
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"Output directory is not empty: {destination}")
        print(f"Starting Blocket RAEv2 codec on Modal: {job_id} ({steps=}, {batch_size=})")
        codec_init_bytes: bytes | None = None
        if init_checkpoint:
            codec_init = Path(init_checkpoint).expanduser().resolve()
            if not codec_init.is_file():
                raise FileNotFoundError(codec_init)
            codec_init_bytes = codec_init.read_bytes()
        summary = train_codec_remote_h100.remote(
            job_id,
            {
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "workers": workers,
                "log_every": log_every,
                "eval_samples": eval_samples,
                "image_size": 64,
                "context_frames": context_frames,
                "future_frames": future_frames,
                "feature_weight": codec_feature_weight,
                "foreground_weight": foreground_weight,
                "puck_weight": puck_weight,
                "warmup_steps": warmup_steps,
                "latent_dim": latent_dim,
                "decoder_width": decoder_width,
                "decoder_depth": decoder_depth,
                "decoder_heads": decoder_heads,
            },
            codec_init_bytes,
        )
        print(json.dumps(summary, indent=2))
        count = _download(job_id, destination)
        if count == 0:
            raise RuntimeError(f"No codec artifacts found for Modal job {job_id}")
        if not keep_remote:
            try:
                results.remove_file(job_id, recursive=True)
            except Exception as error:
                print(f"Warning: could not remove remote artifacts: {error}")
        print(f"Blocket codec artifacts downloaded to: {destination}")
        return
    if stage == "latent":
        checkpoint = Path(codec_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("--codec-checkpoint must point to a trained codec.pt")
        job_id = f"latent-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        destination = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else HERE / "outputs" / job_id
        )
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"Output directory is not empty: {destination}")
        print(f"Starting Blocket latent world model on Modal: {job_id} ({preset=}, {steps=}, {batch_size=})")
        summary = train_latent_remote_h100.remote(
            job_id,
            {
                "preset": preset,
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "workers": workers,
                "log_every": log_every,
                "eval_samples": eval_samples,
                "integration_steps": integration_steps,
                "image_size": 64,
                "context_frames": context_frames,
                "future_frames": future_frames,
                "rollout_frames": latent_rollout_frames,
                "cache_samples": latent_cache_samples,
                "late_frame_weight": late_frame_weight,
                "ema_decay": ema_decay,
                "warmup_steps": warmup_steps,
            },
            checkpoint.read_bytes(),
        )
        print(json.dumps(summary, indent=2))
        count = _download(job_id, destination)
        if count == 0:
            raise RuntimeError(f"No latent artifacts found for Modal job {job_id}")
        if not keep_remote:
            try:
                results.remove_file(job_id, recursive=True)
            except Exception as error:
                print(f"Warning: could not remove remote artifacts: {error}")
        print(f"Blocket latent artifacts downloaded to: {destination}")
        return
    if stage == "direct":
        checkpoint = Path(codec_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("--codec-checkpoint must point to a trained codec.pt")
        job_id = f"direct-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        destination = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else HERE / "outputs" / job_id
        )
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"Output directory is not empty: {destination}")
        print(
            f"Starting direct latent transformer on Modal: "
            f"{job_id} ({preset=}, {steps=}, {batch_size=})"
        )
        summary = train_direct_remote_h100.remote(
            job_id,
            {
                "preset": preset,
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "workers": workers,
                "log_every": log_every,
                "eval_samples": eval_samples,
                "image_size": 64,
                "rollout_frames": latent_rollout_frames,
                "cache_samples": latent_cache_samples,
                "ema_decay": ema_decay,
                "warmup_steps": warmup_steps,
            },
            checkpoint.read_bytes(),
        )
        print(json.dumps(summary, indent=2))
        count = _download(job_id, destination)
        if count == 0:
            raise RuntimeError(f"No direct-model artifacts found for Modal job {job_id}")
        if not keep_remote:
            try:
                results.remove_file(job_id, recursive=True)
            except Exception as error:
                print(f"Warning: could not remove remote artifacts: {error}")
        print(f"Direct latent transformer artifacts downloaded to: {destination}")
        return
    if stage == "pixel-direct":
        job_id = f"pixel-direct-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        destination = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else HERE / "outputs" / job_id
        )
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"Output directory is not empty: {destination}")
        print(
            f"Starting direct pixel transformer on Modal: "
            f"{job_id} ({preset=}, {steps=}, {batch_size=})"
        )
        summary = train_pixel_direct_remote_h100.remote(
            job_id,
            {
                "preset": preset,
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "workers": workers,
                "log_every": log_every,
                "eval_samples": eval_samples,
                "image_size": 64,
                "patch_size": patch_size,
                "history_frames": pixel_history_frames,
                "rollout_frames": latent_rollout_frames,
                "cache_samples": latent_cache_samples,
                "late_frame_weight": late_frame_weight,
                "ema_decay": ema_decay,
                "warmup_steps": warmup_steps,
            },
        )
        print(json.dumps(summary, indent=2))
        count = _download(job_id, destination)
        if count == 0:
            raise RuntimeError(f"No direct pixel artifacts found for Modal job {job_id}")
        if not keep_remote:
            try:
                results.remove_file(job_id, recursive=True)
            except Exception as error:
                print(f"Warning: could not remove remote artifacts: {error}")
        print(f"Direct pixel transformer artifacts downloaded to: {destination}")
        return
    job_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else HERE / "outputs" / job_id
    )
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Output directory is not empty: {destination}")
    init_checkpoint_bytes: bytes | None = None
    if init_checkpoint:
        init_checkpoint_path = Path(init_checkpoint).expanduser().resolve()
        if not init_checkpoint_path.is_file():
            raise FileNotFoundError(init_checkpoint_path)
        init_checkpoint_bytes = init_checkpoint_path.read_bytes()
    print(f"Starting Blocket League on Modal: {job_id} ({preset=}, {steps=}, {batch_size=}, {gpu=})")
    remote_function = {
        "L4": train_remote_l4,
        "A100": train_remote_a100,
        "H100": train_remote_h100,
    }[gpu]
    summary = remote_function.remote(
        job_id,
        {
            "preset": preset,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "workers": workers,
            "preview_ddim_steps": preview_ddim_steps,
            "eval_samples": eval_samples,
            "eval_ddim_steps": eval_ddim_steps,
            "context_frames": context_frames,
            "future_frames": future_frames,
            "prediction_type": prediction_type,
            "patch_size": patch_size,
            "attention_mode": attention_mode,
            "foreground_weight": foreground_weight,
            "puck_weight": puck_weight,
            "terminal_timestep_fraction": terminal_timestep_fraction,
            "late_frame_weight": late_frame_weight,
            "ema_decay": ema_decay,
            "warmup_steps": warmup_steps,
            "log_every": log_every,
            "rollout_context_fraction": rollout_context_fraction,
            "rollout_context_start_step": rollout_context_start_step,
            "rollout_context_ramp_steps": rollout_context_ramp_steps,
            "rollout_context_ddim_steps": rollout_context_ddim_steps,
        },
        probe_samples,
        init_checkpoint_bytes,
    )
    print(json.dumps(summary, indent=2))
    count = _download(job_id, destination)
    if count == 0:
        raise RuntimeError(f"No artifacts found for Modal job {job_id}")
    if not keep_remote:
        try:
            results.remove_file(job_id, recursive=True)
        except Exception as error:
            print(f"Warning: could not remove remote artifacts: {error}")
    print(f"Blocket League artifacts downloaded to: {destination}")
