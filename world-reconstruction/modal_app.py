from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import modal


APP_NAME = "hyworld-worldmirror"
HYWORLD_COMMIT = "7f668e67c74338d50684e57be46a438459b6bbe1"
HYWORLD_ROOT = "/opt/hyworld"
MODEL_CACHE_PATH = "/cache"
JOBS_PATH = "/jobs"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

HERE = Path(__file__).resolve().parent
ATTENTION_PATCH = HERE / "patch_attention.py"
VIEWER_HTML = HERE / "viewer.html"

model_cache = modal.Volume.from_name(
    "hyworld-worldmirror-model-cache", create_if_missing=True
)
jobs = modal.Volume.from_name("hyworld-worldmirror-jobs", create_if_missing=True)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface-hub[hf_xet]")
    .env(
        {
            "HF_HOME": f"{MODEL_CACHE_PATH}/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "ninja-build",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HOME": f"{MODEL_CACHE_PATH}/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONPATH": HYWORLD_ROOT,
            # L40S is Ada (SM 8.9). Specifying it lets gsplat compile without
            # needing a GPU attached during the image-build step.
            "TORCH_CUDA_ARCH_LIST": "8.9",
        }
    )
    .uv_pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "einops",
        "huggingface-hub[hf_xet]",
        "matplotlib==3.10.3",
        "moviepy==1.0.3",
        "numpy==1.26.4",
        "omegaconf",
        "onnxruntime",
        "open3d==0.18.0",
        "opencv-python-headless==4.10.0.84",
        "pillow",
        "pillow-heif",
        "plyfile",
        "pycolmap==3.10.0",
        "requests",
        "safetensors",
        "scipy==1.14.1",
        "torchmetrics",
        "tqdm",
        "trimesh",
        "uniception",
        "wheel",
    )
    .run_commands(
        f"git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git {HYWORLD_ROOT}",
        f"cd {HYWORLD_ROOT} && git checkout {HYWORLD_COMMIT}",
    )
    .apt_install("libglm-dev")
    .add_local_file(ATTENTION_PATCH, "/tmp/patch_attention.py", copy=True)
    .run_commands(
        f"python /tmp/patch_attention.py {HYWORLD_ROOT}",
        (
            f"cd {HYWORLD_ROOT}/hyworld2/worldgen/third_party/gsplat_maskgaussian "
            "&& python -m pip install --no-build-isolation -e ."
        ),
        env={
            "CC": "/usr/bin/gcc",
            "CUDAHOSTCXX": "/usr/bin/g++",
            "CXX": "/usr/bin/g++",
        },
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=download_image,
    cpu=2.0,
    memory=4096,
    timeout=30 * 60,
    volumes={MODEL_CACHE_PATH: model_cache},
)
def prepare_model() -> dict[str, object]:
    """Populate the shared cache on CPU so GPU time is not spent downloading."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    repo_root = snapshot_download(
        repo_id="tencent/HY-World-2.0",
        allow_patterns=["HY-WorldMirror-2.0/*"],
    )
    model_dir = Path(repo_root) / "HY-WorldMirror-2.0"
    checkpoint = model_dir / "model.safetensors"
    if not checkpoint.is_file():
        raise RuntimeError("Hugging Face download did not produce model.safetensors")
    model_cache.commit()
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


@app.function(
    image=cuda_image,
    gpu="L40S",
    timeout=10 * 60,
    single_use_containers=True,
)
def doctor() -> dict[str, object]:
    """Verify CUDA, gsplat, and the patched WorldMirror import without weights."""
    import torch
    from gsplat.cuda._backend import _C  # noqa: F401
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline  # noqa: F401

    return {
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
    }


def _convert_worldmirror_ply_to_splat(
    source: Path,
    destination: Path,
    max_splats: int,
) -> int:
    """Convert WorldMirror's PLY into a compact, browser-friendly .splat file."""
    import numpy as np
    from plyfile import PlyData

    sh_c0 = 0.28209479177387814
    vertices = PlyData.read(str(source))["vertex"].data

    required = (
        "x",
        "y",
        "z",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "opacity",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    )
    finite = np.ones(len(vertices), dtype=bool)
    for field in required:
        finite &= np.isfinite(vertices[field])
    vertices = vertices[finite]

    # Put larger, more opaque splats first. This gives the browser sorter a
    # useful subset if the generated scene needs to be capped.
    volume = np.exp(
        vertices["scale_0"].astype(np.float64)
        + vertices["scale_1"].astype(np.float64)
        + vertices["scale_2"].astype(np.float64)
    )
    score = volume * vertices["opacity"].astype(np.float64)
    order = np.argsort(-score, kind="stable")
    if max_splats > 0:
        order = order[:max_splats]
    vertices = vertices[order]

    record_type = np.dtype(
        [
            ("position", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("color", "u1", (4,)),
            ("rotation", "u1", (4,)),
        ]
    )
    output = np.empty(len(vertices), dtype=record_type)

    # WorldMirror is Y-down. Rotate 180 degrees around X for a Y-up viewer.
    output["position"][:, 0] = vertices["x"]
    output["position"][:, 1] = -vertices["y"]
    output["position"][:, 2] = -vertices["z"]
    output["scale"][:, 0] = np.exp(vertices["scale_0"])
    output["scale"][:, 1] = np.exp(vertices["scale_1"])
    output["scale"][:, 2] = np.exp(vertices["scale_2"])

    rgb = np.column_stack(
        [
            0.5 + sh_c0 * vertices["f_dc_0"],
            0.5 + sh_c0 * vertices["f_dc_1"],
            0.5 + sh_c0 * vertices["f_dc_2"],
        ]
    )
    output["color"][:, :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    # WorldMirror saves opacity after sigmoid, so it is already in [0, 1].
    output["color"][:, 3] = np.clip(
        vertices["opacity"] * 255.0, 0, 255
    ).astype(np.uint8)

    qw = vertices["rot_0"].astype(np.float64)
    qx = vertices["rot_1"].astype(np.float64)
    qy = vertices["rot_2"].astype(np.float64)
    qz = vertices["rot_3"].astype(np.float64)
    rotation = np.column_stack((-qx, qw, -qz, qy))
    norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    rotation /= norm
    output["rotation"] = np.clip(rotation * 128.0 + 128.0, 0, 255).astype(
        np.uint8
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(destination)
    return len(output)


@app.function(
    image=cuda_image,
    gpu="L40S",
    cpu=8.0,
    memory=32768,
    timeout=60 * 60,
    single_use_containers=True,
    volumes={MODEL_CACHE_PATH: model_cache, JOBS_PATH: jobs},
)
def reconstruct(
    job_id: str,
    target_size: int = 518,
    max_splats: int = 1_200_000,
) -> dict[str, object]:
    import subprocess
    import time

    import torch
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

    input_dir = Path(JOBS_PATH) / "inputs" / job_id
    output_dir = Path(JOBS_PATH) / "outputs" / job_id
    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported photos were uploaded for job {job_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[job {job_id}] {len(image_paths)} photos on {gpu_name}")

    pipeline = WorldMirrorPipeline.from_pretrained(
        "tencent/HY-World-2.0",
        subfolder="HY-WorldMirror-2.0",
        enable_bf16=True,
    )
    pipeline(
        str(input_dir),
        strict_output_path=str(output_dir),
        target_size=target_size,
        save_depth=True,
        save_normal=True,
        save_gs=True,
        save_camera=True,
        save_points=True,
        save_colmap=False,
        save_conf=False,
        apply_sky_mask=False,
        apply_edge_mask=True,
        apply_confidence_mask=False,
        compress_pts=True,
        compress_pts_max_points=2_000_000,
        # Preserve the upstream high-detail PLY (up to its normal 5M cap).
        # Only the derived browser .splat is capped by max_splats.
        compress_gs_max_points=5_000_000,
        save_rendered=False,
    )

    gaussian_ply = output_dir / "gaussians.ply"
    if not gaussian_ply.is_file():
        raise RuntimeError("WorldMirror completed without producing gaussians.ply")
    splat_count = _convert_worldmirror_ply_to_splat(
        gaussian_ply,
        output_dir / "gaussians.splat",
        max_splats=max_splats,
    )

    elapsed = time.perf_counter() - started
    manifest = {
        "job_id": job_id,
        "created_at": datetime.now(UTC).isoformat(),
        "input_images": len(image_paths),
        "target_size": target_size,
        "viewer_splats": splat_count,
        "gpu": gpu_name,
        "elapsed_seconds": round(elapsed, 3),
        "nvidia_smi": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    model_cache.commit()
    jobs.commit()
    print(f"[job {job_id}] finished in {elapsed:.1f}s with {splat_count:,} splats")
    return manifest


def _select_images(input_dir: Path, max_images: int) -> list[Path]:
    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"No supported photos in {input_dir}. Expected: {supported}")
    if max_images < 1:
        raise ValueError("max_images must be at least 1")
    if len(images) <= max_images:
        return images
    if max_images == 1:
        return [images[len(images) // 2]]

    # Keep coverage across the entire capture instead of using only its start.
    indexes = [round(i * (len(images) - 1) / (max_images - 1)) for i in range(max_images)]
    return [images[index] for index in indexes]


def _download_volume_directory(remote_dir: str, local_dir: Path) -> int:
    from modal.types import FileEntryType

    local_dir.mkdir(parents=True, exist_ok=True)
    normalized_remote = remote_dir.strip("/")
    downloaded = 0
    for entry in jobs.iterdir(normalized_remote, recursive=True):
        if entry.type is not FileEntryType.FILE:
            continue
        remote_path = entry.path.strip("/")
        relative = Path(remote_path).relative_to(normalized_remote)
        destination = local_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in jobs.read_file(remote_path):
                handle.write(chunk)
        downloaded += 1
        print(f"downloaded {relative} ({entry.size / 1024 / 1024:.1f} MiB)")
    return downloaded


@app.local_entrypoint()
def main(
    input_dir: str,
    output_dir: str = "",
    target_size: int = 518,
    max_images: int = 32,
    max_splats: int = 1_200_000,
    keep_remote: bool = False,
) -> None:
    source = Path(input_dir).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Input directory does not exist: {source}")
    if not 280 <= target_size <= 952:
        raise ValueError("target_size must be between 280 and 952")

    selected = _select_images(source, max_images=max_images)
    job_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    remote_input = f"inputs/{job_id}"
    remote_output = f"outputs/{job_id}"
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else HERE / "outputs" / job_id
    )
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Output directory is not empty: {destination}")

    print("Preparing the persistent model cache on CPU (fast after the first run)")
    model_info = prepare_model.remote()
    print(
        f"Model ready ({int(model_info['checkpoint_bytes']) / 1024**3:.2f} GiB checkpoint)"
    )
    print(f"Uploading {len(selected)} photo(s) from {source}")
    if len(selected) == 1:
        print("Warning: one photo works, but overlapping multi-view photos reconstruct better.")
    with jobs.batch_upload() as upload:
        for image_path in selected:
            upload.put_file(image_path, f"/{remote_input}/{image_path.name}")

    print(f"Starting one-shot L40S reconstruction: {job_id}")
    result = reconstruct.remote(
        job_id=job_id,
        target_size=target_size,
        max_splats=max_splats,
    )
    print(json.dumps(result, indent=2))

    count = _download_volume_directory(remote_output, destination)
    if count == 0:
        raise RuntimeError(f"No output files found for Modal job {job_id}")
    shutil.copyfile(VIEWER_HTML, destination / "viewer.html")

    if not keep_remote:
        try:
            jobs.remove_file(remote_input, recursive=True)
            jobs.remove_file(remote_output, recursive=True)
        except Exception as error:  # cleanup must not hide a successful result
            print(f"Warning: could not clean Modal job files: {error}")

    print(f"\nReconstruction downloaded to: {destination}")
    print("Explore it with:")
    print(f"  ./view.sh {shlex_quote(str(destination))}")


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)
