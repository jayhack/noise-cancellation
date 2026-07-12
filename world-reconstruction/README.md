# HY-WorldMirror on Modal

One command uploads a directory of overlapping photos, starts a single-use
Modal L40S GPU, reconstructs the scene with HY-WorldMirror 2.0, downloads all
artifacts, and shuts the GPU container down.

## Quick start

No camera labels, poses, or manual ordering are required. Put overlapping photos
of one scene in a directory, then run:

```bash
./run.sh /absolute/path/to/photos
./view.sh ./outputs/<job-id>
```

The reconstruction command prints the exact output path. Inputs may be named
anything, although capture-order filenames are helpful because the runner sorts
them lexicographically.

The generated bundle includes:

- `gaussians.splat` — capped and transformed for the included browser viewer
- `gaussians.ply` — full WorldMirror Gaussian output
- `points.ply` — point cloud
- `depth/` and `normal/` — per-view maps
- `camera_params.json` — recovered cameras
- `manifest.json` and `pipeline_timing.json` — run metadata

## Capture photos

Use 8–24 JPEG, PNG, or WebP photos with substantial overlap. Walk slowly around
the subject or through the space. Keep exposure stable, avoid motion blur, and
make sure distinctive details appear in several neighboring frames. The runner
accepts up to 32 images by default and samples evenly when a folder contains
more.

HEIC is not accepted by the upstream pipeline. Export iPhone photos as JPEG
first.

## Run

From this directory:

```bash
./run.sh /absolute/path/to/photos
```

The first run builds the CUDA image and downloads the approximately 4.7 GiB
WorldMirror checkpoint. The checkpoint download happens in a CPU container so
you do not pay L40S rates for network transfer. Both are cached for subsequent
runs. The model cache is persistent, but the L40S container is marked single-use
and exits after the one reconstruction.

Useful options are passed after the photo directory:

```bash
./run.sh /absolute/path/to/photos \
  --target-size 518 \
  --max-images 24 \
  --max-splats 1200000 \
  --output-dir /absolute/path/to/my-reconstruction
```

- `--target-size`: 280–952. Start with 518; 952 is slower and uses more memory.
- `--max-images`: limits and evenly samples the input set; default 32.
- `--max-splats`: browser copy cap; the full `gaussians.ply` is still retained.
- `--keep-remote`: retain uploaded inputs and outputs in the Modal jobs Volume.

By default, inputs and outputs are removed from the jobs Volume after a
successful local download. The model cache remains so later runs start faster.

To verify the CUDA image without downloading model weights or running a
reconstruction:

```bash
uvx --from modal modal run modal_app.py::doctor
```

## Explore the result

The generation command prints the exact output directory. Serve it with:

```bash
./view.sh ./outputs/<job-id>
```

This opens a local WebGL Gaussian-splat viewer. The viewer imports its JavaScript
dependencies from jsDelivr, so it needs an internet connection when opened.

## Modal resources

The app lazily creates these Volumes:

- `hyworld-worldmirror-model-cache`
- `hyworld-worldmirror-jobs`

Inspect them with `modal volume list`. Delete the model cache only if you are
comfortable downloading the checkpoint again.

## Notes for agents

- This folder is intentionally independent of the repository's Next.js app.
- `modal_app.py` owns upload, CPU model caching, L40S inference, output download,
  and remote cleanup.
- `patch_attention.py` applies the minimal SDPA fallback to the pinned upstream
  HY-World commit during the Modal image build.
- `viewer.html` is copied into every downloaded output; `view.sh` serves it over
  localhost so its module and Web Worker dependencies function correctly.
- Keep `HYWORLD_COMMIT` pinned. Re-run `doctor` and a low-resolution multi-photo
  reconstruction before updating it.
- Never commit model weights or generated geometry. The local ignore file covers
  checkpoints, tensor dumps, PLY files, splats, and `outputs/`.
- The persistent Modal model-cache Volume is intentional. Per-job inputs and
  outputs are deleted after a successful download unless `--keep-remote` is set.

## Model terms

HY-World 2.0 uses the Tencent HY-WORLD 2.0 Community License and contains
geographic/use restrictions. Review the upstream
[license](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/License.txt)
before running or distributing outputs.
