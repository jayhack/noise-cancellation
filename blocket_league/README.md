# Blocket League

Blocket League is a tiny, fully observed 2D game for studying what an
action-conditioned video diffusion model learns about dynamics. A player disc
accelerates around a fixed arena and strikes a puck into a goal. The model sees
only RGB frames and future player actions; exact position, velocity, collision,
and reset state are retained for evaluation and probes.

## World and data

The simulator runs at 20 FPS with four physics substeps per frame. Models use
six context frames and either eight or twelve future frames at 64×64 resolution.
The future action at index `t` causes the transition into future frame `t`.

Clips are generated deterministically on the fly, so normal training does not
need a stored dataset. To inspect or freeze a corpus:

```bash
uv run --with numpy python -m blocket_league.data ./blocket-dataset --samples 128
```

Each exported `.npz` records RGB frames, actions, privileged state vectors, and
event IDs. Privileged state is never passed to the world model.

## Modal training

The Modal entry point builds a GPU image, trains on an L4, A100, or H100,
persists artifacts in a Modal Volume, and downloads the completed run. The
current strong baseline is:

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --preset tiny \
  --steps 30000 \
  --batch-size 16 \
  --context-frames 6 \
  --patch-size 4 \
  --attention-mode factorized \
  --foreground-weight 10 \
  --puck-weight 28 \
  --terminal-timestep-fraction 0.35 \
  --gpu H100 \
  --probe-samples 2048
```

Use `--probe-samples 512` on a substantive run to fit layerwise ridge probes for
player and puck position and velocity. The presets are intended for a scaling
grid:

To re-probe a downloaded checkpoint at the near-pure-noise generation endpoint
without retraining:

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --probe-checkpoint blocket_league/outputs/my-run/checkpoint.pt \
  --probe-samples 1024 \
  --output-dir blocket_league/outputs/my-run/probes-high-noise.json
```

| Preset | Width | Layers | Attention heads |
| --- | ---: | ---: | ---: |
| micro | 128 | 4 | 4 |
| tiny | 192 | 6 | 6 |
| small | 256 | 8 | 8 |

Every run downloads `checkpoint.pt`, `rollout.png`, `rollout-ddim-1.png`,
`train.jsonl`, and `summary.json` to `blocket_league/outputs/<job-id>/`.

## MIRA-style representation autoencoder

The latent path follows MIRA's RAEv2 recipe at Blocket scale. It is a
deterministic representation autoencoder rather than a Gaussian VAE: a frozen
self-supervised vision encoder supplies semantic features, a learned bottleneck
compresses them in space and time, and a causal decoder reconstructs RGB. We use
public DINOv2-small in place of MIRA's gated DINOv3 checkpoint, aggregate four
intermediate feature maps, compress every two 64×64 RGB frames to one
32-channel 8×8 latent, and decode through the simulator's exact nine-color
palette.

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --stage codec \
  --init-checkpoint blocket_league/outputs/rae-dinov2-grid8-6000/codec.pt \
  --steps 4000 \
  --batch-size 64 \
  --learning-rate 0.0002 \
  --latent-dim 32 \
  --decoder-width 160 \
  --decoder-depth 5 \
  --decoder-heads 5 \
  --gpu H100 \
  --output-dir blocket_league/outputs/rae-dinov2-grid8-palette-4000
```

This run warm-starts the compatible feature projection and decoder trunk from
the preceding continuous 8×8 codec, then replaces its RGB head with the exact
palette head. The selected codec has 2.24M trainable parameters plus the frozen
feature encoder. Its latent is 12× smaller than the paired RGB input. Across
held-out clips it reconstructs the player at 0.37 px error and the puck at
0.46 px error (0.42 px mean entity error). An earlier 4×4 latent gave a larger
48× reduction but visually erased the tiny bodies, so it was rejected despite
acceptable aggregate metrics.

The world model never sees RGB. It normalizes cached codec latents, assigns an
independent flow time to every future latent, conditions on paired actions, and
learns velocity flow with clean shifted-past conditioning. At inference it
generates one latent at a time, feeds that latent back causally, and decodes only
after the full rollout.

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --stage latent \
  --codec-checkpoint blocket_league/outputs/rae-dinov2-grid8-palette-4000/codec.pt \
  --preset tiny \
  --steps 30000 \
  --batch-size 64 \
  --context-frames 6 \
  --future-frames 12 \
  --latent-rollout-frames 24 \
  --latent-cache-samples 32768 \
  --integration-steps 10 \
  --late-frame-weight 2 \
  --gpu H100 \
  --output-dir blocket_league/outputs/latent-rae-tiny-30000
```

This 5.81M-parameter latent model trained for 16.4 minutes after a 46.5-second
codec pass over 32,768 deterministic clips. With the original eight-step
linear sampler, held-out mean entity error is 5.16 px over future frames 1–12
and 14.18 px over frames 13–24. The decoded frames remain crisp and preserve
one player and one puck rather than turning into pixel entropy, but the robust
pixel-space model is still more accurate. This is therefore a useful
representation/rollout experiment, not a win over the pixel baseline.

### Direct autoregressive latent transformer

The matched direct baseline freezes the same codec but removes flow matching
and its numerical solver. An eight-latent causal transformer predicts the
normalized residual to the next latent, producing two rendered frames in one
forward pass. Training corrupts prior states with progressively larger latent
noise and targets the clean next state, so the model learns to correct the
off-manifold errors it encounters during free-running play.

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --stage direct \
  --codec-checkpoint blocket_league/outputs/rae-dinov2-grid8-palette-4000/codec.pt \
  --preset tiny \
  --steps 30000 \
  --batch-size 32 \
  --learning-rate 0.0003 \
  --latent-rollout-frames 64 \
  --latent-cache-samples 16384 \
  --gpu H100 \
  --output-dir blocket_league/outputs/direct-tiny-30000
```

The 3.67M-parameter direct transformer completed 30,000 steps in 9.0 minutes
after a 34.8-second codec cache pass. Across 128 held-out 64-frame rollouts,
mean entity error is 13.67 px (11.07 px player, 16.28 px puck); final-frame
errors are 16.05 px and 24.90 px. Short rollouts remain crisp and coherent,
while the 64-frame evaluation deliberately exposes accumulated state drift.

### Passive raw-pixel transformer (no actions)

The cleaner causal experiment removes both the pretrained representation codec
and the control channel. The source videos contain two discs with randomized
initial momentum; after initialization they evolve only through drag, walls,
goals, and disc collisions. The checkpoint consumes the world's exact
nine-color rendered pixels through an ordinary learned patch projection and
predicts the next pixel class. It has no action input or action parameters.

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --stage pixel-direct \
  --preset tiny \
  --steps 30000 \
  --batch-size 16 \
  --learning-rate 0.0003 \
  --pixel-history-frames 8 \
  --patch-size 4 \
  --latent-cache-samples 16384 \
  --latent-rollout-frames 64 \
  --eval-samples 128 \
  --gpu H100 \
  --output-dir blocket_league/outputs/passive-pixel-direct-tiny-30000
```

The 3.67M-parameter model reaches 0.88 px mean entity error across the first 12
autoregressive frames and 6.88 px over 64 frames. A post-hoc block-6 linear
probe recovers visually measured large-disc velocity at 0.92 R² on 256 held-out
clips. A single downstream-averaged +x activation direction, fit on a separate
512-clip split and written for only four frames, leaves the disc +1.38 px from
baseline after 12 frames with 79.3% sign consistency. The displacement grows by
another +0.79 px after writes stop; a matched random direction ends at -0.07 px.
The assay uses rendered pixels for its readout and routing, never simulator
state or action labels.

Run the assay with:

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --interpret-pixel-checkpoint \
    blocket_league/outputs/passive-pixel-direct-tiny-30000/checkpoint.pt \
  --interpret-samples 512 \
  --interpret-batch-size 32 \
  --intervention-samples 256 \
  --intervention-asset-strength 8 \
  --output-dir blocket_league/outputs/passive-pixel-interpretability-strong
```

### Browser player

The lab's final exhibit runs the frozen passive pixel transformer entirely in
the browser through ONNX Runtime WebGPU. It starts from eight exact 64x64 pixel
frames and predicts one new frame per forward pass. Arrow keys or WASD do not
enter the model as actions. Instead, they write the recovered x/y velocity
directions into the block-6 spatial token containing the green circle; the
white puck remains uncontrolled. The checked-in FP32 graph is 14.2 MB and
contains all 3.67M parameters.

Regenerate the checked-in browser graphs and starter state with:

```bash
uv run --python 3.12 \
  --with torch --with numpy --with pillow \
  --with onnx --with onnxscript --with onnxruntime \
  python -m blocket_league.export_browser_pixel \
  blocket_league/outputs/passive-pixel-direct-tiny-30000/checkpoint.pt \
  blocket_league/outputs/passive-pixel-interpretability-xy/passive-pixel-manifest.json \
  public/blocket-league/live
```

The exporter checks class agreement and logit error against PyTorch before
writing the graph, starter context, and intervention manifest. Chrome/Edge use
WebGPU; unsupported browsers fall back to WASM.

## Strong baseline

The 30k-step `tiny` run has 3.78M parameters and trained for 22.9 minutes on an
H100. On 128 deterministic held-out clips, the 8-step sampler reached 2.08 px
mean entity-position error across the prediction horizon: 2.33 px for the
player and 1.83 px for the puck. Final-frame errors were 4.21 px and 3.46 px.

The improvement over the first baseline comes from 4×4 patches, factorized
spatial/temporal attention, six-frame motion context, EMA weights, cosine
learning-rate decay, oversampling the pure-noise endpoint, and RGB-only loss
reweighting for the small moving objects. No privileged simulator state is used
for training. State is used only by evaluation metrics and post-hoc probes.

## Long-horizon robust model

The 12-frame model is initialized from the strong 8-frame checkpoint. Its frame
position table is extended by four entries; all other weights transfer exactly.
After step 3,000, generated-context scheduled sampling ramps to 35%: the EMA
model generates a 12-frame prefix, its final six frames replace clean context,
and the next 12 simulator frames provide supervision.

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --init-checkpoint blocket_league/outputs/juiced-tiny-30000/checkpoint.pt \
  --preset tiny \
  --steps 30000 \
  --batch-size 12 \
  --learning-rate 0.0001 \
  --context-frames 6 \
  --future-frames 12 \
  --late-frame-weight 2.25 \
  --rollout-context-fraction 0.35 \
  --rollout-context-start-step 3000 \
  --rollout-context-ramp-steps 7000 \
  --rollout-context-ddim-steps 1 \
  --gpu H100 \
  --output-dir blocket_league/outputs/long12-robust-30000
```

The 3.78M-parameter run trained for 24.3 minutes. Across 128 held-out clips,
the 8-step sampler reached 2.81 px mean entity-position error over all 12 direct
predictions: 3.10 px for the player and 2.52 px for the puck. A second complete
12-frame autoregressive pass remains substantially harder at 9.71 px mean
error, so arbitrary-length rollout is not yet solved.

## Trajectory theater

The `/blocket-league` lab compares matched held-out samples from the latent
world model, the robust 12-frame pixel model, and the original 8-frame pixel
checkpoint. It also exposes the representation codec directly: source RGB,
hard reconstruction, and a three-component projection of the 8×8 latent. Each
trajectory scenario aligns simulator truth with three independent samples and
exposes playback, scrubbing, actions, events, and model-specific trajectory
error.

Regenerate the compact PNG atlases and manifest on an A100 with:

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --trajectory-checkpoint blocket_league/outputs/juiced-tiny-30000/checkpoint.pt \
  --trajectory-seeds 2000007,2009980,2129656 \
  --trajectory-samples 3 \
  --trajectory-ddim-steps 8 \
  --trajectory-rollout-frames 12 \
  --trajectory-asset-prefix /blocket-league/trajectories \
  --output-dir public/blocket-league/trajectories
```

For the robust model, switch the checkpoint, output directory, and asset prefix
to `long12-robust-30000/checkpoint.pt`, `trajectories-long12`, and
`/blocket-league/trajectories-long12` respectively.

For the latent model, use `latent-rae-tiny-30000/checkpoint.pt`, request 24
rollout frames, and write to `public/blocket-league/trajectories-latent` with
the `/blocket-league/trajectories-latent` asset prefix. The same entry point
detects the latent checkpoint and switches `--trajectory-ddim-steps` to the
flow integrator step count.

The checked-in web assets are static; loading the lab does not invoke Modal or
PyTorch in the browser.

## Latent interpretability audit

The lab's final section fits post-hoc linear readouts to the frozen latent
checkpoint. Each example contains 16 observed frames, a two-frame action pair,
and a future latent initialized as pure noise. Readouts are trained, tuned, and
tested on disjoint deterministic simulator trajectories. Controls include the
codec latent, DiT input, an untrained transformer, action-only features, and
shuffled labels. The audit then contrasts the correlational probe direction
with a J-lens-inspired direction obtained by averaging the activation-to-render
Jacobian across held-out contexts and all flow-solver evaluations. Because the
puck moves across spatial tokens, gradients are translated into a puck-centered
coordinate system before averaging.

The intervention no longer receives simulator coordinates. A shared linear
token scorer is trained with visual self-supervision from puck-colored pixels in
observed RGB and reaches 86.5% held-out 4x4-cell accuracy. During generated
rollouts it uses Block 3 activations alone, agreeing with the decoded puck cell
79.7% of the time. That activation-inferred path determines where the causal
template is written. Simulator coordinates are retained only for evaluation.

With frozen weights and matched sampler noise, a persistent ±2σ write at block
3 changes puck x velocity by 0.81 px/frame and separates the final x position by
9.70 px after 12 frames; 85% of 256 held-out rollouts have the intended sign.
The ordinary linear-probe direction changes velocity by 0.02 px/frame and an
equal-norm random direction by -0.01 px/frame. A separately averaged speed lens
raises speed by 0.31 px/frame against the matched baseline, but its symmetric
positive-minus-negative effect is only 0.10 px/frame with a 54% expected-sign
rate. Speed control is therefore weaker evidence than signed velocity at this
sample size. These
are causal effects on decoded motion, not merely probe accuracy.

Regenerate the checked-in manifest on an A100 with:

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --interpret-checkpoint blocket_league/outputs/latent-rae-tiny-30000/checkpoint.pt \
  --interpret-samples 768 \
  --interpret-batch-size 32 \
  --intervention-samples 256 \
  --output-dir public/blocket-league/interpretability
```

The implementation is in `latent_probe.py`. Privileged simulator state is used
only as a post-hoc label; it is never supplied to the world model.

### Direct-transformer interpretability comparison

The same split, targets, controls, activation-only puck locator, and persistent
causal protocol now run on the one-pass direct transformer. Its internal state
is more linearly accessible: block 4 reaches 0.98 position R², 0.50 velocity R²,
and 0.82 polar R², while block 6 reaches 0.89 collision AUROC. The locator is
95.3% accurate against visual puck cells and uses no coordinate label when
placing an intervention.

Across 256 held-out 12-frame rollouts, a single puck-centered direction averaged
over 64 independent activation-to-render Jacobians changes signed x velocity by
1.23 px/frame and final x by 14.75 px; 91.8% of rollouts have the intended sign.
The matched linear-probe and equal-norm random directions change velocity by
−0.016 and +0.018 px/frame. A separately fitted global speed direction changes
speed by +0.50 px/frame with 91.0% sign consistency. This supplies two
cross-rollout causal directions, not a direction refit for each trajectory.

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --interpret-direct-checkpoint blocket_league/outputs/direct-tiny-30000/checkpoint.pt \
  --interpret-samples 768 \
  --interpret-batch-size 32 \
  --intervention-samples 256 \
  --output-dir public/blocket-league/interpretability
```

The direct implementation is in `direct_probe.py`; its manifest is stored next
to the preserved diffusion manifest so the lab can switch between studies.

### Frozen 64-frame failure horizon

The final lab exhibit keeps every trained weight frozen and samples 32 latent
steps, or 64 future RGB frames, from one shared six-frame observation. Four
independent samples are generated with the same actions and the same ten-step
MIRA-style flow sampler:

```bash
uv run --with torch --with transformers==4.53.0 --with numpy==2.2.6 --with pillow==11.2.1 \
  python -m blocket_league.latent_assets \
  blocket_league/outputs/latent-rae-tiny-30000/checkpoint.pt \
  public/blocket-league/entropy-rollout \
  --seeds 2000007 \
  --samples 4 \
  --integration-steps 10 \
  --rollout-frames 64 \
  --asset-url-prefix /blocket-league/entropy-rollout
```

The four samples reach 16–23 px player error and 19–22 px puck error over the
full horizon. Because the task-specific decoder makes a hard choice among nine
simulator colors, the output does not dissolve into RGB static: visual geometry
stays crisp while trajectory state and event timing drift into four mutually
incompatible worlds. The failure is dynamical entropy rather than pixel noise.

This rollout is intentionally diagnostic rather than a quality claim. Across
the nine checked-in samples, mean player error rises from 2.48 px in the direct
eight-frame window to 5.47 px after re-feeding generated context; puck error
rises from 1.56 px to 3.71 px. A robust long-horizon model therefore needs
training on longer targets and/or generated-context corruption.

## Local training

With PyTorch, NumPy, and Pillow installed:

```bash
python -m blocket_league.train --steps 100 --preset micro --workers 0
python -m blocket_league.probe blocket_league/outputs/local/checkpoint.pt
```

The model is a deliberately plain VideoDiT with a 100-step cosine noise
schedule. Clean-frame (`x0`) prediction is the default because the arena is
sparse; velocity (`v`) and noise (`epsilon`) prediction remain selectable
baselines. Dense runs alternate spatial attention within frames and temporal
attention at each spatial cell. Future tokens receive diffusion-time and
per-frame action embeddings. Every transformer block remains directly
available for hooks, activation replacement, and causal interventions.
