# Active Sound Control Lab

An interactive 2D explanation of active noise cancellation, from an ideal continuous boundary to discrete speaker arrays, multiple listeners, reflective obstacles, transfer-matrix estimation, and environment-aware recovery.

## Run locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Blocket League world-model lab

The independent [Blocket League lab](http://localhost:3000/blocket-league/) is a
playable two-disc physics world plus an action-conditioned video-diffusion
research scaffold. Its browser simulator lives at `/blocket-league`; the Python
simulator, VideoDiT, Modal training entry point, and layerwise probes are in
[`blocket_league`](./blocket_league).

The lab now ends with a held-out latent interpretability audit for Cartesian
position, velocity, polar geometry, and imminent collision, including untrained,
action-only, shuffled-label, and matched-noise causal-edit controls. A
puck-centered, activation-located downstream-Jacobian intervention now causally writes
both signed velocity and speed through 12 frozen-model rollout frames, while
ordinary probe and equal-norm random directions remain near zero.

Run the current strong baseline with the authenticated Modal client:

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --preset tiny --steps 30000 --batch-size 16 \
  --context-frames 6 --patch-size 4 \
  --attention-mode factorized --gpu H100 \
  --probe-samples 2048
```

See [`blocket_league/README.md`](./blocket_league/README.md) for the clip schema,
model presets, exported artifacts, and probe command.

## What the sequence demonstrates

1. An ideal continuous cancellation ring.
2. A discrete microphone/speaker array tracking one moving listener.
3. Simultaneous cancellation bubbles for three listeners.
4. Failure when an open-field controller is used around reflective buildings.
5. Estimation of the environmental transfer matrix from a moving microphone probe.
6. Re-optimization using the estimated transfer matrix.
7. Multi-frequency cancellation of a public-domain chainsaw recording, including the uncancelled broadband residual.
8. An evidence review that bounds a realistic café-use claim with lab, field, simulation, and negative results.
9. A concept prototype around a construction site and the quieter café-table outcome it is meant to create.

The acoustic model is intentionally educational: it uses a 2D Helmholtz-style field with direct paths, first-order reflections, simplified transmission, and corner diffraction. Its decibel readouts are relative changes, not calibrated real-world SPL measurements.

## Chainsaw recording

The seventh experiment uses a continuous 12-second excerpt from [“Chainsaw 12” by ezwa](https://commons.wikimedia.org/wiki/File:Chainsaw_12.ogg), released into the public domain. Run `python3 scripts/analyze-chainsaw.py` to regenerate the extracted waveform, spectrum, periodic cycle, and Fourier profile. The analysis script requires FFmpeg, NumPy, and SciPy.

## 3D scene reconstruction

The independent [`world-reconstruction`](./world-reconstruction) utility turns
an unordered folder of overlapping scene photos into a Gaussian-splat 3D
reconstruction on a one-shot Modal L40S GPU. It downloads the output locally and
includes a browser viewer:

```bash
cd world-reconstruction
./run.sh /absolute/path/to/photos
./view.sh ./outputs/<job-id>
```

See its [README](./world-reconstruction/README.md) for capture guidance, Modal
resources, outputs, and agent maintenance notes.
