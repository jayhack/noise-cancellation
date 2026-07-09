# Active Sound Control Lab

An interactive 2D explanation of active noise cancellation, from an ideal continuous boundary to discrete speaker arrays, multiple listeners, reflective obstacles, transfer-matrix estimation, and environment-aware recovery.

## Run locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## What the sequence demonstrates

1. An ideal continuous cancellation ring.
2. A discrete microphone/speaker array tracking one moving listener.
3. Simultaneous cancellation bubbles for three listeners.
4. Failure when an open-field controller is used around reflective buildings.
5. Estimation of the environmental transfer matrix from a moving microphone probe.
6. Re-optimization using the estimated transfer matrix.

The acoustic model is intentionally educational: it uses a 2D Helmholtz-style field with direct paths, first-order reflections, simplified transmission, and corner diffraction. Its decibel readouts are relative changes, not calibrated real-world SPL measurements.
