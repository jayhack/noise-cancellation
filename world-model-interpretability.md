# Physics Representations in Video World Models

Notes on what world models like MIRA learn about physics, how they learn it, and
what interpretability research can currently say about the representations
inside them. Written July 2026.

## Context

MIRA ([General Intuition & Kyutai, with Epic Games][mira]) trains a 5B-parameter
diffusion transformer to simulate Rocket League end-to-end: controller actions
in, video out, at 20 fps in real time for four simultaneous players. The
training signal is nothing but (video, action) pairs — roughly 10,000
match-hours of bot self-play. The ground-truth physics state (exact ball and
car positions/orientations, logged via BakkesMod) is recorded but **never used
for training**, only for evaluation.

The result simulates the game convincingly: cars kick the ball, goals get
scored, demolitions work, and the model tracks non-visual state like the boost
gauge (you can't boost at zero). There is no physics engine, no renderer, and
no explicit 3D representation anywhere in the architecture.

This raises the question these notes are about: **if an action-conditioned
video model predicts physical dynamics accurately, what does its internal
representation of physics look like — and can we find it?**

## How the model works, briefly

- Each frame is encoded by a **video representation codec** into a grid of
  latent tokens. MIRA's codec uses a frozen self-supervised encoder
  (DINO-style) with a learned decoder back to pixels. Notably, pixel-space
  diffusion failed on this task; the choice of latent space was the single
  biggest lever on model quality.
- A **diffusion transformer** is trained with a per-sample L2 regression
  (denoising / flow-matching objective) to predict future frame latents from
  past latents and the players' action streams. The network parametrizes a
  time-dependent velocity field; sampling means drawing Gaussian noise and
  integrating that field for a few steps ([Lipman et al. 2022][flow-matching],
  [Liu et al. 2022][rectified-flow]).
- Training uses **diffusion forcing** — independent noise levels per frame
  ([Chen et al. 2024][diffusion-forcing]) — which lets the model treat its own
  generated history as "approximately trustworthy" context and is a large part
  of why rollouts are stable indefinitely.

The point to hold onto: the only objective is next-frame prediction. Any
physics the model knows, it learned because physics is useful for predicting
pixels under action conditioning.

## Evidence that a physics representation exists

Three kinds of evidence, from MIRA itself:

1. **Game-state probing.** Probes trained on the model's activations can read
   out the true positions and velocities of the ball and cars — state the
   model was never shown. Prediction under action-conditioning forces the
   network to maintain something informationally equivalent to the simulator's
   state, because that state is the minimal sufficient statistic for the
   future.
2. **Persistent non-visual state.** The boost gauge: a hidden resource tracked
   across time, with the rule (no boost at zero) enforced. That is a learned
   state machine, not texture statistics.
3. **Cross-view consistency.** Four first-person views stay synchronized —
   same ball, same cars, four cameras — with no explicit 3D representation,
   implying an implicit shared scene state.

This mirrors older results up the stack. Sequence models trained on Othello
moves build a linearly decodable board state ([Li et al. 2022][othello-gpt]).
Stable Diffusion, trained only on 2D images, internally encodes depth maps and
foreground/background segmentation — decodable with linear probes, emerging
early in denoising, and *causal*: intervening on those directions changes the
generated geometry ([Chen et al. 2023][beyond-surface]). Broader probing finds
scene geometry, support relations, shadows, and depth in Stable Diffusion and
DINOv2 features ([Zhan et al. 2023][sd-3d-scene]).

## What the representation looks like: the interpretability results

The central question the recent literature answers: does the model implement
something like a **physics engine** (compact, factorized state variables —
position, velocity, mass — reused across tasks), or a **distributed,
task-specific representation** that merely suffices for prediction?

The answer, so far, is firmly the latter.

### The Physics Emergence Zone

[Joseph et al. 2026][interpreting-physics] ran the first mechanistic
interpretability study of physical variables inside large video encoders
(V-JEPA 2 and VideoMAE-v2), using layerwise probing, subspace geometry,
patch-level decoding, and attention ablations. Key findings:

- **A sharp intermediate-depth transition** — the *Physics Emergence Zone*
  (PEZ), at roughly one-third depth — where physical variables suddenly become
  linearly accessible. Physics representations peak shortly after the PEZ and
  *degrade toward the output layers*. Middle layers, not final layers, are
  where physics lives.
- **Hierarchical emergence**: scalar speed and acceleration are decodable from
  the earliest layers, but motion *direction* only becomes accessible at the
  PEZ — a progression the authors compare to the V1 → MT motion hierarchy in
  primate visual cortex.
- **Population codes, not neurons**: direction is encoded as a circular,
  high-dimensional population code. Steering the decoded direction requires
  coordinated intervention across dozens of approximately orthogonal
  dimensions — orders of magnitude more than the low-dimensional steering
  vectors that work in language models.
- **Task-specific, not shared**: direction and possible/impossible physics
  judgments co-emerge at the PEZ but occupy nearly orthogonal subspaces —
  evidence against compact reusable latent state. They do share a circuit-level
  substrate: unusually *local* spatiotemporal attention heads in the PEZ,
  whose ablation degrades physics and temporal reasoning while leaving static
  tasks (e.g. ImageNet classification) largely intact.

Their summary table is worth internalizing: every "physics engine" prediction
(staged derivation of acceleration from velocity, Cartesian encoding, shared
latent physics, compact state variables, object-centric slots) is contradicted
by what probing actually finds.

### It generalizes across architectures and objectives

[Esmati, Nath et al. 2026][invisible-hand] asked whether the same structure
exists in *generative* video diffusion models (WAN, CogVideoX, LTX) — trained
for denoising, not prediction. To probe them on real videos, they approximately
invert the sampling ODE (integrate the learned velocity field backward from a
clean latent to noise) and probe activations along the recovered trajectory.
Findings:

- Physical plausibility is **linearly decodable from mid-network transformer
  blocks** (~81% average accuracy on IntPhys/InfLevel), outperforming dedicated
  representation learners like V-JEPA and VideoMAE on the same protocol.
- The signal is **absent from the VAE latent input** and emerges *inside* the
  denoising transformer — physical representations arise as a byproduct of
  generative denoising itself.
- The depth profile matches the PEZ: best blocks cluster in the middle third,
  across architectures with different training objectives. Intermediate-depth
  physics emergence looks like a general property of video models, not an
  artifact of one recipe.
- Strikingly, the internal signal exists **even when the generated output
  violates the same physical laws** — the models know more than they show.

### The representations are causally steerable

[Alam 2026][physics-steering] takes the linear probe's weight vector at a PEZ
layer as a concept activation vector and injects it into hidden states at
inference. This reliably shifts the model's physical-plausibility expectations
in either direction (sign of the steering vector), with no weight changes —
but *only* when the intervention lands inside the PEZ. Different intuitive
physics principles (object permanence, continuity, ...) occupy distinct
directions in the subspace. The physics representation is not just readable
but writable.

## The honest caveat: a manifold, not laws

Behavioral evidence keeps the enthusiasm calibrated. [Kang et al.
2024][physical-law] tested video models on out-of-distribution mechanics and
found *case-based* rather than *rule-based* generalization: models mimic the
nearest training example instead of extrapolating the law. MIRA's failure
modes tell the same story:

- Park all four cars away from the ball (out-of-distribution play) and the
  model visually "melts." Newtonian mechanics doesn't care where the cars are;
  this model's physics is only valid on the data manifold.
- Occluded cars get forgotten by the single-player model — state that isn't
  observable and isn't needed for short-horizon prediction decays.
- Post-goal replays are confabulated: with a ~4-second context the model
  invents a plausible replay rather than recalling the actual goal. Dynamics,
  yes; episodic memory, no.

Synthesis: video world models converge on **distributed, hierarchically
organized, task-sufficient population codes** from which symbolic physical
variables are linearly *recoverable* — but not on factorized, invariant
physical law. They interpolate dynamics on the training manifold extremely
well and extrapolate them poorly. Whether scale and data diversity close that
gap, or whether it is structural, is the open question.

## Open directions

- **Interventions as repair**: can steering PEZ subspaces fix physics errors
  in generation (beyond flipping plausibility judgments)?
- **Sparse autoencoders on video models**: still nascent — do SAE features
  decompose the population codes into anything cleaner than probes reveal?
- **Games as microscopes**: game world models (MIRA-style) offer perfect
  ground-truth state for probing, which natural video never does. Expect the
  sharpest interpretability results to come from this setting.
- **Sim-to-real transfer**: does pre-training on game physics measurably help
  real-world dynamics models? MIRA's authors flag this as unknown at scale.
- **Representation-space design**: MIRA's biggest quality jump came from the
  codec (semantic latents), not scale — suggesting perception and dynamics are
  separable, and that physics is learnable over object-centric features and
  nearly unlearnable over raw pixels at current scale. Interpretability may
  move from diagnostic to design principle.

## References

### Primary case study

- **MIRA: Multiplayer Interactive World Models with Representation
  Autoencoders** — General Intuition & Kyutai, in collaboration with Epic
  Games, 2026. [Blog post][mira] (dataset, training/inference code, and
  technical report linked from there).

### Interpretability of physics in video models

- **Interpreting Physics in Video World Models** — Joseph, Garrido,
  Balestriero, Kowal, Fel, Bakhtiari, Richards, Rabbat (Meta et al.), 2026.
  [arXiv:2602.07050][interpreting-physics] ·
  [author's walkthrough](https://www.soniajoseph.ai/interpreting-ph/)
- **The Invisible Hand of Physics: When Video Diffusion Models Know More Than
  They Show** — Esmati, Nath, Hofmann, Nowrouzezahrai, Kahou, Mirmehdi, 2026.
  [arXiv:2606.05328][invisible-hand]
- **Causal Physics Steering in Video World Models via Concept Activation
  Vectors** — Alam, 2026. [arXiv:2605.24322][physics-steering]

### Scene representations in image generators

- **Beyond Surface Statistics: Scene Representations in a Latent Diffusion
  Model** — Chen, Viégas, Wattenberg, 2023.
  [arXiv:2306.05720][beyond-surface]
- **What Does Stable Diffusion Know about the 3D Scene?** (later retitled *A
  General Protocol to Probe Large Vision Models for 3D Physical
  Understanding*) — Zhan, Zheng, Xie, Zisserman, 2023.
  [arXiv:2310.06836][sd-3d-scene]

### Emergent world models and physical understanding

- **Emergent World Representations: Exploring a Sequence Model Trained on a
  Synthetic Task** ("OthelloGPT") — Li, Hopkins, Bau, Viégas, Pfister,
  Wattenberg, 2022. [arXiv:2210.13382][othello-gpt]
- **How Far is Video Generation from World Model: A Physical Law
  Perspective** — Kang et al. (ByteDance), 2024.
  [arXiv:2411.02385][physical-law]
- **Intuitive Physics Understanding Emerges from Self-Supervised Pretraining
  on Natural Videos** — Garrido et al. (Meta), 2025.
  [arXiv:2502.11831][intuitive-physics]
- **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction
  and Planning** — Assran et al. (Meta), 2025. [arXiv:2506.09985][vjepa2]

### Generative modeling background

- **Flow Matching for Generative Modeling** — Lipman, Chen, Ben-Hamu, Nickel,
  Le, 2022. [arXiv:2210.02747][flow-matching]
- **Flow Straight and Fast: Learning to Generate and Transfer Data with
  Rectified Flow** — Liu, Gong, Liu, 2022. [arXiv:2209.03003][rectified-flow]
- **Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion** —
  Chen, Martí Monsó, Du, Simchowitz, Tedrake, Sitzmann, 2024.
  [arXiv:2407.01392][diffusion-forcing]

[mira]: https://mira-wm.com/blog-post/
[interpreting-physics]: https://arxiv.org/abs/2602.07050
[invisible-hand]: https://arxiv.org/abs/2606.05328
[physics-steering]: https://arxiv.org/abs/2605.24322
[beyond-surface]: https://arxiv.org/abs/2306.05720
[sd-3d-scene]: https://arxiv.org/abs/2310.06836
[othello-gpt]: https://arxiv.org/abs/2210.13382
[physical-law]: https://arxiv.org/abs/2411.02385
[intuitive-physics]: https://arxiv.org/abs/2502.11831
[vjepa2]: https://arxiv.org/abs/2506.09985
[flow-matching]: https://arxiv.org/abs/2210.02747
[rectified-flow]: https://arxiv.org/abs/2209.03003
[diffusion-forcing]: https://arxiv.org/abs/2407.01392
