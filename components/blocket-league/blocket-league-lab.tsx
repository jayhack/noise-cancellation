"use client";

import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  Braces,
  CircleDot,
  Cpu,
  Eye,
  EyeOff,
  Gauge,
  Pause,
  Play,
  RotateCcw,
  ScanLine,
  Sparkles,
} from "lucide-react";

import {
  ACTION_NAMES,
  autopilotAction,
  createWorld,
  keyboardAction,
  resetRound,
  snapshotWorld,
  stepWorld,
  WORLD,
  type Vec2,
  type WorldState,
} from "@/lib/blocket-league/sim";

import styles from "./blocket-league-lab.module.css";
import { CodecViewer } from "./codec-viewer";
import { InterpretabilityViewer } from "./interpretability-viewer";
import { LiveWorldModel } from "./live-world-model";
import { LongRolloutViewer } from "./long-rollout-viewer";
import { TrajectoryViewer } from "./trajectory-viewer";

const PAD_ACTIONS = [8, 1, 2, 7, 0, 3, 6, 5, 4];
const PAD_LABELS = ["↖", "↑", "↗", "←", "·", "→", "↙", "↓", "↘"];

type TrailPoint = { player: Vec2; puck: Vec2 };

function drawArrow(
  context: CanvasRenderingContext2D,
  origin: Vec2,
  velocity: Vec2,
  color: string,
) {
  const scale = 0.13;
  const end = { x: origin.x + velocity.x * scale, y: origin.y + velocity.y * scale };
  const length = Math.hypot(end.x - origin.x, end.y - origin.y);
  if (length < 0.008) return;
  const angle = Math.atan2(end.y - origin.y, end.x - origin.x);
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 0.008;
  context.beginPath();
  context.moveTo(origin.x, origin.y);
  context.lineTo(end.x, end.y);
  context.stroke();
  context.beginPath();
  context.moveTo(end.x, end.y);
  context.lineTo(end.x - Math.cos(angle - 0.52) * 0.026, end.y - Math.sin(angle - 0.52) * 0.026);
  context.lineTo(end.x - Math.cos(angle + 0.52) * 0.026, end.y - Math.sin(angle + 0.52) * 0.026);
  context.closePath();
  context.fill();
}

function drawDisc(
  context: CanvasRenderingContext2D,
  position: Vec2,
  radius: number,
  fill: string,
  core: string,
) {
  context.shadowColor = "rgba(0, 0, 0, 0.38)";
  context.shadowBlur = 0.022;
  context.shadowOffsetY = 0.012;
  context.fillStyle = fill;
  context.beginPath();
  context.arc(position.x, position.y, radius, 0, Math.PI * 2);
  context.fill();
  context.shadowColor = "transparent";
  context.fillStyle = core;
  context.beginPath();
  context.arc(position.x, position.y, radius * 0.31, 0, Math.PI * 2);
  context.fill();
}

function drawWorld(
  canvas: HTMLCanvasElement,
  state: WorldState,
  showVectors: boolean,
  trail: TrailPoint[],
) {
  const bounds = canvas.getBoundingClientRect();
  if (bounds.width <= 0) return;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const side = bounds.width;
  const width = Math.round(side * pixelRatio);
  if (canvas.width !== width || canvas.height !== width) {
    canvas.width = width;
    canvas.height = width;
  }
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, side, side);
  context.fillStyle = "#070b10";
  context.fillRect(0, 0, side, side);

  context.save();
  context.scale(side, side);
  const wall = WORLD.wall;
  context.fillStyle = "#0c2022";
  context.fillRect(wall, wall, 1 - wall * 2, 1 - wall * 2);
  const wash = context.createRadialGradient(0.5, 0.5, 0.05, 0.5, 0.5, 0.7);
  wash.addColorStop(0, "rgba(29, 78, 77, 0.16)");
  wash.addColorStop(1, "rgba(3, 9, 12, 0.08)");
  context.fillStyle = wash;
  context.fillRect(wall, wall, 1 - wall * 2, 1 - wall * 2);

  context.strokeStyle = "rgba(78, 132, 126, 0.32)";
  context.lineWidth = 0.006;
  context.beginPath();
  context.moveTo(0.5, wall);
  context.lineTo(0.5, 1 - wall);
  context.stroke();
  context.beginPath();
  context.arc(0.5, 0.5, 0.14, 0, Math.PI * 2);
  context.stroke();

  context.fillStyle = "rgba(238, 181, 62, 0.18)";
  context.fillRect(1 - wall - 0.045, WORLD.goalLow, 0.045, WORLD.goalHigh - WORLD.goalLow);
  context.strokeStyle = "#526769";
  context.lineWidth = 0.013;
  context.lineCap = "round";
  context.beginPath();
  context.moveTo(wall, wall);
  context.lineTo(1 - wall, wall);
  context.moveTo(wall, 1 - wall);
  context.lineTo(1 - wall, 1 - wall);
  context.moveTo(wall, wall);
  context.lineTo(wall, 1 - wall);
  context.moveTo(1 - wall, wall);
  context.lineTo(1 - wall, WORLD.goalLow);
  context.moveTo(1 - wall, WORLD.goalHigh);
  context.lineTo(1 - wall, 1 - wall);
  context.stroke();
  context.strokeStyle = "#eeb53e";
  context.lineWidth = 0.012;
  context.beginPath();
  context.moveTo(1 - wall, WORLD.goalLow);
  context.lineTo(1 - wall, WORLD.goalHigh);
  context.stroke();

  if (showVectors && trail.length > 1) {
    context.lineWidth = 0.007;
    context.lineCap = "round";
    context.strokeStyle = "rgba(50, 213, 173, 0.22)";
    context.beginPath();
    trail.forEach((point, index) => {
      if (index === 0) context.moveTo(point.player.x, point.player.y);
      else context.lineTo(point.player.x, point.player.y);
    });
    context.stroke();
    context.strokeStyle = "rgba(239, 242, 233, 0.18)";
    context.beginPath();
    trail.forEach((point, index) => {
      if (index === 0) context.moveTo(point.puck.x, point.puck.y);
      else context.lineTo(point.puck.x, point.puck.y);
    });
    context.stroke();
  }

  drawDisc(context, state.playerPosition, WORLD.playerRadius, "#32d5ad", "#0b3934");
  drawDisc(context, state.puckPosition, WORLD.puckRadius, "#eff2e9", "#96a199");
  if (showVectors) {
    drawArrow(context, state.playerPosition, state.playerVelocity, "rgba(50, 213, 173, 0.88)");
    drawArrow(context, state.puckPosition, state.puckVelocity, "rgba(239, 242, 233, 0.78)");
  }

  if (state.resetTimer > 0) {
    context.fillStyle = "rgba(7, 11, 16, 0.72)";
    context.fillRect(0.3, 0.43, 0.4, 0.14);
    context.fillStyle = "#eeb53e";
    context.font = "500 0.055px Geist, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText("GOAL", 0.5, 0.5);
  }
  context.restore();
}

function format(value: number) {
  const fixed = Math.abs(value) < 0.0005 ? 0 : value;
  return fixed.toFixed(2);
}

export function BlocketLeagueLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [initialWorld] = useState(() => createWorld(17));
  const worldRef = useRef<WorldState>(initialWorld);
  const trailRef = useRef<TrailPoint[]>([]);
  const keysRef = useRef(new Set<string>());
  const manualActionRef = useRef<number | null>(null);
  const [snapshot, setSnapshot] = useState(() => snapshotWorld(initialWorld));
  const [currentAction, setCurrentAction] = useState(0);
  const [paused, setPaused] = useState(false);
  const [autoplay, setAutoplay] = useState(true);
  const [showVectors, setShowVectors] = useState(true);

  useEffect(() => {
    const movementKeys = new Set(["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft", "w", "a", "s", "d"]);
    const keyDown = (event: KeyboardEvent) => {
      if (!movementKeys.has(event.key)) return;
      event.preventDefault();
      keysRef.current.add(event.key);
      setAutoplay(false);
    };
    const keyUp = (event: KeyboardEvent) => {
      if (!movementKeys.has(event.key)) return;
      keysRef.current.delete(event.key);
    };
    window.addEventListener("keydown", keyDown);
    window.addEventListener("keyup", keyUp);
    return () => {
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
    };
  }, []);

  useEffect(() => {
    let animationFrame = 0;
    let previous = performance.now();
    let accumulator = 0;
    const stepDuration = 1_000 / WORLD.fps;

    const animate = (now: number) => {
      const world = worldRef.current;
      accumulator = Math.min(accumulator + now - previous, 250);
      previous = now;
      let advanced = false;
      while (!paused && accumulator >= stepDuration) {
        const action =
          manualActionRef.current ?? (autoplay ? autopilotAction(world) : keyboardAction(keysRef.current));
        setCurrentAction(action);
        stepWorld(world, action);
        trailRef.current.push({
          player: { ...world.playerPosition },
          puck: { ...world.puckPosition },
        });
        if (trailRef.current.length > 30) trailRef.current.shift();
        accumulator -= stepDuration;
        advanced = true;
      }
      if (advanced) setSnapshot(snapshotWorld(world));
      if (canvasRef.current) drawWorld(canvasRef.current, world, showVectors, trailRef.current);
      animationFrame = requestAnimationFrame(animate);
    };
    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, [autoplay, paused, showVectors]);

  const reset = () => {
    const world = worldRef.current;
    resetRound(world, true);
    trailRef.current = [];
    setCurrentAction(0);
    setSnapshot(snapshotWorld(world));
  };

  const beginManualAction = (action: number) => {
    manualActionRef.current = action;
    setCurrentAction(action);
    setAutoplay(false);
    setPaused(false);
  };

  return (
    <main className={styles.root}>
      <header className={styles.header}>
        <a className={styles.wordmark} href="#top" aria-label="Blocket League home">
          <span className={styles.mark}><CircleDot aria-hidden="true" /></span>
          <span>BLOCKET LEAGUE</span>
        </a>
        <div className={styles.headerMeta}>
          <span className={styles.statusDot} />
          WORLD 01 · LIVE
        </div>
      </header>

      <section className={styles.hero} id="top">
        <p className={styles.eyebrow}>A MINIMAL WORLD MODEL EXPERIMENT</p>
        <h1>Can a tiny world model<br />discover the rules?</h1>
        <p className={styles.heroCopy}>
          Start with MIRA-style latent diffusion. Replace it with a faster direct transformer.
          Then read—and rewrite—the physics hidden inside.
        </p>
        <a className={styles.jumpLink} href="#world">
          Enter the world <ArrowRight aria-hidden="true" />
        </a>
      </section>

      <section className={styles.labSection} id="world" aria-labelledby="world-title">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.sectionIndex}>01 / THE WORLD</p>
            <h2 id="world-title">Play the training distribution.</h2>
          </div>
          <p>
            Use WASD or the arrow keys. Velocity is deliberately invisible to the model; turn on X-ray
            mode to see the state it must reconstruct across frames.
          </p>
        </div>

        <div className={styles.simulatorShell}>
          <div className={styles.canvasColumn}>
            <div className={styles.canvasHeader}>
              <div className={styles.liveLabel}><span /> SIMULATOR</div>
              <div className={styles.score}>GOALS <strong>{snapshot.score.toString().padStart(2, "0")}</strong></div>
            </div>
            <canvas
              ref={canvasRef}
              className={styles.canvas}
              role="img"
              aria-label="A playable square arena with a teal player disc, a white puck, and a gold goal on the right."
            />
            <div className={styles.timeline} aria-hidden="true">
              {Array.from({ length: 24 }, (_, index) => <span key={index} className={index > 18 ? styles.timelineFuture : undefined} />)}
            </div>
          </div>

          <aside className={styles.controlColumn} aria-label="Simulator controls and state">
            <div className={styles.controlTop}>
              <div>
                <span className={styles.controlLabel}>CONTROL</span>
                <strong>{autoplay ? "AUTOPILOT" : "MANUAL"}</strong>
              </div>
              <button
                className={`${styles.iconButton} ${autoplay ? styles.iconButtonActive : ""}`}
                type="button"
                onClick={() => setAutoplay((value) => !value)}
                aria-pressed={autoplay}
                aria-label="Toggle autopilot"
              >
                <Sparkles aria-hidden="true" />
              </button>
            </div>

            <div className={styles.pad} aria-label="Direction pad">
              {PAD_ACTIONS.map((action, index) => (
                <button
                  key={action}
                  type="button"
                  className={action === 0 ? styles.padCenter : undefined}
                  aria-label={ACTION_NAMES[action]}
                  onPointerDown={() => beginManualAction(action)}
                  onPointerUp={() => { manualActionRef.current = null; }}
                  onPointerCancel={() => { manualActionRef.current = null; }}
                  onPointerLeave={() => { manualActionRef.current = null; }}
                >
                  {PAD_LABELS[index]}
                </button>
              ))}
            </div>

            <div className={styles.transport}>
              <button type="button" onClick={() => setPaused((value) => !value)}>
                {paused ? <Play aria-hidden="true" /> : <Pause aria-hidden="true" />}
                {paused ? "Resume" : "Pause"}
              </button>
              <button type="button" onClick={reset}>
                <RotateCcw aria-hidden="true" /> Reset
              </button>
            </div>

            <button
              type="button"
              className={`${styles.xrayButton} ${showVectors ? styles.xrayActive : ""}`}
              onClick={() => setShowVectors((value) => !value)}
              aria-pressed={showVectors}
            >
              {showVectors ? <Eye aria-hidden="true" /> : <EyeOff aria-hidden="true" />}
              <span><strong>X-RAY STATE</strong><small>velocity vectors + trails</small></span>
              <span className={styles.switchTrack}><span /></span>
            </button>

            <div className={styles.readout}>
              <div className={styles.readoutHeading}>
                <ScanLine aria-hidden="true" /> PRIVILEGED STATE
              </div>
              <dl>
                <div><dt>player p</dt><dd>{format(snapshot.playerPosition.x)}, {format(snapshot.playerPosition.y)}</dd></div>
                <div><dt>player v</dt><dd>{format(snapshot.playerVelocity.x)}, {format(snapshot.playerVelocity.y)}</dd></div>
                <div><dt>puck p</dt><dd>{format(snapshot.puckPosition.x)}, {format(snapshot.puckPosition.y)}</dd></div>
                <div><dt>puck v</dt><dd>{format(snapshot.puckVelocity.x)}, {format(snapshot.puckVelocity.y)}</dd></div>
              </dl>
              <div className={styles.eventLine}>
                <span className={styles.eventPulse} />
                {snapshot.lastEvent.toUpperCase()} · {ACTION_NAMES[currentAction].toUpperCase()}
              </div>
            </div>
          </aside>
        </div>
      </section>

      <section className={styles.codecSection} aria-labelledby="codec-title">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.sectionIndex}>02 / REPRESENTATION CODEC</p>
            <h2 id="codec-title">Compress pixels. Keep the game.</h2>
          </div>
          <p>
            A frozen DINO encoder distills every pair of RGB frames into one compact 8 × 8 latent.
            The learned causal decoder must preserve the tiny bodies and arena well enough for dynamics.
          </p>
        </div>
        <CodecViewer />
      </section>

      <section className={styles.trajectorySection} aria-labelledby="trajectory-title">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.sectionIndex}>03 / TRAJECTORY THEATER</p>
            <h2 id="trajectory-title">One past. Several possible futures.</h2>
          </div>
          <p>
            First, test a MIRA-style latent diffusion model. It predicts one two-frame code at a time,
            feeds each prediction back as context, and branches into several plausible futures.
          </p>
        </div>
        <TrajectoryViewer />
      </section>

      <section className={styles.modelSection} aria-labelledby="model-title">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.sectionIndex}>04 / THE MODEL</p>
            <h2 id="model-title">A small network with nowhere to hide.</h2>
          </div>
          <p>
            Our first model is a latent diffusion transformer: six clean frames establish state,
            then flow matching denoises each future latent while previous predictions become context.
          </p>
        </div>

        <div className={styles.pipeline}>
          <div className={styles.pipelineNode}>
            <span><Braces aria-hidden="true" /></span>
            <p>OBSERVE</p>
            <strong>6 RGB frames</strong>
            <small>position implies velocity</small>
          </div>
          <ArrowRight className={styles.pipelineArrow} aria-hidden="true" />
          <div className={styles.pipelineNode}>
            <span><ScanLine aria-hidden="true" /></span>
            <p>COMPRESS</p>
            <strong>DINO RAE</strong>
            <small>2 RGB → 1 × 8 × 8 × 32</small>
          </div>
          <ArrowRight className={styles.pipelineArrow} aria-hidden="true" />
          <div className={`${styles.pipelineNode} ${styles.pipelineCore}`}>
            <span><Cpu aria-hidden="true" /></span>
            <p>DENOISE</p>
            <strong>Latent DiT</strong>
            <small>flow matching + action pairs</small>
          </div>
          <ArrowRight className={styles.pipelineArrow} aria-hidden="true" />
          <div className={styles.pipelineNode}>
            <span><Activity aria-hidden="true" /></span>
            <p>DECODE</p>
            <strong>24 future frames</strong>
            <small>2-frame causal autoregression</small>
          </div>
          <div className={styles.probeRail}>
            <div><ScanLine aria-hidden="true" /> linear probes</div>
            <div><Gauge aria-hidden="true" /> causal edits</div>
          </div>
        </div>
      </section>

      <section className={styles.entropySection} aria-labelledby="entropy-title">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.sectionIndex}>05 / FAILURE HORIZON</p>
            <h2 id="entropy-title">Freeze the weights. Keep rolling.</h2>
          </div>
          <p>
            Push that diffusion checkpoint to 64 frames—more than five times its trained horizon.
            The palette decoder stays crisp while the dynamics drift toward four incompatible worlds,
            separating representation quality from world-model accuracy.
          </p>
        </div>
        <LongRolloutViewer />
      </section>

      <section className={styles.liveSection} aria-labelledby="live-title">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.sectionIndex}>06 / PLAY THE MODEL</p>
            <h2 id="live-title">Drive the frozen dream.</h2>
          </div>
          <p>
            Diffusion works, but its iterative solver is too slow for play. A second model directly
            regresses the next latent in one transformer pass. Load it onto your browser GPU and steer
            with WASD—no simulator or diffusion solver runs underneath it.
          </p>
        </div>
        <LiveWorldModel />
      </section>

      <section className={styles.interpretabilitySection} aria-labelledby="interpretability-title">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.sectionIndex}>07 / OPEN THE MODEL</p>
            <h2 id="interpretability-title">Can we find—and steer—the physics?</h2>
          </div>
          <p>
            Now open the direct transformer that powers the browser game. Linear probes test whether
            position and velocity are readable; activation writes test whether those representations
            actually control the hallucinated physics.
          </p>
        </div>
        <InterpretabilityViewer />
      </section>

      <section className={styles.scaleSection} aria-labelledby="scale-title">
        <div className={styles.scaleIntro}>
          <p className={styles.sectionIndex}>08 / NEXT WORLDS</p>
          <h2 id="scale-title">Add one difficulty at a time.</h2>
        </div>
        <ol className={styles.ladder}>
          <li className={styles.ladderActive}><span>01</span><strong>Strike</strong><small>force · drag · impact</small></li>
          <li><span>02</span><strong>Score</strong><small>events · resets · memory</small></li>
          <li><span>03</span><strong>Defend</strong><small>coupled agents</small></li>
          <li><span>04</span><strong>Occlude</strong><small>object permanence</small></li>
          <li><span>05</span><strong>Shift</strong><small>variable world rules</small></li>
        </ol>
      </section>

      <footer className={styles.footer}>
        <span>BLOCKET LEAGUE · WORLD 01</span>
        <span>PIXELS → REPRESENTATION → PHYSICS → INTERVENTION</span>
      </footer>
    </main>
  );
}
