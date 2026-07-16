"use client";

import { useEffect, useRef, useState } from "react";
import Image from "next/image";
import {
  Activity,
  ArrowRight,
  CircleDot,
  Cpu,
  Eye,
  EyeOff,
  Gauge,
  Pause,
  Play,
  RotateCcw,
  ScanLine,
} from "lucide-react";

import {
  createPassiveWorld,
  resetPassiveRound,
  snapshotWorld,
  stepWorld,
  WORLD,
  type Vec2,
  type WorldState,
} from "@/lib/blocket-league/sim";

import styles from "./blocket-league-lab.module.css";
import { LiveWorldModel } from "./live-world-model";
import { PixelInterpretabilityViewer } from "./pixel-interpretability-viewer";

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
  const [initialWorld] = useState(() => createPassiveWorld(17));
  const worldRef = useRef<WorldState>(initialWorld);
  const trailRef = useRef<TrailPoint[]>([]);
  const [snapshot, setSnapshot] = useState(() => snapshotWorld(initialWorld));
  const [paused, setPaused] = useState(false);
  const [showVectors, setShowVectors] = useState(true);

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
        stepWorld(world, 0, true);
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
  }, [paused, showVectors]);

  const reset = () => {
    const world = worldRef.current;
    resetPassiveRound(world, true);
    trailRef.current = [];
    setSnapshot(snapshotWorld(world));
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
          Train a transformer only to watch pixels move. Find the velocity it invents.
          Then turn that hidden direction into the controls it never saw during training.
        </p>
        <a className={styles.jumpLink} href="#world">
          Enter the world <ArrowRight aria-hidden="true" />
        </a>
      </section>

      <section className={styles.labSection} id="world" aria-labelledby="world-title">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.sectionIndex}>01 / THE WORLD</p>
            <h2 id="world-title">A world with no controls.</h2>
          </div>
          <p>
            Every clip begins with randomized momentum. After that, the two circles simply coast,
            collide, bounce, and score. Goal-centered clips teach the pause and a moving kickoff
            selected by the visible score. The training set contains pixels—not actions or state vectors.
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

          <aside className={styles.controlColumn} aria-label="Passive simulator state">
            <div className={styles.controlTop}>
              <div>
                <span className={styles.controlLabel}>TRAINING MODE</span>
                <strong>OBSERVATION ONLY</strong>
              </div>
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
                {snapshot.lastEvent.toUpperCase()} · NO INPUT
              </div>
            </div>
          </aside>
        </div>
      </section>

      <section className={styles.modelSection} aria-labelledby="model-title">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.sectionIndex}>02 / THE MODEL</p>
            <h2 id="model-title">Pixels in. Pixels out.</h2>
          </div>
          <p>
            A 3.67-million-parameter causal transformer sees eight exact rendered frames and predicts
            the ninth. There is no encoder, decoder, simulator state, or control token in the path.
          </p>
        </div>

        <div className={styles.pipeline}>
          <div className={styles.pipelineNode}>
            <span><ScanLine aria-hidden="true" /></span>
            <p>OBSERVE</p>
            <strong>8 pixel frames</strong>
            <small>9 exact colors · 64 × 64</small>
          </div>
          <ArrowRight className={styles.pipelineArrow} aria-hidden="true" />
          <div className={`${styles.pipelineNode} ${styles.pipelineCore}`}>
            <span><Cpu aria-hidden="true" /></span>
            <p>PREDICT</p>
            <strong>Pixel transformer</strong>
            <small>6 causal blocks · no actions</small>
          </div>
          <ArrowRight className={styles.pipelineArrow} aria-hidden="true" />
          <div className={styles.pipelineNode}>
            <span><Activity aria-hidden="true" /></span>
            <p>UNROLL</p>
            <strong>1 future frame</strong>
            <small>feed pixels back · repeat</small>
          </div>
          <div className={styles.probeRail}>
            <div><ScanLine aria-hidden="true" /> linear probes</div>
            <div><Gauge aria-hidden="true" /> causal edits</div>
          </div>
        </div>
      </section>

      <section className={styles.trajectorySection} aria-labelledby="prediction-title">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.sectionIndex}>03 / THE PREDICTION</p>
            <h2 id="prediction-title">It keeps both circles in motion.</h2>
          </div>
          <p>
            On 128 unseen worlds, the model averages 0.93 pixels of entity-position error through
            frame 12. It was also trained on its own predicted histories and split-disc corruptions,
            so malformed circles are pulled back toward the game manifold instead of compounding.
          </p>
        </div>
        <figure className={styles.pixelRolloutFigure}>
          <Image src={`${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/blocket-league/passive/rollout.png`} width={2356} height={658} unoptimized alt="Observed context, true future, and pixel-transformer prediction across twelve frames." />
          <figcaption><span>8 observed frames → 12 autonomous predictions</span><span>0.93 px short-horizon · 6.53 px over 64 frames</span></figcaption>
        </figure>
      </section>

      <section className={styles.interpretabilitySection} aria-labelledby="interpretability-title">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.sectionIndex}>04 / OPEN THE MODEL</p>
            <h2 id="interpretability-title">The hidden velocity becomes writable.</h2>
          </div>
          <p>
            Velocity becomes increasingly readable as pixels pass through the six blocks. More
            importantly, one direction found on a fit split changes motion across unseen worlds—even
            after the write stops.
          </p>
        </div>
        <PixelInterpretabilityViewer />
      </section>

      <section className={styles.liveSection} aria-labelledby="live-title">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.sectionIndex}>05 / PLAY THE INTERVENTION</p>
            <h2 id="live-title">The controls are brain surgery.</h2>
          </div>
          <p>
            The model never trained on keystrokes. Here, WASD is mapped directly onto the recovered
            ±x and ±y activation directions for the green circle. The white puck moves only when the
            hallucinated physics says it should.
          </p>
        </div>
        <LiveWorldModel />
      </section>

      <section className={styles.scaleSection} aria-labelledby="scale-title">
        <div className={styles.scaleIntro}>
          <p className={styles.sectionIndex}>06 / NEXT WORLDS</p>
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
        <span>PIXELS → PREDICTION → HIDDEN PHYSICS → INTERVENTION</span>
      </footer>
    </main>
  );
}
