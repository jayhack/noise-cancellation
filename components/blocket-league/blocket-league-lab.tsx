"use client";

import { useEffect, useRef, useState } from "react";
import Image from "next/image";
import {
  ArrowRight,
  CircleDot,
  Eye,
  EyeOff,
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

const MODEL_HISTORY = [
  { label: "t−7", player: [24, 68], puck: [72, 28] },
  { label: "t−6", player: [28, 63], puck: [69, 31] },
  { label: "t−5", player: [33, 58], puck: [65, 35] },
  { label: "t−4", player: [38, 53], puck: [61, 39] },
  { label: "t−3", player: [43, 48], puck: [57, 43] },
  { label: "t−2", player: [48, 43], puck: [53, 47] },
  { label: "t−1", player: [53, 38], puck: [49, 51] },
  { label: "t", player: [58, 33], puck: [45, 55] },
] as const;

function DiagramFrame({
  label,
  player,
  puck,
  predicted = false,
}: {
  label: string;
  player: readonly [number, number];
  puck: readonly [number, number];
  predicted?: boolean;
}) {
  return (
    <div className={`${styles.diagramFrame} ${predicted ? styles.diagramFramePredicted : ""}`}>
      <span className={styles.diagramGoal} />
      <span
        className={styles.diagramPlayer}
        style={{ left: `${player[0]}%`, top: `${player[1]}%` }}
      />
      <span
        className={styles.diagramPuck}
        style={{ left: `${puck[0]}%`, top: `${puck[1]}%` }}
      />
      <small>{label}</small>
    </div>
  );
}

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
            <p className={styles.sectionIndex}>02 / THE ARCHITECTURE</p>
            <h2 id="model-title">Previous frames become the next.</h2>
          </div>
          <p>
            The transformer is trained like a visual next-token predictor: previous rendered frames
            go in, and it predicts the pixels of the next. There are no actions or coordinates in the
            input. During a rollout, each predicted image becomes part of the history, so position,
            motion, collisions, and resets must survive through the model&apos;s own pixels.
          </p>
        </div>

        <div
          className={styles.architectureDiagram}
          role="img"
          aria-label="A perspective stack of previous pixel frames is patch embedded into one six-block causal transformer, which predicts the next categorical image. The prediction is then appended to the history for the next step."
        >
          <div className={styles.architectureFlow}>
            <div className={styles.architectureInput}>
              <div className={styles.architectureLabel}>
                <span>PREVIOUS FRAMES</span>
                <strong>… x<sub>t−2</sub>, x<sub>t−1</sub>, x<sub>t</sub></strong>
                <small>64 × 64 categorical images · no actions</small>
              </div>
              <div className={styles.historyPerspective} aria-hidden="true">
                {MODEL_HISTORY.map((frame, index) => (
                  <DiagramFrame
                    key={frame.label}
                    {...frame}
                    label={["earlier", "", "", "…", "", "t−2", "t−1", "t"][index]}
                  />
                ))}
              </div>
              <div className={styles.convergenceLines} aria-hidden="true">
                {Array.from({ length: 8 }, (_, index) => <span key={index} />)}
              </div>
            </div>

            <div className={styles.architectureEdge}>
              <span>4 × 4 PATCH EMBEDDING</span>
              <ArrowRight aria-hidden="true" />
            </div>

            <div className={`${styles.architectureNode} ${styles.transformerNode}`}>
              <span>PIXEL TRANSFORMER</span>
              <strong>6 causal blocks</strong>
              <small>space attention → time attention → MLP</small>
              <div className={styles.blockStack} aria-hidden="true">
                {Array.from({ length: 6 }, (_, index) => <i key={index}>B{index + 1}</i>)}
              </div>
              <em>3.67M parameters · no action input</em>
            </div>

            <ArrowRight className={styles.architectureArrow} aria-hidden="true" />

            <div className={styles.architecturePrediction}>
              <span>NEXT FRAME</span>
              <DiagramFrame
                label="x̂t+1"
                player={[63, 29]}
                puck={[41, 59]}
                predicted
              />
              <small>64 × 64 × 9 logits → argmax</small>
            </div>
          </div>

          <div className={styles.feedbackRail}>
            <span>ROLL FORWARD</span>
            <strong>the predicted image becomes a previous frame</strong>
            <span>REPEAT</span>
          </div>
        </div>
      </section>

      <section className={styles.lensSection} aria-labelledby="lens-title">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.sectionIndex}>03 / TAKE THE LENS</p>
            <h2 id="lens-title">Average what this activation can cause downstream.</h2>
          </div>
          <p>
            The <a href="https://transformer-circuits.pub/2026/workspace/index.html#the-jacobian-lens" target="_blank" rel="noreferrer">J-space paper&apos;s Jacobian lens</a> averages how an intermediate activation changes future verbal outputs. Our physics analogue replaces words with the green disc&apos;s rendered x/y position.
          </p>
        </div>

        <div
          className={styles.lensDiagram}
          role="img"
          aria-label="Across 512 rendered contexts, select the block five activation at the green disc's spatial token, run the frozen downstream model to the next rendered frame, backpropagate the green centroid's x and y coordinates, and average those gradients into global x and y steering directions."
        >
          <div className={styles.lensFlow}>
            <div className={`${styles.lensStage} ${styles.lensContexts}`}>
              <div className={styles.diagramStageHeader}>
                <span>01 / SAMPLE</span>
                <strong>512 worlds</strong>
                <small>separate fit contexts</small>
              </div>
              <div className={styles.contextFan} aria-hidden="true">
                {MODEL_HISTORY.slice(2, 5).map((frame, index) => (
                  <DiagramFrame key={frame.label} {...frame} label={`world ${index + 1}`} />
                ))}
              </div>
            </div>

            <ArrowRight className={styles.lensArrow} aria-hidden="true" />

            <div className={`${styles.lensStage} ${styles.lensActivation}`}>
              <div className={styles.diagramStageHeader}>
                <span>02 / LOCATE</span>
                <strong>h<sub>ℓ,p</sub> at block 5</strong>
                <small>p = green spatial token</small>
              </div>
              <div className={styles.activationGrid} aria-hidden="true">
                {Array.from({ length: 25 }, (_, index) => (
                  <span key={index} className={index === 17 ? styles.activationCellActive : undefined} />
                ))}
              </div>
              <div className={styles.activationVector}>192D activation</div>
            </div>

            <div className={styles.jacobianBridge}>
              <div className={styles.forwardRail}><span>FROZEN FORWARD</span><ArrowRight aria-hidden="true" /></div>
              <div className={styles.bridgeBlocks}>
                <span>B6</span><span>NORM</span><span>PIXEL HEAD</span>
              </div>
              <div className={styles.backwardRail}><ArrowRight aria-hidden="true" /><span>BACKPROP ∂ŷ / ∂h</span></div>
            </div>

            <div className={`${styles.lensStage} ${styles.lensReadout}`}>
              <div className={styles.diagramStageHeader}>
                <span>03 / MEASURE</span>
                <strong>Next-frame centroid</strong>
                <small>soft readout from green logits</small>
              </div>
              <div className={styles.centroidBoard} aria-hidden="true">
                <span className={styles.centroidDisc} />
                <span className={styles.centroidCrossX} />
                <span className={styles.centroidCrossY} />
              </div>
              <div className={styles.centroidCoordinates}>ŷ = (x̂, ŷ)</div>
            </div>

            <ArrowRight className={styles.lensArrow} aria-hidden="true" />

            <div className={`${styles.lensStage} ${styles.lensDirections}`}>
              <div className={styles.diagramStageHeader}>
                <span>04 / AVERAGE</span>
                <strong>Global directions</strong>
                <small>reusable across unseen rollouts</small>
              </div>
              <div className={styles.directionAxes} aria-hidden="true">
                <div><span>v<sub>x</sub></span><i>→</i></div>
                <div><span>v<sub>y</sub></span><i>↓</i></div>
              </div>
              <div className={styles.directionWrite}>h ← h + αv</div>
            </div>
          </div>

          <div className={styles.lensEquation}>
            <span>PHYSICS J-LENS</span>
            <strong>v<sub>x</sub> = normalize [ 1/K · Σ<sub>i</sub> ∂x̂<sub>i,t+1</sub> / ∂h<sub>i,ℓ,p</sub> ]</strong>
            <small>Repeat for y. Average across contexts to remove rollout-specific accidents and retain a reusable downstream effect.</small>
          </div>

          <div className={styles.lensComparison}>
            <div>
              <span>ANTHROPIC J-LENS</span>
              <strong>activation → future final residuals → vocabulary logits</strong>
              <small>Average across prompts, source positions, and future positions.</small>
            </div>
            <div>
              <span>OUR MOTION LENS</span>
              <strong>player token → next-frame pixels → rendered centroid</strong>
              <small>Average across worlds; then write the resulting x/y direction back into the same token.</small>
            </div>
          </div>
        </div>
      </section>

      <section className={styles.trajectorySection} aria-labelledby="prediction-title">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.sectionIndex}>04 / THE PREDICTION</p>
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
            <p className={styles.sectionIndex}>05 / OPEN THE MODEL</p>
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
            <p className={styles.sectionIndex}>06 / PLAY THE INTERVENTION</p>
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
          <p className={styles.sectionIndex}>07 / NEXT WORLDS</p>
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
