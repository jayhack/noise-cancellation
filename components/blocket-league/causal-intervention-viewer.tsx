"use client";

import { useEffect, useRef, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";

import styles from "./blocket-league-lab.module.css";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

type Scenario = {
  id: string;
  title: string;
  atlas: string;
  finalSeparationPx: number;
  meanVelocityDeltaPxPerFrame: number;
  baselinePuckXY: [number, number][];
  activatedPuckXY: [number, number][];
};

type Manifest = {
  frameSize: number;
  frames: number;
  playbackFps: number;
  sourceLayer: string;
  writeStrengthSigmas: number;
  directionContexts: number;
  locatorAccuracy: number;
  scenarios: Scenario[];
};

function InterventionBoard({
  image,
  frame,
  row,
  frameSize,
  positions,
  label,
  activated,
}: {
  image: HTMLImageElement | null;
  frame: number;
  row: number;
  frameSize: number;
  positions: [number, number][];
  label: string;
  activated: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !image) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, frameSize, frameSize);
    context.drawImage(image, frame * frameSize, row * frameSize, frameSize, frameSize,
      0, 0, frameSize, frameSize);
    const trail = positions.slice(0, frame + 1);
    if (trail.length > 1) {
      context.beginPath();
      context.moveTo(trail[0][0], trail[0][1]);
      for (const [x, y] of trail.slice(1)) context.lineTo(x, y);
      context.strokeStyle = activated ? "rgba(255, 177, 64, .9)" : "rgba(115, 235, 211, .72)";
      context.lineWidth = 0.8;
      context.stroke();
    }
  }, [activated, frame, frameSize, image, positions, row]);
  return <canvas ref={ref} width={frameSize} height={frameSize} role="img"
    aria-label={`${label}, frame ${frame + 1}`} className={styles.interventionCanvas} />;
}

export function CausalInterventionViewer() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [scenarioIndex, setScenarioIndex] = useState(0);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${BASE_PATH}/blocket-league/interventions/manifest.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`Intervention manifest returned ${response.status}`);
        return response.json() as Promise<Manifest>;
      })
      .then(setManifest)
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, []);

  const scenario = manifest?.scenarios[scenarioIndex];
  useEffect(() => {
    if (!scenario) return;
    const next = new Image();
    next.onload = () => setImage(next);
    next.onerror = () => setError("Intervention atlas could not be loaded");
    next.src = `${BASE_PATH}${scenario.atlas}`;
  }, [scenario]);

  useEffect(() => {
    if (!manifest || !playing) return;
    const timer = window.setInterval(() => {
      setFrame((current) => (current + 1) % manifest.frames);
    }, 1000 / manifest.playbackFps);
    return () => window.clearInterval(timer);
  }, [manifest, playing]);

  if (error) return <p className={styles.interpretabilityLoading}>Causal film unavailable: {error}</p>;
  if (!manifest || !scenario) return <p className={styles.interpretabilityLoading}>Rendering matched causal films…</p>;

  const separation = scenario.activatedPuckXY[frame][0] - scenario.baselinePuckXY[frame][0];
  return (
    <section className={styles.interventionFilm} aria-label="Matched causal intervention rollouts">
      <div className={styles.interventionFilmHeader}>
        <div>
          <span>PIXEL-LEVEL CAUSAL FILM · {manifest.sourceLayer.toUpperCase()}</span>
          <h3>Inject a phantom force.</h3>
          <p>Same observed world. Same coast actions. Same frozen weights. Only the hidden velocity direction changes.</p>
        </div>
        <div className={styles.interventionScenarioPicker} role="group" aria-label="Choose held-out causal world">
          {manifest.scenarios.map((item, index) => (
            <button key={item.id} type="button" aria-pressed={scenarioIndex === index}
              onClick={() => { setScenarioIndex(index); setFrame(0); }}>{index + 1}</button>
          ))}
        </div>
      </div>

      <div className={styles.interventionBoards}>
        <article>
          <div><span>BEFORE · BASELINE</span><strong>no activation write</strong></div>
          <InterventionBoard image={image} frame={frame} row={0} frameSize={manifest.frameSize}
            positions={scenario.baselinePuckXY} label="Baseline hallucinated board" activated={false} />
        </article>
        <div className={styles.interventionArrow} aria-hidden="true">→</div>
        <article className={styles.interventionBoardActive}>
          <div><span>AFTER · VECTOR ON</span><strong>+{manifest.writeStrengthSigmas.toFixed(0)}σ every transition</strong></div>
          <InterventionBoard image={image} frame={frame} row={1} frameSize={manifest.frameSize}
            positions={scenario.activatedPuckXY} label="Velocity-intervened hallucinated board" activated />
        </article>
      </div>

      <div className={styles.interventionReadout}>
        <div><span>FRAME</span><strong>{String(frame).padStart(2, "0")} / {manifest.frames - 1}</strong></div>
        <div><span>PUCK X SEPARATION</span><strong>{separation >= 0 ? "+" : ""}{separation.toFixed(1)} px</strong></div>
        <div><span>FINAL SEPARATION</span><strong>+{scenario.finalSeparationPx.toFixed(1)} px</strong></div>
        <div><span>IMPLIED ΔVX</span><strong>+{scenario.meanVelocityDeltaPxPerFrame.toFixed(2)} px/f</strong></div>
      </div>

      <div className={styles.interventionTransport}>
        <button type="button" aria-label={playing ? "Pause causal film" : "Play causal film"}
          onClick={() => setPlaying((value) => !value)}>{playing ? <Pause /> : <Play />}</button>
        <button type="button" aria-label="Restart causal film" onClick={() => setFrame(0)}><RotateCcw /></button>
        <input type="range" min={0} max={manifest.frames - 1} value={frame}
          aria-label="Causal film frame" onChange={(event) => { setFrame(Number(event.target.value)); setPlaying(false); }} />
      </div>
      <p className={styles.interventionMethod}>The player receives only COAST. The injected direction was averaged over {manifest.directionContexts} other worlds; the activation-only puck locator is {(manifest.locatorAccuracy * 100).toFixed(1)}% accurate.</p>
    </section>
  );
}
