"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";

import styles from "./blocket-league-lab.module.css";

type Lane = {
  id: string;
  label: string;
  kind: "truth" | "sample";
  playerErrorPx: number;
  puckErrorPx: number;
  directPlayerErrorPx?: number;
  directPuckErrorPx?: number;
  rolledPlayerErrorPx?: number;
  rolledPuckErrorPx?: number;
};

type Scenario = {
  id: string;
  title: string;
  description: string;
  seed: number;
  atlas: string;
  actions: string[];
  events: string[];
  lanes: Lane[];
};

type Manifest = {
  frameSize: number;
  contextFrames: number;
  futureFrames: number;
  modelFutureFrames?: number;
  rolloutBoundaries?: number[];
  playbackFps: number;
  ddimSteps: number;
  checkpointStep: number;
  samplerLabel?: string;
  generationLabel?: string;
  latentStepFrames?: number;
  metricBoundary?: number;
  scenarios: Scenario[];
};

const MODEL_VARIANTS = [
  {
    id: "latent-rae",
    label: "RAE latent diffusion",
    detail: "2-frame causal rollout · diffusion forcing",
    manifest: "/blocket-league/trajectories-latent/manifest.json",
  },
  {
    id: "robust-12",
    label: "12-frame robust",
    detail: "direct horizon · rollout trained",
    manifest: "/blocket-league/trajectories-long12/manifest.json",
  },
  {
    id: "baseline-8",
    label: "8-frame baseline",
    detail: "re-fed after frame 8",
    manifest: "/blocket-league/trajectories/manifest.json",
  },
] as const;

function AtlasFrame({
  image,
  frame,
  row,
  frameSize,
  label,
}: {
  image: HTMLImageElement | null;
  frame: number;
  row: number;
  frameSize: number;
  label: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, frameSize, frameSize);
    context.drawImage(
      image,
      frame * frameSize,
      row * frameSize,
      frameSize,
      frameSize,
      0,
      0,
      frameSize,
      frameSize,
    );
  }, [frame, frameSize, image, row]);

  return (
    <canvas
      ref={canvasRef}
      className={styles.trajectoryCanvas}
      width={frameSize}
      height={frameSize}
      role="img"
      aria-label={`${label}, frame ${frame + 1}`}
    />
  );
}

export function TrajectoryViewer() {
  const [manifests, setManifests] = useState<Manifest[] | null>(null);
  const [modelIndex, setModelIndex] = useState(0);
  const [scenarioIndex, setScenarioIndex] = useState(0);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [atlasImage, setAtlasImage] = useState<HTMLImageElement | null>(null);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    Promise.all(
      MODEL_VARIANTS.map((variant) =>
        fetch(variant.manifest).then((response) => {
          if (!response.ok) throw new Error("trajectory manifest unavailable");
          return response.json() as Promise<Manifest>;
        }),
      ),
    )
      .then((values) => {
        if (!cancelled) setManifests(values);
      })
      .catch(() => {
        if (!cancelled) setLoadError(true);
      });
    return () => { cancelled = true; };
  }, []);

  const manifest = manifests?.[modelIndex] ?? null;
  const scenario = manifest?.scenarios[scenarioIndex] ?? null;
  const totalFrames = manifest ? manifest.contextFrames + manifest.futureFrames : 1;
  const modelFutureFrames = manifest?.modelFutureFrames ?? manifest?.futureFrames ?? 1;
  const rolloutBoundaries = manifest?.rolloutBoundaries ?? [];

  useEffect(() => {
    if (!scenario) return;
    let cancelled = false;
    const image = new Image();
    image.onload = () => {
      if (!cancelled) setAtlasImage(image);
    };
    image.src = scenario.atlas;
    return () => { cancelled = true; };
  }, [scenario]);

  useEffect(() => {
    if (!manifest || !playing) return;
    const timer = window.setTimeout(() => {
      setFrame((value) => {
        if (value >= totalFrames - 1) {
          setPlaying(false);
          return value;
        }
        return value + 1;
      });
    }, 1_000 / manifest.playbackFps);
    return () => window.clearTimeout(timer);
  }, [frame, manifest, playing, totalFrames]);

  const isObserved = Boolean(manifest && frame < manifest.contextFrames);
  const futureIndex = manifest ? frame - manifest.contextFrames : -1;
  const isAutoregressive = Boolean(
    manifest && futureIndex >= modelFutureFrames,
  );
  const usesLateMetrics = Boolean(
    manifest && futureIndex >= (manifest.metricBoundary ?? modelFutureFrames),
  );
  const phaseDetail = useMemo(() => {
    if (!scenario || futureIndex < 0) return "shared observation";
    if (manifest?.latentStepFrames) {
      return `latent ${Math.floor(futureIndex / manifest.latentStepFrames) + 1} · ${scenario.actions[futureIndex]} · ${scenario.events[futureIndex]}`;
    }
    const prefix = futureIndex >= modelFutureFrames ? "pass 2 · " : "pass 1 · ";
    return `${prefix}${scenario.actions[futureIndex]} · ${scenario.events[futureIndex]}`;
  }, [futureIndex, manifest?.latentStepFrames, modelFutureFrames, scenario]);

  const chooseScenario = (index: number) => {
    setScenarioIndex(index);
    setAtlasImage(null);
    setFrame(0);
    setPlaying(true);
  };

  const chooseModel = (index: number) => {
    setModelIndex(index);
    setScenarioIndex(0);
    setAtlasImage(null);
    setFrame(0);
    setPlaying(true);
  };

  const restart = () => {
    setFrame(0);
    setPlaying(true);
  };

  if (loadError) {
    return <p className={styles.trajectoryLoading}>Trajectory samples could not be loaded.</p>;
  }
  if (!manifests || !manifest || !scenario) {
    return <p className={styles.trajectoryLoading}>Loading checkpoint samples…</p>;
  }

  return (
    <div className={styles.trajectoryViewer}>
      <div className={styles.modelPicker} role="group" aria-label="World model checkpoint">
        {MODEL_VARIANTS.map((variant, index) => (
          <button
            key={variant.id}
            type="button"
            className={index === modelIndex ? styles.modelActive : undefined}
            onClick={() => chooseModel(index)}
            aria-pressed={index === modelIndex}
          >
            <strong>{variant.label}</strong>
            <span>{variant.detail}</span>
          </button>
        ))}
      </div>
      <div className={styles.scenarioPicker} role="group" aria-label="Held-out trajectory">
        {manifest.scenarios.map((option, index) => (
          <button
            key={option.id}
            type="button"
            className={index === scenarioIndex ? styles.scenarioActive : undefined}
            onClick={() => chooseScenario(index)}
            aria-pressed={index === scenarioIndex}
          >
            <span>0{index + 1}</span>
            {option.title}
          </button>
        ))}
      </div>

      <div className={styles.trajectoryHeader}>
        <div>
          <strong>{scenario.title}</strong>
          <span>{scenario.description}</span>
        </div>
        <div className={`${styles.phaseBadge} ${isObserved ? styles.phaseObserved : isAutoregressive ? styles.phaseRolled : styles.phaseImagined}`}>
          <span /> {isObserved ? "OBSERVED" : manifest.generationLabel ?? (isAutoregressive ? "RE-FED · PASS 2" : "DIRECT · PASS 1")}
        </div>
      </div>

      <div className={styles.trajectoryGrid}>
        {scenario.lanes.map((lane, row) => (
          <article className={styles.trajectoryLane} key={lane.id}>
            <div className={styles.laneHeader}>
              <div>
                <span className={lane.kind === "truth" ? styles.truthDot : styles.sampleDot} />
                <strong>{isObserved ? "Known past" : lane.label}</strong>
              </div>
              <small>
                {isObserved || lane.kind === "truth"
                  ? isObserved ? "identical input" : "reference"
                  : usesLateMetrics
                    ? `P ${(lane.rolledPlayerErrorPx ?? lane.playerErrorPx).toFixed(1)} · K ${(lane.rolledPuckErrorPx ?? lane.puckErrorPx).toFixed(1)} px`
                    : `P ${(lane.directPlayerErrorPx ?? lane.playerErrorPx).toFixed(1)} · K ${(lane.directPuckErrorPx ?? lane.puckErrorPx).toFixed(1)} px`}
              </small>
            </div>
            <AtlasFrame
              image={atlasImage}
              frame={frame}
              row={row}
              frameSize={manifest.frameSize}
              label={lane.label}
            />
          </article>
        ))}
      </div>

      <div className={styles.trajectoryTransport}>
        <button type="button" onClick={() => setPlaying((value) => !value)}>
          {playing ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
          {playing ? "Pause" : "Play"}
        </button>
        <button type="button" onClick={restart}>
          <RotateCcw aria-hidden="true" /> Restart
        </button>
        <label>
          <span>FRAME {String(frame + 1).padStart(2, "0")} / {totalFrames}</span>
          <input
            type="range"
            min={0}
            max={totalFrames - 1}
            value={frame}
            onChange={(event) => {
              setFrame(Number(event.target.value));
              setPlaying(false);
            }}
            aria-label="Trajectory frame"
          />
        </label>
        <div className={styles.frameReadout}>
          <strong>{phaseDetail.toUpperCase()}</strong>
          <span>{manifest.samplerLabel ?? `${manifest.ddimSteps}-step DDIM`} · checkpoint {manifest.checkpointStep.toLocaleString()}</span>
        </div>
      </div>

      <div
        className={styles.branchTimeline}
        style={{ gridTemplateColumns: `repeat(${totalFrames}, 1fr)` }}
        aria-label={manifest.latentStepFrames
          ? `${manifest.contextFrames} observed frames, then ${manifest.futureFrames / manifest.latentStepFrames} autoregressive latent steps producing ${manifest.futureFrames} frames`
          : `${manifest.contextFrames} observed frames, ${modelFutureFrames} direct predictions, then ${Math.max(0, manifest.futureFrames - modelFutureFrames)} autoregressive predictions`}
      >
        {Array.from({ length: totalFrames }, (_, index) => (
          <button
            key={index}
            type="button"
            className={`${index >= manifest.contextFrames ? styles.branchFuture : ""} ${rolloutBoundaries.some((boundary) => index >= boundary) ? styles.branchRolled : ""} ${index === frame ? styles.branchCurrent : ""}`}
            onClick={() => {
              setFrame(index);
              setPlaying(false);
            }}
            aria-label={`Go to frame ${index + 1}`}
          />
        ))}
        <span style={{ left: `${(manifest.contextFrames / totalFrames) * 100}%` }}>PREDICTION BOUNDARY</span>
        {rolloutBoundaries.map((boundary, index) => (
          <span key={boundary} style={{ left: `${(boundary / totalFrames) * 100}%` }}>
            RE-FEED · PASS {index + 2}
          </span>
        ))}
      </div>
    </div>
  );
}
