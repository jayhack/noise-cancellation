"use client";

import { useEffect, useRef, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";

import styles from "./blocket-league-lab.module.css";

type LongLane = {
  id: string;
  label: string;
  kind: "truth" | "sample";
  playerErrorPx: number;
  puckErrorPx: number;
};

type LongScenario = {
  atlas: string;
  actions: string[];
  events: string[];
  lanes: LongLane[];
};

type LongManifest = {
  frameSize: number;
  contextFrames: number;
  futureFrames: number;
  playbackFps: number;
  checkpointStep: number;
  samplerLabel?: string;
  latentStepFrames: number;
  scenarios: LongScenario[];
};

function LongFrame({
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
      className={styles.entropyCanvas}
      width={frameSize}
      height={frameSize}
      role="img"
      aria-label={`${label}, frame ${frame + 1}`}
    />
  );
}

export function LongRolloutViewer() {
  const [manifest, setManifest] = useState<LongManifest | null>(null);
  const [atlasImage, setAtlasImage] = useState<HTMLImageElement | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch("/blocket-league/entropy-rollout/manifest.json")
      .then((response) => {
        if (!response.ok) throw new Error("long rollout manifest unavailable");
        return response.json() as Promise<LongManifest>;
      })
      .then((value) => { if (!cancelled) setManifest(value); })
      .catch(() => { if (!cancelled) setLoadError(true); });
    return () => { cancelled = true; };
  }, []);

  const scenario = manifest?.scenarios[0] ?? null;
  useEffect(() => {
    if (!scenario) return;
    let cancelled = false;
    const image = new Image();
    image.onload = () => { if (!cancelled) setAtlasImage(image); };
    image.src = scenario.atlas;
    return () => { cancelled = true; };
  }, [scenario]);

  const totalFrames = manifest ? manifest.contextFrames + manifest.futureFrames : 1;
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

  if (loadError) return <p className={styles.trajectoryLoading}>Long rollout samples could not be loaded.</p>;
  if (!manifest || !scenario) return <p className={styles.trajectoryLoading}>Loading the frozen rollout…</p>;

  const sampleLanes = scenario.lanes.filter((lane) => lane.kind === "sample");
  const futureIndex = frame - manifest.contextFrames;
  const isObserved = futureIndex < 0;
  const phase = isObserved
    ? "KNOWN PAST"
    : futureIndex < 12
      ? "TRAINED HORIZON"
      : futureIndex < 24
        ? "EXTRAPOLATING"
        : "FREE-RUNNING";
  const action = futureIndex >= 0 ? scenario.actions[futureIndex] : "shared input";
  const event = futureIndex >= 0 ? scenario.events[futureIndex] : "identical state";

  return (
    <div className={styles.entropyViewer}>
      <div className={styles.entropyHeader}>
        <div>
          <span className={styles.statusDot} />
          <strong>{phase}</strong>
        </div>
        <span>
          {isObserved
            ? "one six-frame past"
            : `future ${String(futureIndex + 1).padStart(2, "0")} / ${manifest.futureFrames} · ${action} · ${event}`}
        </span>
      </div>

      <div className={styles.entropyGrid}>
        {sampleLanes.map((lane, index) => (
          <article className={styles.entropyLane} key={lane.id}>
            <div className={styles.laneHeader}>
              <div><span className={styles.sampleDot} /><strong>{isObserved ? "Known past" : lane.label}</strong></div>
              <small>
                {isObserved
                  ? "identical input"
                  : `64F · P ${lane.playerErrorPx.toFixed(1)} · K ${lane.puckErrorPx.toFixed(1)} px`}
              </small>
            </div>
            <LongFrame
              image={atlasImage}
              frame={frame}
              row={index + 1}
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
        <button type="button" onClick={() => { setFrame(0); setPlaying(true); }}>
          <RotateCcw aria-hidden="true" /> Restart
        </button>
        <label>
          <span>FRAME {String(frame + 1).padStart(2, "0")} / {totalFrames}</span>
          <input
            type="range"
            min={0}
            max={totalFrames - 1}
            value={frame}
            onChange={(event) => { setFrame(Number(event.target.value)); setPlaying(false); }}
            aria-label="Long rollout frame"
          />
        </label>
        <div className={styles.frameReadout}>
          <strong>
            {isObserved
              ? "CONTEXT"
              : `LATENT ${String(Math.ceil((futureIndex + 1) / manifest.latentStepFrames)).padStart(2, "0")}`}
          </strong>
          <span>{manifest.samplerLabel ?? "flow sampler"} · frozen checkpoint {manifest.checkpointStep.toLocaleString()}</span>
        </div>
      </div>

      <div className={styles.entropyTimeline} aria-label="Six observed frames, a twelve-frame trained horizon, and a sixty-four-frame autoregressive rollout">
        <span style={{ left: `${(manifest.contextFrames / totalFrames) * 100}%` }}>PREDICT</span>
        <span style={{ left: `${((manifest.contextFrames + 12) / totalFrames) * 100}%` }}>12F TRAINED</span>
        <span style={{ left: `${((manifest.contextFrames + 24) / totalFrames) * 100}%` }}>24F</span>
        <i style={{ width: `${(manifest.contextFrames / totalFrames) * 100}%` }} />
        <i style={{ width: `${(12 / totalFrames) * 100}%` }} />
        <i style={{ width: `${(12 / totalFrames) * 100}%` }} />
        <i className={styles.entropyTail} />
      </div>
    </div>
  );
}
