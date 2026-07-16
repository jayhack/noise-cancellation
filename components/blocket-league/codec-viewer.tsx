"use client";

import { useEffect, useRef, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";

import styles from "./blocket-league-lab.module.css";

type CodecScenario = {
  id: string;
  title: string;
  description: string;
  atlas: string;
};

type CodecManifest = {
  frameSize: number;
  totalFrames: number;
  playbackFps: number;
  temporalDownsample: number;
  latentGrid: number;
  latentChannels: number;
  compressionRatio: number;
  checkpointStep: number;
  scenarios: CodecScenario[];
};

const ROWS = [
  ["Pixels", "source frame"],
  ["RAE decode", "hard palette reconstruction"],
  ["Latent PCA", "8 × 8 representation"],
] as const;

function CodecFrame({
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
      className={styles.codecCanvas}
      width={frameSize}
      height={frameSize}
      role="img"
      aria-label={`${label}, frame ${frame + 1}`}
    />
  );
}

export function CodecViewer() {
  const [manifest, setManifest] = useState<CodecManifest | null>(null);
  const [scenarioIndex, setScenarioIndex] = useState(0);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [atlasImage, setAtlasImage] = useState<HTMLImageElement | null>(null);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch("/blocket-league/codec/manifest.json")
      .then((response) => {
        if (!response.ok) throw new Error("codec manifest unavailable");
        return response.json() as Promise<CodecManifest>;
      })
      .then((value) => { if (!cancelled) setManifest(value); })
      .catch(() => { if (!cancelled) setLoadError(true); });
    return () => { cancelled = true; };
  }, []);

  const scenario = manifest?.scenarios[scenarioIndex] ?? null;
  useEffect(() => {
    if (!scenario) return;
    let cancelled = false;
    const image = new Image();
    image.onload = () => { if (!cancelled) setAtlasImage(image); };
    image.src = scenario.atlas;
    return () => { cancelled = true; };
  }, [scenario]);

  useEffect(() => {
    if (!manifest || !playing) return;
    const timer = window.setTimeout(() => {
      setFrame((value) => (value + 1) % manifest.totalFrames);
    }, 1_000 / manifest.playbackFps);
    return () => window.clearTimeout(timer);
  }, [frame, manifest, playing]);

  if (loadError) return <p className={styles.trajectoryLoading}>Codec samples could not be loaded.</p>;
  if (!manifest || !scenario) return <p className={styles.trajectoryLoading}>Loading codec reconstructions…</p>;

  const latentIndex = Math.floor(frame / manifest.temporalDownsample);
  return (
    <div className={styles.codecViewer}>
      <div className={styles.codecStats} aria-label="Representation codec shape">
        <div><span>RGB INPUT</span><strong>2 × 64 × 64 × 3</strong></div>
        <div><span>RAE LATENT</span><strong>1 × {manifest.latentGrid} × {manifest.latentGrid} × {manifest.latentChannels}</strong></div>
        <div><span>COMPRESSION</span><strong>{manifest.compressionRatio.toFixed(0)}×</strong></div>
      </div>
      <div className={styles.scenarioPicker} role="group" aria-label="Codec clip">
        {manifest.scenarios.map((option, index) => (
          <button
            key={option.id}
            type="button"
            className={index === scenarioIndex ? styles.scenarioActive : undefined}
            onClick={() => {
              setScenarioIndex(index);
              setFrame(0);
              setPlaying(true);
              setAtlasImage(null);
            }}
            aria-pressed={index === scenarioIndex}
          >
            <span>0{index + 1}</span>{option.title}
          </button>
        ))}
      </div>
      <div className={styles.codecGrid}>
        {ROWS.map(([label, detail], row) => (
          <article className={styles.codecLane} key={label}>
            <div className={styles.laneHeader}>
              <div><span className={row === 1 ? styles.sampleDot : styles.truthDot} /><strong>{label}</strong></div>
              <small>{detail}</small>
            </div>
            <CodecFrame
              image={atlasImage}
              frame={frame}
              row={row}
              frameSize={manifest.frameSize}
              label={label}
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
          <span>RGB FRAME {String(frame + 1).padStart(2, "0")} / {manifest.totalFrames}</span>
          <input
            type="range"
            min={0}
            max={manifest.totalFrames - 1}
            value={frame}
            onChange={(event) => { setFrame(Number(event.target.value)); setPlaying(false); }}
            aria-label="Codec frame"
          />
        </label>
        <div className={styles.frameReadout}>
          <strong>LATENT {String(latentIndex + 1).padStart(2, "0")}</strong>
          <span>covers RGB {latentIndex * 2 + 1}–{latentIndex * 2 + 2} · codec {manifest.checkpointStep.toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
}
