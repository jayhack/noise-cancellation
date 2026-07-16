"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Cpu, Gauge, Pause, Play, RotateCcw } from "lucide-react";

import { ACTION_NAMES, ACTION_VECTORS, keyboardAction } from "@/lib/blocket-league/sim";

import styles from "./blocket-league-lab.module.css";

const PAD_ACTIONS = [8, 1, 2, 7, 0, 3, 6, 5, 4];
const PAD_LABELS = ["↖", "↑", "↗", "←", "·", "→", "↙", "↓", "↘"];
const MOVEMENT_KEYS = new Set(["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft", "w", "a", "s", "d"]);
const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

type OrtRuntime = typeof import("onnxruntime-web");
type OrtSession = import("onnxruntime-web").InferenceSession;

type LiveManifest = {
  modelKind: "passive-direct-pixel-autoregressive";
  checkpointStep: number;
  sourceFps: number;
  frameSize: number;
  patchSize: number;
  gridSize: number;
  historyFrames: number;
  hiddenSize: number;
  interventionBlock: number;
  interventionStrength: number;
  modelParameters: number;
  modelBytes: number;
  palette: number[][];
  directions: { x: number[]; y: number[] };
  assets: { dynamics: string; starterContext: string; starterFrame: string };
};

type EngineState = {
  runtime: OrtRuntime;
  dynamics: OrtSession;
  provider: "webgpu" | "wasm";
  manifest: LiveManifest;
  starterContext: Float32Array;
  history: Float32Array;
};

type DreamFrame = { image: ImageData; action: number };
type PlayerStatus = "idle" | "loading" | "ready" | "running" | "paused" | "error";

function assetUrl(path: string) {
  return `${BASE_PATH}${path}`;
}

function normalizeMovementKey(key: string) {
  return key.length === 1 ? key.toLowerCase() : key;
}

async function fetchBytes(url: string, onProgress?: (loaded: number) => void) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
  if (!response.body) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded);
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function classesToImage(classes: Float32Array, manifest: LiveManifest) {
  const pixels = manifest.frameSize * manifest.frameSize;
  const rgba = new Uint8ClampedArray(pixels * 4);
  for (let pixel = 0; pixel < pixels; pixel += 1) {
    const color = manifest.palette[Number(classes[pixel])] ?? manifest.palette[0];
    const output = pixel * 4;
    rgba[output] = color[0];
    rgba[output + 1] = color[1];
    rgba[output + 2] = color[2];
    rgba[output + 3] = 255;
  }
  return new ImageData(rgba, manifest.frameSize, manifest.frameSize);
}

function logitsToClasses(logits: Float32Array, manifest: LiveManifest) {
  const pixels = manifest.frameSize * manifest.frameSize;
  const output = new Float32Array(pixels);
  for (let pixel = 0; pixel < pixels; pixel += 1) {
    let bestClass = 0;
    let bestValue = -Infinity;
    for (let classIndex = 0; classIndex < manifest.palette.length; classIndex += 1) {
      const value = logits[classIndex * pixels + pixel];
      if (value > bestValue) {
        bestValue = value;
        bestClass = classIndex;
      }
    }
    output[pixel] = bestClass;
  }
  return output;
}

function greenTokenMask(history: Float32Array, manifest: LiveManifest) {
  const pixels = manifest.frameSize * manifest.frameSize;
  const offset = history.length - pixels;
  let mass = 0;
  let sumX = 0;
  let sumY = 0;
  for (let pixel = 0; pixel < pixels; pixel += 1) {
    const value = Number(history[offset + pixel]);
    if (value !== 5 && value !== 6) continue;
    mass += 1;
    sumX += (pixel % manifest.frameSize) + 0.5;
    sumY += Math.floor(pixel / manifest.frameSize) + 0.5;
  }
  const mask = new Float32Array(manifest.historyFrames * manifest.gridSize * manifest.gridSize);
  if (!mass) return mask;
  const patchX = Math.max(0, Math.min(manifest.gridSize - 1, Math.floor(sumX / mass / manifest.patchSize)));
  const patchY = Math.max(0, Math.min(manifest.gridSize - 1, Math.floor(sumY / mass / manifest.patchSize)));
  const timeOffset = (manifest.historyFrames - 1) * manifest.gridSize * manifest.gridSize;
  mask[timeOffset + patchY * manifest.gridSize + patchX] = 1;
  return mask;
}

function steeringVector(action: number, manifest: LiveManifest) {
  const vector = ACTION_VECTORS[action] ?? ACTION_VECTORS[0];
  const length = Math.hypot(vector.x, vector.y) || 1;
  const x = vector.x / length;
  const y = vector.y / length;
  const output = new Float32Array(manifest.hiddenSize);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = manifest.interventionStrength * (
      x * manifest.directions.x[index] + y * manifest.directions.y[index]
    );
  }
  return output;
}

async function generateFrame(engine: EngineState, action: number) {
  const { manifest, runtime } = engine;
  const result = await engine.dynamics.run({
    pixel_history: new runtime.Tensor(
      "float32",
      engine.history,
      [1, manifest.historyFrames, manifest.frameSize, manifest.frameSize],
    ),
    intervention: new runtime.Tensor(
      "float32",
      steeringVector(action, manifest),
      [1, manifest.hiddenSize],
    ),
    intervention_mask: new runtime.Tensor(
      "float32",
      greenTokenMask(engine.history, manifest),
      [1, manifest.historyFrames, manifest.gridSize * manifest.gridSize],
    ),
  });
  const logitsTensor = result.next_logits as import("onnxruntime-web").Tensor;
  const next = logitsToClasses(logitsTensor.data as Float32Array, manifest);
  logitsTensor.dispose();
  const pixels = manifest.frameSize * manifest.frameSize;
  engine.history.copyWithin(0, pixels);
  engine.history.set(next, engine.history.length - pixels);
  return classesToImage(next, manifest);
}

export function LiveWorldModel() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<EngineState | null>(null);
  const keysRef = useRef(new Set<string>());
  const manualActionRef = useRef<number | null>(null);
  const startPlaybackRef = useRef<() => void>(() => {});
  const runningRef = useRef(false);
  const loopIdRef = useRef(0);
  const playbackTimerRef = useRef<number | null>(null);
  const queueRef = useRef<DreamFrame[]>([]);
  const [status, setStatus] = useState<PlayerStatus>("idle");
  const [loadProgress, setLoadProgress] = useState(0);
  const [loadLabel, setLoadLabel] = useState("MODEL SLEEPING");
  const [error, setError] = useState("");
  const [provider, setProvider] = useState<"webgpu" | "wasm" | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelMegabytes, setModelMegabytes] = useState(14.2);
  const [modelParameters, setModelParameters] = useState(3_667_992);
  const [inputAction, setInputAction] = useState(0);
  const [generatedFrames, setGeneratedFrames] = useState(0);
  const [frameMilliseconds, setFrameMilliseconds] = useState<number | null>(null);

  const drawStarter = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const image = new Image();
    image.onload = () => {
      const context = canvas.getContext("2d");
      if (!context) return;
      context.imageSmoothingEnabled = false;
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
    };
    image.src = assetUrl("/blocket-league/live/starter-frame.png");
  }, []);

  useEffect(() => { drawStarter(); }, [drawStarter]);

  const stopPlayback = useCallback(() => {
    runningRef.current = false;
    loopIdRef.current += 1;
    if (playbackTimerRef.current !== null) {
      window.clearInterval(playbackTimerRef.current);
      playbackTimerRef.current = null;
    }
  }, []);

  useEffect(() => () => {
    stopPlayback();
    const engine = engineRef.current;
    if (engine) void engine.dynamics.release();
  }, [stopPlayback]);

  const loadModel = async () => {
    if (status === "loading" || engineRef.current) return;
    setStatus("loading");
    setError("");
    try {
      setLoadProgress(0.02);
      setLoadLabel("READING PIXEL TRANSFORMER MANIFEST");
      const response = await fetch(assetUrl("/blocket-league/live/manifest.json"));
      if (!response.ok) throw new Error("The browser model manifest is missing.");
      const manifest = await response.json() as LiveManifest;
      if (manifest.modelKind !== "passive-direct-pixel-autoregressive") {
        throw new Error("The loaded checkpoint is not the passive pixel model.");
      }
      setModelMegabytes(manifest.modelBytes / 1_048_576);
      setModelParameters(manifest.modelParameters);
      let loaded = 0;
      setLoadLabel(`DOWNLOADING ${(manifest.modelBytes / 1_048_576).toFixed(1)} MB · LOCAL ONLY`);
      const [modelBytes, starterBytes] = await Promise.all([
        fetchBytes(assetUrl(manifest.assets.dynamics), (value) => {
          loaded = value;
          setLoadProgress(0.05 + 0.7 * loaded / manifest.modelBytes);
        }),
        fetchBytes(assetUrl(manifest.assets.starterContext)),
      ]);
      setLoadProgress(0.78);
      setLoadLabel("COMPILING FROZEN TRANSFORMER");
      const runtime = await import("onnxruntime-web/webgpu");
      runtime.env.logLevel = "warning";
      runtime.env.wasm.numThreads = 1;
      runtime.env.wasm.proxy = false;
      let selectedProvider: "webgpu" | "wasm" = "wasm";
      let dynamics: OrtSession;
      if ("gpu" in navigator) {
        try {
          dynamics = await runtime.InferenceSession.create(modelBytes, {
            executionProviders: ["webgpu"], graphOptimizationLevel: "all",
          });
          selectedProvider = "webgpu";
        } catch (gpuError) {
          console.warn("WebGPU initialization failed; falling back to WASM.", gpuError);
          dynamics = await runtime.InferenceSession.create(modelBytes, {
            executionProviders: ["wasm"], graphOptimizationLevel: "all",
          });
        }
      } else {
        dynamics = await runtime.InferenceSession.create(modelBytes, {
          executionProviders: ["wasm"], graphOptimizationLevel: "all",
        });
      }
      const starterContext = new Float32Array(starterBytes.slice().buffer);
      const expected = manifest.historyFrames * manifest.frameSize * manifest.frameSize;
      if (starterContext.length !== expected) {
        await dynamics.release();
        throw new Error(`Starter context has ${starterContext.length} pixels; expected ${expected}.`);
      }
      engineRef.current = {
        runtime: runtime as OrtRuntime,
        dynamics,
        provider: selectedProvider,
        manifest,
        starterContext: starterContext.slice(),
        history: starterContext.slice(),
      };
      setProvider(selectedProvider);
      setModelLoaded(true);
      setLoadProgress(1);
      setLoadLabel(`${selectedProvider.toUpperCase()} READY · NO ACTION CHANNEL`);
      setStatus("ready");
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : String(loadError));
      setLoadLabel("MODEL FAILED TO WAKE");
      setStatus("error");
    }
  };

  const startPlayback = () => {
    const engine = engineRef.current;
    if (!engine || runningRef.current) return;
    runningRef.current = true;
    const loopId = loopIdRef.current + 1;
    loopIdRef.current = loopId;
    setStatus("running");
    playbackTimerRef.current = window.setInterval(() => {
      const next = queueRef.current.shift();
      if (!next) return;
      const context = canvasRef.current?.getContext("2d");
      if (context) {
        context.imageSmoothingEnabled = false;
        context.putImageData(next.image, 0, 0);
      }
      setGeneratedFrames((value) => value + 1);
    }, 1_000 / engine.manifest.sourceFps);
    const inferenceLoop = async () => {
      while (runningRef.current && loopIdRef.current === loopId) {
        if (queueRef.current.length >= 2) {
          await new Promise((resolve) => window.setTimeout(resolve, 8));
          continue;
        }
        const action = manualActionRef.current ?? keyboardAction(keysRef.current);
        const started = performance.now();
        try {
          const image = await generateFrame(engine, action);
          if (!runningRef.current || loopIdRef.current !== loopId) return;
          queueRef.current.push({ image, action });
          setFrameMilliseconds(performance.now() - started);
        } catch (inferenceError) {
          stopPlayback();
          setError(inferenceError instanceof Error ? inferenceError.message : String(inferenceError));
          setStatus("error");
          return;
        }
      }
    };
    void inferenceLoop();
  };

  useEffect(() => { startPlaybackRef.current = startPlayback; });

  useEffect(() => {
    const publish = () => setInputAction(manualActionRef.current ?? keyboardAction(keysRef.current));
    const keyDown = (event: KeyboardEvent) => {
      const key = normalizeMovementKey(event.key);
      if (!MOVEMENT_KEYS.has(key)) return;
      event.preventDefault();
      keysRef.current.add(key);
      publish();
      if (engineRef.current && !runningRef.current) startPlaybackRef.current();
    };
    const keyUp = (event: KeyboardEvent) => {
      const key = normalizeMovementKey(event.key);
      if (!MOVEMENT_KEYS.has(key)) return;
      event.preventDefault();
      keysRef.current.delete(key);
      publish();
    };
    const clear = () => { keysRef.current.clear(); manualActionRef.current = null; setInputAction(0); };
    window.addEventListener("keydown", keyDown);
    window.addEventListener("keyup", keyUp);
    window.addEventListener("blur", clear);
    return () => {
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
      window.removeEventListener("blur", clear);
    };
  }, []);

  const resetDream = () => {
    stopPlayback();
    queueRef.current = [];
    const engine = engineRef.current;
    if (engine) engine.history = engine.starterContext.slice();
    keysRef.current.clear();
    manualActionRef.current = null;
    setInputAction(0);
    setGeneratedFrames(0);
    setFrameMilliseconds(null);
    setError("");
    drawStarter();
    setStatus(engine ? "ready" : "idle");
  };

  const beginManualAction = (action: number) => {
    manualActionRef.current = action;
    setInputAction(action);
    if (engineRef.current && !runningRef.current) startPlaybackRef.current();
  };
  const endManualAction = () => {
    manualActionRef.current = null;
    setInputAction(keyboardAction(keysRef.current));
  };
  const theoreticalFps = frameMilliseconds ? 1_000 / frameMilliseconds : null;

  return (
    <div className={styles.livePlayer}>
      <div className={styles.livePlayerHeader}>
        <div className={styles.liveLabel}><span className={status === "running" ? styles.livePulse : undefined} />FROZEN PIXEL TRANSFORMER · {provider?.toUpperCase() ?? "NOT LOADED"}</div>
        <div>{(modelParameters / 1_000_000).toFixed(2)}M PARAMETERS · {modelMegabytes.toFixed(1)} MB FP32</div>
      </div>
      <div className={styles.livePlayerGrid}>
        <div className={styles.liveDreamColumn}>
          <div className={styles.liveCanvasHeader}><strong>MODEL&apos;S DREAM</strong><span>{generatedFrames.toString().padStart(4, "0")} GENERATED FRAMES</span></div>
          <div className={styles.liveCanvasWrap}>
            <canvas ref={canvasRef} className={styles.liveCanvas} width={64} height={64} role="img" aria-label="Live frames imagined by the passive Blocket League pixel transformer." />
            {(status === "idle" || status === "loading" || status === "error") && (
              <div className={styles.liveCanvasOverlay}>
                {status === "idle" && <button type="button" onClick={loadModel}><Cpu aria-hidden="true" /> Load local model</button>}
                {status === "loading" && <><div className={styles.liveLoadTrack} aria-label={`${Math.round(loadProgress * 100)} percent loaded`}><span style={{ width: `${loadProgress * 100}%` }} /></div><strong>{Math.round(loadProgress * 100)}%</strong></>}
                {status === "error" && <button type="button" onClick={resetDream}><RotateCcw aria-hidden="true" /> Reset player</button>}
              </div>
            )}
          </div>
          <div className={styles.liveStatusLine}><span>{loadLabel}</span><span>{error || "PIXELS → TRANSFORMER → PIXELS · CONTROL = HIDDEN-STATE WRITE"}</span></div>
        </div>
        <aside className={styles.liveControls} aria-label="Activation steering controls">
          <div className={styles.liveMetrics}>
            <div><span>COMPUTE</span><strong>{frameMilliseconds ? `${frameMilliseconds.toFixed(0)} ms` : "—"}</strong></div>
            <div><span>HEADROOM</span><strong>{theoreticalFps ? `${theoreticalFps.toFixed(0)} fps` : "—"}</strong></div>
            <div><span>WRITE</span><strong>{ACTION_NAMES[inputAction]}</strong></div>
          </div>
          <div className={styles.liveQuality}><div><Gauge aria-hidden="true" /><span>INTERVENTION</span></div><div><button type="button" className={styles.liveQualityActive}>block 6 · 8σ</button></div></div>
          <div className={styles.pad} aria-label="Hidden-state direction pad">
            {PAD_ACTIONS.map((action, index) => (
              <button key={action} type="button" className={[action === 0 ? styles.padCenter : "", inputAction === action ? styles.padActive : ""].filter(Boolean).join(" ") || undefined} aria-label={`${ACTION_NAMES[action]} activation write`} aria-pressed={inputAction === action} data-input-active={inputAction === action ? "true" : "false"} onPointerDown={() => beginManualAction(action)} onPointerUp={endManualAction} onPointerCancel={endManualAction} onPointerLeave={endManualAction}>{PAD_LABELS[index]}</button>
            ))}
          </div>
          <div className={styles.liveTransport}>
            {status === "running" ? <button type="button" onClick={() => { stopPlayback(); setStatus("paused"); }}><Pause aria-hidden="true" /> Pause dream</button> : <button type="button" onClick={startPlayback} disabled={!modelLoaded || status === "loading"}><Play aria-hidden="true" /> Enter dream</button>}
            <button type="button" onClick={resetDream} disabled={!modelLoaded}><RotateCcw aria-hidden="true" /> Reset rollout</button>
          </div>
          <p className={styles.liveHint}>Hold WASD or arrow keys to write the recovered velocity directions into the green circle&apos;s block-6 activation. The white puck has no controls.</p>
        </aside>
      </div>
    </div>
  );
}
