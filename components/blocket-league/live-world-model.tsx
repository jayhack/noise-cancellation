"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Cpu, Gauge, Pause, Play, RotateCcw } from "lucide-react";

import { ACTION_NAMES, keyboardAction } from "@/lib/blocket-league/sim";

import styles from "./blocket-league-lab.module.css";

const PAD_ACTIONS = [8, 1, 2, 7, 0, 3, 6, 5, 4];
const PAD_LABELS = ["↖", "↑", "↗", "←", "·", "→", "↙", "↓", "↘"];
const MOVEMENT_KEYS = new Set(["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft", "w", "a", "s", "d"]);
const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

type OrtRuntime = typeof import("onnxruntime-web");
type OrtSession = import("onnxruntime-web").InferenceSession;

type LiveManifest = {
  modelKind?: "diffusion" | "direct-autoregressive";
  checkpointStep: number;
  sourceFps: number;
  frameSize: number;
  latentChannels: number;
  latentGrid: number;
  historyLatents: number;
  sequenceLatents: number;
  temporalDownsample: number;
  historyNoiseLevel: number;
  defaultIntegrationSteps: number;
  modelParameters: number;
  decoderParameters: number;
  modelBytes: number;
  decoderBytes: number;
  palette: number[][];
  assets: {
    dynamics: string;
    decoder: string;
    starterContext: string;
    starterActions?: string;
    starterFrame: string;
  };
};

type EngineState = {
  runtime: OrtRuntime;
  dynamics: OrtSession;
  decoder: OrtSession;
  provider: "webgpu" | "wasm";
  manifest: LiveManifest;
  starterContext: Float32Array;
  starterActions: BigInt64Array;
  cleanHistory: Float32Array;
  noisyHistory: Float32Array;
  historyActions: BigInt64Array;
  historyTimes: Float32Array;
  gaussian: () => number;
};

type DreamFrame = {
  image: ImageData;
  action: number;
};

type PlayerStatus = "idle" | "loading" | "ready" | "running" | "paused" | "error";

function assetUrl(path: string) {
  return `${BASE_PATH}${path}`;
}

function normalizeMovementKey(key: string) {
  return key.length === 1 ? key.toLowerCase() : key;
}

function createGaussian(seed: number) {
  let state = seed >>> 0;
  let spare: number | null = null;
  const uniform = () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    const radius = Math.sqrt(-2 * Math.log(Math.max(uniform(), 1e-7)));
    const angle = Math.PI * 2 * uniform();
    spare = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  };
}

function integrationTimes(steps: number) {
  const linearSteps = Math.floor(steps / 2);
  const quadraticSteps = steps - linearSteps;
  const values: number[] = [];
  for (let index = 0; index < linearSteps; index += 1) {
    values.push((0.1 * index) / linearSteps);
  }
  const start = Math.sqrt(0.1);
  for (let index = 0; index <= quadraticSteps; index += 1) {
    const value = start + ((1 - start) * index) / quadraticSteps;
    values.push(value * value);
  }
  return values;
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

function logitsToFrames(logits: Float32Array, manifest: LiveManifest) {
  const pixels = manifest.frameSize * manifest.frameSize;
  const classes = manifest.palette.length;
  const frames: ImageData[] = [];
  for (let frame = 0; frame < manifest.temporalDownsample; frame += 1) {
    const rgba = new Uint8ClampedArray(pixels * 4);
    const frameOffset = frame * classes * pixels;
    for (let pixel = 0; pixel < pixels; pixel += 1) {
      let bestClass = 0;
      let bestValue = -Infinity;
      for (let classIndex = 0; classIndex < classes; classIndex += 1) {
        const value = logits[frameOffset + classIndex * pixels + pixel];
        if (value > bestValue) {
          bestValue = value;
          bestClass = classIndex;
        }
      }
      const color = manifest.palette[bestClass];
      const output = pixel * 4;
      rgba[output] = color[0];
      rgba[output + 1] = color[1];
      rgba[output + 2] = color[2];
      rgba[output + 3] = 255;
    }
    frames.push(new ImageData(rgba, manifest.frameSize, manifest.frameSize));
  }
  return frames;
}

function resetEngineState(engine: EngineState, dreamSeed: number) {
  engine.cleanHistory = engine.starterContext.slice();
  engine.noisyHistory = engine.starterContext.slice();
  engine.historyActions = new BigInt64Array(engine.manifest.historyLatents * 2);
  if (engine.starterActions.length) {
    engine.historyActions.set(engine.starterActions);
  }
  engine.historyTimes = new Float32Array(engine.manifest.historyLatents).fill(1);
  engine.gaussian = createGaussian(dreamSeed);
}

async function generatePair(engine: EngineState, action: number, integrationSteps: number) {
  const { manifest, runtime } = engine;
  const latentSize = manifest.latentChannels * manifest.latentGrid * manifest.latentGrid;
  const historySize = manifest.historyLatents * latentSize;
  const sequenceSize = manifest.sequenceLatents * latentSize;
  if (manifest.modelKind === "direct-autoregressive") {
    const actionPairs = engine.historyActions.slice();
    actionPairs[actionPairs.length - 2] = BigInt(action);
    actionPairs[actionPairs.length - 1] = BigInt(action);
    const result = await engine.dynamics.run({
      latent_history: new runtime.Tensor(
        "float32",
        engine.cleanHistory,
        [1, manifest.historyLatents, manifest.latentChannels, manifest.latentGrid, manifest.latentGrid],
      ),
      actions: new runtime.Tensor(
        "int64",
        actionPairs,
        [1, manifest.historyLatents, manifest.temporalDownsample],
      ),
    });
    const latentTensor = result.next_latent as import("onnxruntime-web").Tensor;
    const sample = (latentTensor.data as Float32Array).slice();
    latentTensor.dispose();
    const decodeSequence = new Float32Array(sequenceSize);
    decodeSequence.set(engine.cleanHistory);
    decodeSequence.set(sample, historySize);
    const decoded = await engine.decoder.run({
      normalized_sequence: new runtime.Tensor(
        "float32",
        decodeSequence,
        [1, manifest.sequenceLatents, manifest.latentChannels, manifest.latentGrid, manifest.latentGrid],
      ),
    });
    const logitsTensor = decoded.logits as import("onnxruntime-web").Tensor;
    const frames = logitsToFrames(logitsTensor.data as Float32Array, manifest);
    logitsTensor.dispose();
    engine.cleanHistory.copyWithin(0, latentSize);
    engine.cleanHistory.set(sample, engine.cleanHistory.length - latentSize);
    engine.noisyHistory = engine.cleanHistory.slice();
    actionPairs.copyWithin(0, 2);
    actionPairs[actionPairs.length - 2] = BigInt(0);
    actionPairs[actionPairs.length - 1] = BigInt(0);
    engine.historyActions = actionPairs;
    return frames;
  }
  const noisySequence = new Float32Array(sequenceSize);
  const cleanSequence = new Float32Array(sequenceSize);
  const sequenceActions = new BigInt64Array(manifest.sequenceLatents * 2);
  const sequenceTimes = new Float32Array(manifest.sequenceLatents);
  noisySequence.set(engine.noisyHistory);
  cleanSequence.set(engine.cleanHistory);
  sequenceActions.set(engine.historyActions);
  sequenceActions[sequenceActions.length - 2] = BigInt(action);
  sequenceActions[sequenceActions.length - 1] = BigInt(action);
  sequenceTimes.set(engine.historyTimes);

  const sample = new Float32Array(latentSize);
  for (let index = 0; index < latentSize; index += 1) sample[index] = engine.gaussian();
  const times = integrationTimes(integrationSteps);
  for (let step = 0; step < integrationSteps; step += 1) {
    noisySequence.set(sample, historySize);
    cleanSequence.set(sample, historySize);
    sequenceTimes[sequenceTimes.length - 1] = times[step];
    const result = await engine.dynamics.run({
      noisy_sequence: new runtime.Tensor(
        "float32",
        noisySequence,
        [1, manifest.sequenceLatents, manifest.latentChannels, manifest.latentGrid, manifest.latentGrid],
      ),
      actions: new runtime.Tensor(
        "int64",
        sequenceActions,
        [1, manifest.sequenceLatents, manifest.temporalDownsample],
      ),
      times: new runtime.Tensor("float32", sequenceTimes, [1, manifest.sequenceLatents]),
      clean_sequence: new runtime.Tensor(
        "float32",
        cleanSequence,
        [1, manifest.sequenceLatents, manifest.latentChannels, manifest.latentGrid, manifest.latentGrid],
      ),
    });
    const velocityTensor = result.velocity as import("onnxruntime-web").Tensor;
    const velocity = velocityTensor.data as Float32Array;
    const delta = times[step + 1] - times[step];
    for (let index = 0; index < latentSize; index += 1) {
      sample[index] = Math.max(-8, Math.min(8, sample[index] + delta * velocity[index]));
    }
    velocityTensor.dispose();
  }

  cleanSequence.set(sample, historySize);
  const decoded = await engine.decoder.run({
    normalized_sequence: new runtime.Tensor(
      "float32",
      cleanSequence,
      [1, manifest.sequenceLatents, manifest.latentChannels, manifest.latentGrid, manifest.latentGrid],
    ),
  });
  const logitsTensor = decoded.logits as import("onnxruntime-web").Tensor;
  const frames = logitsToFrames(logitsTensor.data as Float32Array, manifest);
  logitsTensor.dispose();

  engine.cleanHistory.copyWithin(0, latentSize);
  engine.cleanHistory.set(sample, engine.cleanHistory.length - latentSize);
  const noiseLevel = manifest.historyNoiseLevel;
  engine.noisyHistory.copyWithin(0, latentSize);
  const noisyOffset = engine.noisyHistory.length - latentSize;
  for (let index = 0; index < latentSize; index += 1) {
    engine.noisyHistory[noisyOffset + index] =
      (1 - noiseLevel) * sample[index] + noiseLevel * engine.gaussian();
  }
  engine.historyActions.copyWithin(0, 2);
  engine.historyActions[engine.historyActions.length - 2] = BigInt(action);
  engine.historyActions[engine.historyActions.length - 1] = BigInt(action);
  engine.historyTimes.copyWithin(0, 1);
  engine.historyTimes[engine.historyTimes.length - 1] = 1 - noiseLevel;
  return frames;
}

export function LiveWorldModel() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<EngineState | null>(null);
  const keysRef = useRef(new Set<string>());
  const manualActionRef = useRef<number | null>(null);
  const pendingActionRef = useRef<number | null>(null);
  const startPlaybackRef = useRef<() => void>(() => {});
  const runningRef = useRef(false);
  const loopIdRef = useRef(0);
  const playbackTimerRef = useRef<number | null>(null);
  const queueRef = useRef<DreamFrame[]>([]);
  const resetCountRef = useRef(0);
  const stepsRef = useRef(1);
  const [status, setStatus] = useState<PlayerStatus>("idle");
  const [loadProgress, setLoadProgress] = useState(0);
  const [loadLabel, setLoadLabel] = useState("MODEL SLEEPING");
  const [error, setError] = useState("");
  const [provider, setProvider] = useState<"webgpu" | "wasm" | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelMegabytes, setModelMegabytes] = useState(22.8);
  const [modelParameters, setModelParameters] = useState(5_881_961);
  const [modelKind, setModelKind] = useState<LiveManifest["modelKind"]>("direct-autoregressive");
  const [steps, setSteps] = useState(1);
  const [inputAction, setInputAction] = useState(0);
  const [lastSteeredAction, setLastSteeredAction] = useState<number | null>(null);
  const [generatedFrames, setGeneratedFrames] = useState(0);
  const [pairMilliseconds, setPairMilliseconds] = useState<number | null>(null);

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

  useEffect(() => {
    drawStarter();
  }, [drawStarter]);

  useEffect(() => {
    const publishKeyboardAction = () => {
      const action = manualActionRef.current ?? keyboardAction(keysRef.current);
      setInputAction(action);
      return action;
    };
    const keyDown = (event: KeyboardEvent) => {
      const key = normalizeMovementKey(event.key);
      if (!MOVEMENT_KEYS.has(key)) return;
      event.preventDefault();
      keysRef.current.add(key);
      pendingActionRef.current = publishKeyboardAction();
      if (engineRef.current && !runningRef.current) startPlaybackRef.current();
    };
    const keyUp = (event: KeyboardEvent) => {
      const key = normalizeMovementKey(event.key);
      if (!MOVEMENT_KEYS.has(key)) return;
      event.preventDefault();
      keysRef.current.delete(key);
      publishKeyboardAction();
    };
    const clearKeys = () => {
      keysRef.current.clear();
      manualActionRef.current = null;
      pendingActionRef.current = null;
      setInputAction(0);
    };
    window.addEventListener("keydown", keyDown);
    window.addEventListener("keyup", keyUp);
    window.addEventListener("blur", clearKeys);
    return () => {
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
      window.removeEventListener("blur", clearKeys);
    };
  }, []);

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
    if (engine) {
      void engine.dynamics.release();
      void engine.decoder.release();
    }
  }, [stopPlayback]);

  const loadModel = async () => {
    if (status === "loading" || engineRef.current) return;
    setStatus("loading");
    setError("");
    setLoadProgress(0.01);
    setLoadLabel("READING CHECKPOINT MANIFEST");
    try {
      const manifestResponse = await fetch(assetUrl("/blocket-league/live/manifest.json"));
      if (!manifestResponse.ok) throw new Error("The browser model manifest is missing.");
      const manifest = await manifestResponse.json() as LiveManifest;
      setModelMegabytes((manifest.modelBytes + manifest.decoderBytes) / 1_048_576);
      setModelParameters(manifest.modelParameters + manifest.decoderParameters);
      setModelKind(manifest.modelKind ?? "diffusion");
      setSteps(manifest.defaultIntegrationSteps);
      stepsRef.current = manifest.defaultIntegrationSteps;
      const totalModelBytes = manifest.modelBytes + manifest.decoderBytes;
      const loaded = [0, 0];
      const reportProgress = () => {
        setLoadProgress(0.05 + 0.67 * ((loaded[0] + loaded[1]) / totalModelBytes));
      };
      setLoadLabel(`DOWNLOADING ${(totalModelBytes / 1_048_576).toFixed(1)} MB · LOCAL ONLY`);
      const [dynamicsBytes, decoderBytes, starterBytes, starterActionBytes] = await Promise.all([
        fetchBytes(assetUrl(manifest.assets.dynamics), (value) => {
          loaded[0] = value;
          reportProgress();
        }),
        fetchBytes(assetUrl(manifest.assets.decoder), (value) => {
          loaded[1] = value;
          reportProgress();
        }),
        fetchBytes(assetUrl(manifest.assets.starterContext)),
        manifest.assets.starterActions
          ? fetchBytes(assetUrl(manifest.assets.starterActions))
          : Promise.resolve(new Uint8Array()),
      ]);
      setLoadProgress(0.76);
      setLoadLabel("COMPILING GPU GRAPHS");
      const runtime = await import("onnxruntime-web/webgpu");
      runtime.env.logLevel = "warning";
      runtime.env.wasm.numThreads = 1;
      runtime.env.wasm.proxy = false;

      const createSessions = async (executionProvider: "webgpu" | "wasm") => {
        const dynamics = await runtime.InferenceSession.create(dynamicsBytes, {
          executionProviders: [executionProvider],
          graphOptimizationLevel: "all",
        });
        try {
          const decoder = await runtime.InferenceSession.create(decoderBytes, {
            executionProviders: [executionProvider],
            graphOptimizationLevel: "all",
          });
          return { dynamics, decoder };
        } catch (sessionError) {
          await dynamics.release();
          throw sessionError;
        }
      };

      let selectedProvider: "webgpu" | "wasm" = "wasm";
      let sessions: { dynamics: OrtSession; decoder: OrtSession };
      if ("gpu" in navigator) {
        try {
          sessions = await createSessions("webgpu");
          selectedProvider = "webgpu";
        } catch (gpuError) {
          console.warn("WebGPU initialization failed; falling back to WASM.", gpuError);
          setLoadLabel("GPU UNAVAILABLE · COMPILING CPU FALLBACK");
          sessions = await createSessions("wasm");
        }
      } else {
        setLoadLabel("NO WEBGPU · COMPILING CPU FALLBACK");
        sessions = await createSessions("wasm");
      }
      const starterContext = new Float32Array(
        starterBytes.buffer,
        starterBytes.byteOffset,
        starterBytes.byteLength / Float32Array.BYTES_PER_ELEMENT,
      ).slice();
      const expectedValues =
        manifest.historyLatents * manifest.latentChannels * manifest.latentGrid * manifest.latentGrid;
      if (starterContext.length !== expectedValues) {
        await sessions.dynamics.release();
        await sessions.decoder.release();
        throw new Error(`Starter context has ${starterContext.length} values; expected ${expectedValues}.`);
      }
      const engine: EngineState = {
        runtime: runtime as OrtRuntime,
        dynamics: sessions.dynamics,
        decoder: sessions.decoder,
        provider: selectedProvider,
        manifest,
        starterContext,
        starterActions: new BigInt64Array(),
        cleanHistory: starterContext.slice(),
        noisyHistory: starterContext.slice(),
        historyActions: new BigInt64Array(manifest.historyLatents * 2),
        historyTimes: new Float32Array(manifest.historyLatents).fill(1),
        gaussian: createGaussian(manifest.checkpointStep),
      };
      if (starterActionBytes.byteLength) {
        const starterActions = new BigInt64Array(starterActionBytes.slice().buffer);
        engine.historyActions.set(starterActions);
        engine.starterActions = starterActions.slice();
      }
      engineRef.current = engine;
      pendingActionRef.current = null;
      setInputAction(keyboardAction(keysRef.current));
      setModelLoaded(true);
      setProvider(selectedProvider);
      setLoadProgress(1);
      setLoadLabel(`${selectedProvider.toUpperCase()} READY · WEIGHTS FROZEN`);
      setStatus("ready");
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : String(loadError));
      setModelLoaded(false);
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
      const canvas = canvasRef.current;
      const context = canvas?.getContext("2d");
      if (canvas && context) {
        context.imageSmoothingEnabled = false;
        context.putImageData(next.image, 0, 0);
      }
      setGeneratedFrames((value) => value + 1);
    }, 1_000 / engine.manifest.sourceFps);

    const inferenceLoop = async () => {
      while (runningRef.current && loopIdRef.current === loopId) {
        if (queueRef.current.length >= engine.manifest.temporalDownsample) {
          await new Promise((resolve) => window.setTimeout(resolve, 8));
          continue;
        }
        const heldAction = manualActionRef.current ?? keyboardAction(keysRef.current);
        const action = heldAction === 0 ? (pendingActionRef.current ?? 0) : heldAction;
        pendingActionRef.current = null;
        if (action !== 0) setLastSteeredAction(action);
        const started = performance.now();
        try {
          const frames = await generatePair(engine, action, stepsRef.current);
          if (!runningRef.current || loopIdRef.current !== loopId) return;
          queueRef.current.push(...frames.map((image) => ({ image, action })));
          setPairMilliseconds(performance.now() - started);
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

  useEffect(() => {
    startPlaybackRef.current = startPlayback;
  });

  const pausePlayback = () => {
    stopPlayback();
    setStatus("paused");
  };

  const resetDream = () => {
    stopPlayback();
    queueRef.current = [];
    resetCountRef.current += 1;
    const engine = engineRef.current;
    if (engine) resetEngineState(engine, engine.manifest.checkpointStep + resetCountRef.current * 9_973);
    setGeneratedFrames(0);
    setPairMilliseconds(null);
    keysRef.current.clear();
    manualActionRef.current = null;
    pendingActionRef.current = null;
    setInputAction(0);
    setLastSteeredAction(null);
    setError("");
    drawStarter();
    setStatus(engine ? "ready" : "idle");
  };

  const beginManualAction = (action: number) => {
    manualActionRef.current = action;
    pendingActionRef.current = action;
    setInputAction(action);
    if (engineRef.current && !runningRef.current) startPlaybackRef.current();
  };

  const endManualAction = () => {
    manualActionRef.current = null;
    setInputAction(keyboardAction(keysRef.current));
  };

  const theoreticalFps = pairMilliseconds ? 2_000 / pairMilliseconds : null;

  return (
    <div className={styles.livePlayer}>
      <div className={styles.livePlayerHeader}>
        <div className={styles.liveLabel}>
          <span className={status === "running" ? styles.livePulse : undefined} />
          FROZEN CHECKPOINT · {provider?.toUpperCase() ?? "NOT LOADED"}
        </div>
        <div>{(modelParameters / 1_000_000).toFixed(2)}M PARAMETERS · {modelMegabytes.toFixed(1)} MB FP32</div>
      </div>

      <div className={styles.livePlayerGrid}>
        <div className={styles.liveDreamColumn}>
          <div className={styles.liveCanvasHeader}>
            <strong>MODEL&apos;S DREAM</strong>
            <span>{generatedFrames.toString().padStart(4, "0")} GENERATED FRAMES</span>
          </div>
          <div className={styles.liveCanvasWrap}>
            <canvas
              ref={canvasRef}
              className={styles.liveCanvas}
              width={64}
              height={64}
              role="img"
              aria-label="The live frames imagined by the frozen Blocket League latent world model."
            />
            {(status === "idle" || status === "loading" || status === "error") && (
              <div className={styles.liveCanvasOverlay}>
                {status === "idle" && (
                  <button type="button" onClick={loadModel}>
                    <Cpu aria-hidden="true" /> Load local model
                  </button>
                )}
                {status === "loading" && (
                  <>
                    <div className={styles.liveLoadTrack} aria-label={`${Math.round(loadProgress * 100)} percent loaded`}>
                      <span style={{ width: `${loadProgress * 100}%` }} />
                    </div>
                    <strong>{Math.round(loadProgress * 100)}%</strong>
                  </>
                )}
                {status === "error" && (
                  <button type="button" onClick={resetDream}>
                    <RotateCcw aria-hidden="true" /> Reset player
                  </button>
                )}
              </div>
            )}
          </div>
          <div className={styles.liveStatusLine}>
            <span>{loadLabel}</span>
            <span>{error || (modelKind === "direct-autoregressive"
              ? "NO SIMULATOR · ONE TRANSFORMER PASS → LATENT → PIXELS"
              : "NO SIMULATOR · NO SERVER · ACTIONS → LATENTS → PIXELS")}</span>
          </div>
        </div>

        <aside className={styles.liveControls} aria-label="Live world model controls">
          <div className={styles.liveMetrics}>
            <div><span>COMPUTE</span><strong>{pairMilliseconds ? `${pairMilliseconds.toFixed(0)} ms` : "—"}</strong></div>
            <div><span>HEADROOM</span><strong>{theoreticalFps ? `${theoreticalFps.toFixed(0)} fps` : "—"}</strong></div>
            <div><span>INPUT</span><strong>{ACTION_NAMES[inputAction]}</strong></div>
          </div>

          <div className={styles.liveQuality}>
            <div><Gauge aria-hidden="true" /><span>{modelKind === "direct-autoregressive" ? "INFERENCE" : "SAMPLER QUALITY"}</span></div>
            {modelKind === "direct-autoregressive" ? (
              <div><button type="button" className={styles.liveQualityActive}>1 pass</button></div>
            ) : (
              <div role="group" aria-label="Flow integration steps">
                {[4, 6, 10].map((value) => (
                <button
                  key={value}
                  type="button"
                  className={steps === value ? styles.liveQualityActive : undefined}
                  aria-pressed={steps === value}
                  onClick={() => {
                    setSteps(value);
                    stepsRef.current = value;
                  }}
                >
                  {value} step
                </button>
                ))}
              </div>
            )}
          </div>

          <div className={styles.pad} aria-label="Live model direction pad">
            {PAD_ACTIONS.map((action, index) => (
              <button
                key={action}
                type="button"
                className={[
                  action === 0 ? styles.padCenter : "",
                  inputAction === action ? styles.padActive : "",
                ].filter(Boolean).join(" ") || undefined}
                aria-label={ACTION_NAMES[action]}
                aria-pressed={inputAction === action}
                data-input-active={inputAction === action ? "true" : "false"}
                onPointerDown={() => beginManualAction(action)}
                onPointerUp={endManualAction}
                onPointerCancel={endManualAction}
                onPointerLeave={endManualAction}
              >
                {PAD_LABELS[index]}
              </button>
            ))}
          </div>

          <div className={styles.liveTransport}>
            {status === "running" ? (
              <button type="button" onClick={pausePlayback}>
                <Pause aria-hidden="true" /> Pause dream
              </button>
            ) : (
              <button
                type="button"
                onClick={startPlayback}
                disabled={!modelLoaded || status === "loading"}
              >
                <Play aria-hidden="true" /> Enter dream
              </button>
            )}
            <button type="button" onClick={resetDream} disabled={!modelLoaded}>
              <RotateCcw aria-hidden="true" /> {modelKind === "direct-autoregressive" ? "Reset rollout" : "New noise"}
            </button>
          </div>
          <p className={styles.liveHint}>
            Hold WASD or arrow keys to start or steer. Input: {ACTION_NAMES[inputAction]} · {lastSteeredAction === null
              ? "no steer token consumed yet."
              : `last steer consumed: ${ACTION_NAMES[lastSteeredAction]} × 2 frames.`}
          </p>
        </aside>
      </div>
    </div>
  );
}
