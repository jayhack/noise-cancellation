"use client";

import { useEffect, useState } from "react";

import styles from "./blocket-league-lab.module.css";
import { CausalInterventionViewer } from "./causal-intervention-viewer";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

type TaskId = "position" | "velocity" | "polar" | "collision";

type ClassificationResult = {
  auc: number;
  averagePrecision: number;
  prevalence: number;
};

type ProbeResult = {
  taskScores: Record<string, number>;
  continuousR2: Record<string, number>;
  classification: Record<string, ClassificationResult>;
};

type Representation = ProbeResult & {
  id: string;
  label: string;
};

type InterventionTarget = {
  plusMinusDisplacementPx: number;
  medianDisplacementPx: number;
  expectedSignRate: number;
  directionL2: number;
};

type CausalEffect = {
  meanVxDeltaPxPerFrame?: number;
  meanSpeedDeltaPxPerFrame: number;
  finalXDeltaPx?: number;
  plusVsBaselinePxPerFrame?: number;
  expectedSignRate: number;
};

type CausalCurve = {
  vx: number[];
  speed: number[];
};

type InterpretabilityManifest = {
  modelKind?: string;
  checkpointStep: number;
  modelParameters: number;
  protocol: {
    samples: number;
    trainTrajectories: number;
    validationTrajectories: number;
    testTrajectories: number;
    historyFrames: number;
    predictionFrames: number;
    probePoint: string;
    split: string;
  };
  representations: Representation[];
  controls: {
    untrained: ProbeResult;
    actionOnly: ProbeResult;
    shuffledLabels: ProbeResult;
  };
  bestLayers: {
    velocity: string;
    collision: string;
  };
  intervention: {
    layer: string;
    samples: number;
    probeStandardDeviations: number;
    integrationSteps: number;
    targets: Record<string, InterventionTarget>;
  };
  downstreamIntervention: {
    lens: { contexts: number; sourceLayer: string; downstreamTarget: string; spatialAlignment: string };
    speedLens: { contexts: number; sourceLayer: string; downstreamTarget: string; spatialAlignment: string };
    activationLocator: {
      trainingSignal: string;
      deploymentSignal: string;
      layer: string;
      samples: number;
      testTrajectories: number;
      visualCellAccuracy: number;
      simulatorCellAccuracyEvaluationOnly: number;
      withinOneCellAccuracy: number;
      rolloutCellAgreementWithDecodedPosition: number;
      usedForCausalWrite: boolean;
    };
    write: {
      layer: string;
      rolloutFrames: number;
      samples: number;
      strengthProjectionSigmas: number;
      persistence: string;
      matchedNoise: boolean;
      probeJacobianCosine: number;
      velocitySpeedLensCosine: number;
    };
    effects: {
      downstreamJacobian: CausalEffect;
      coordinateOracleCeiling: CausalEffect;
      linearProbe: CausalEffect;
      randomDirection: CausalEffect;
      downstreamSpeed: CausalEffect;
      randomSpeedDirection: CausalEffect;
    };
    curves: Record<string, CausalCurve>;
  };
  targetDefinitions: Record<TaskId, string>;
};

const TASKS: { id: TaskId; label: string; metric: string }[] = [
  { id: "position", label: "X / Y", metric: "R²" },
  { id: "velocity", label: "Velocity", metric: "R²" },
  { id: "polar", label: "Polar", metric: "R²" },
  { id: "collision", label: "Collision", metric: "AUROC" },
];

const TARGETS: Record<TaskId, { id: string; label: string }[]> = {
  position: [
    { id: "player_x", label: "player x" },
    { id: "player_y", label: "player y" },
    { id: "puck_x", label: "puck x" },
    { id: "puck_y", label: "puck y" },
  ],
  velocity: [
    { id: "player_vx", label: "player vx" },
    { id: "player_vy", label: "player vy" },
    { id: "puck_vx", label: "puck vx" },
    { id: "puck_vy", label: "puck vy" },
    { id: "player_speed", label: "player speed" },
    { id: "puck_speed", label: "puck speed" },
  ],
  polar: [
    { id: "relative_distance", label: "distance" },
    { id: "bearing_cos", label: "bearing cos" },
    { id: "bearing_sin", label: "bearing sin" },
  ],
  collision: [
    { id: "any_collision", label: "any collision" },
    { id: "disc_impact", label: "disc impact" },
    { id: "wall_hit", label: "wall hit" },
  ],
};

function taskScore(result: ProbeResult, task: TaskId) {
  return result.taskScores[task];
}

function formatScore(value: number) {
  return (Math.abs(value) < 0.005 ? 0 : value).toFixed(2);
}

export function InterpretabilityViewer() {
  const [manifest, setManifest] = useState<InterpretabilityManifest | null>(null);
  const [studies, setStudies] = useState<Record<string, InterpretabilityManifest>>({});
  const [error, setError] = useState("");
  const [task, setTask] = useState<TaskId>("velocity");
  const [selectedId, setSelectedId] = useState("block-4");
  const [causalMetric, setCausalMetric] = useState<"velocity" | "speed">("velocity");

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch(`${BASE_PATH}/blocket-league/interpretability/direct-manifest.json`),
      fetch(`${BASE_PATH}/blocket-league/interpretability/diffusion-manifest.json`),
    ]).then(async ([directResponse, diffusionResponse]) => {
      if (!directResponse.ok || !diffusionResponse.ok) throw new Error("A comparison manifest is unavailable");
      return {
        direct: await directResponse.json() as InterpretabilityManifest,
        diffusion: await diffusionResponse.json() as InterpretabilityManifest,
      };
    }).then((value) => {
        if (!cancelled) {
          setStudies(value);
          setManifest(value.direct);
          setSelectedId(value.direct.bestLayers.velocity);
        }
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => { cancelled = true; };
  }, []);

  if (error) return <p className={styles.interpretabilityLoading}>Probe data unavailable: {error}</p>;
  if (!manifest) return <p className={styles.interpretabilityLoading}>Loading held-out probe study…</p>;

  const selected = manifest.representations.find((item) => item.id === selectedId) ?? manifest.representations[0];
  const baseline = task === "collision" ? 0.5 : 0;
  const plotLeft = 64;
  const plotRight = 760;
  const plotTop = 28;
  const plotBottom = 218;
  const xStep = (plotRight - plotLeft) / (manifest.representations.length - 1);
  const yPosition = (value: number) => {
    const normalized = Math.max(0, Math.min(1, (value - baseline) / (1 - baseline)));
    return plotBottom - normalized * (plotBottom - plotTop);
  };
  const points = manifest.representations.map((item, index) => ({
    x: plotLeft + index * xStep,
    y: yPosition(taskScore(item, task)),
    item,
  }));
  const pointString = points.map((point) => `${point.x},${point.y}`).join(" ");
  const controls = [
    { label: "selected", value: taskScore(selected, task) },
    { label: "codec", value: taskScore(manifest.representations[0], task) },
    { label: "untrained", value: taskScore(manifest.controls.untrained, task) },
    { label: "action only", value: taskScore(manifest.controls.actionOnly, task) },
    { label: "shuffled", value: taskScore(manifest.controls.shuffledLabels, task) },
  ];
  const velocityBest = Math.max(...manifest.representations.map((item) => taskScore(item, "velocity")));
  const positionBest = Math.max(...manifest.representations.map((item) => taskScore(item, "position")));
  const interventionValues = Object.values(manifest.intervention.targets);
  const meanIntervention = interventionValues.length ? interventionValues.reduce(
    (sum, value) => sum + Math.abs(value.plusMinusDisplacementPx),
    0,
  ) / interventionValues.length : 0;
  const isDirect = manifest.modelKind === "direct-autoregressive";
  const causal = manifest.downstreamIntervention;
  const causalSeries = causalMetric === "velocity"
    ? [
        { id: "jacobianPlus", label: "+ velocity write", values: causal.curves.jacobianPlus.vx },
        { id: "baseline", label: "frozen baseline", values: causal.curves.baseline.vx },
        { id: "jacobianMinus", label: "− velocity write", values: causal.curves.jacobianMinus.vx },
      ]
    : [
        { id: "speedPlus", label: "+ speed write", values: causal.curves.speedPlus.speed },
        { id: "baseline", label: "frozen baseline", values: causal.curves.baseline.speed },
        { id: "speedMinus", label: "− speed write", values: causal.curves.speedMinus.speed },
      ];
  const causalValues = causalSeries.flatMap((series) => series.values);
  const causalMin = Math.min(...causalValues);
  const causalMax = Math.max(...causalValues);
  const causalRange = Math.max(causalMax - causalMin, 0.25);
  const causalX = (index: number) => 52 + index * (680 / (causal.write.rolloutFrames - 1));
  const causalY = (value: number) => 205 - ((value - causalMin) / causalRange) * 164;
  const causalEffect = causalMetric === "velocity"
    ? causal.effects.downstreamJacobian.meanVxDeltaPxPerFrame ?? 0
    : causal.effects.downstreamSpeed.meanSpeedDeltaPxPerFrame;
  const causalRandom = causalMetric === "velocity"
    ? Math.abs(causal.effects.randomDirection.meanVxDeltaPxPerFrame ?? 0)
    : Math.abs(causal.effects.randomSpeedDirection.meanSpeedDeltaPxPerFrame);
  const causalProbe = Math.abs(causal.effects.linearProbe.meanVxDeltaPxPerFrame ?? 0);

  return (
    <div className={styles.interpretabilityShell}>
      <div className={styles.probeTaskBar} role="group" aria-label="Choose a world model study">
        {([
          ["direct", "Direct transformer", "1 pass / step"],
          ["diffusion", "Diffusion transformer", "6 passes / step"],
        ] as const).map(([id, label, detail]) => (
          <button
            key={id}
            type="button"
            className={(isDirect ? id === "direct" : id === "diffusion") ? styles.probeTaskActive : undefined}
            aria-pressed={isDirect ? id === "direct" : id === "diffusion"}
            onClick={() => {
              const study = studies[id];
              if (!study) return;
              setManifest(study);
              setSelectedId(study.bestLayers.velocity);
            }}
          >
            {label}<span>{detail}</span>
          </button>
        ))}
      </div>
      <div className={styles.probeProtocol}>
        <div><span>DATA</span><strong>{manifest.protocol.samples} trajectories</strong></div>
        <div><span>HELD OUT</span><strong>{manifest.protocol.testTrajectories} trajectories</strong></div>
        <div><span>READ POINT</span><strong>{isDirect ? "single transition" : "flow t = 0"}</strong></div>
      </div>

      <div className={styles.evidenceStrip}>
        <div>
          <span>GEOMETRY</span>
          <strong>{positionBest.toFixed(2)} R²</strong>
          <small>linearly recoverable</small>
        </div>
        <div>
          <span>VELOCITY</span>
          <strong>{velocityBest.toFixed(2)} R²</strong>
          <small>emerges inside the transformer</small>
        </div>
        <div className={styles.evidenceCausal}>
          <span>CAUSAL CONTROL</span>
          <strong>{causal.effects.downstreamJacobian.finalXDeltaPx?.toFixed(1)} PX</strong>
          <small>12-frame +/− velocity separation</small>
        </div>
      </div>

      <div className={styles.probeTaskBar} role="group" aria-label="Choose a physical quantity to probe">
        {TASKS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={task === item.id ? styles.probeTaskActive : undefined}
            aria-pressed={task === item.id}
            onClick={() => {
              setTask(item.id);
              const best = manifest.representations.reduce((currentBest, current) => (
                taskScore(current, item.id) > taskScore(currentBest, item.id) ? current : currentBest
              ));
              setSelectedId(best.id);
            }}
          >
            {item.label}<span>{item.metric}</span>
          </button>
        ))}
      </div>

      <div className={styles.probeGrid}>
        <div className={styles.probeChart}>
          <div className={styles.probeChartHeader}>
            <div><span>HELD-OUT LINEAR READOUT</span><strong>{manifest.targetDefinitions[task]}</strong></div>
            <div>{task === "collision" ? "AUROC · 0.50 = CHANCE" : "R² · 0 = MEAN PREDICTOR"}</div>
          </div>
          <svg
            viewBox="0 0 824 274"
            role="img"
            aria-label={`${task} probe scores across the codec, model input, and six transformer blocks.`}
          >
            {[0, 0.5, 1].map((fraction) => {
              const y = plotBottom - fraction * (plotBottom - plotTop);
              const value = baseline + fraction * (1 - baseline);
              return (
                <g key={fraction}>
                  <line x1={plotLeft} x2={plotRight} y1={y} y2={y} />
                  <text x={plotLeft - 12} y={y + 4} textAnchor="end">{value.toFixed(2)}</text>
                </g>
              );
            })}
            <polyline className={styles.probeLine} points={pointString} />
            {points.map((point) => (
              <g key={point.item.id} className={point.item.id === selected.id ? styles.probePointSelected : undefined}>
                <circle cx={point.x} cy={point.y} r={point.item.id === selected.id ? 7 : 5} />
                <text x={point.x} y={plotBottom + 31} textAnchor="middle">{point.item.label.replace(" latent", "")}</text>
                <text className={styles.probeValueLabel} x={point.x} y={point.y - 12} textAnchor="middle">
                  {formatScore(taskScore(point.item, task))}
                </text>
              </g>
            ))}
          </svg>
          <div className={styles.probeLayerPicker} role="group" aria-label="Inspect a representation">
            {manifest.representations.map((item) => (
              <button
                key={item.id}
                type="button"
                className={selected.id === item.id ? styles.probeLayerActive : undefined}
                aria-pressed={selected.id === item.id}
                onClick={() => setSelectedId(item.id)}
              >
                {item.label}<span>{formatScore(taskScore(item, task))}</span>
              </button>
            ))}
          </div>
        </div>

        <aside className={styles.probeDetail} aria-label="Selected probe details">
          <div className={styles.probeDetailTitle}>
            <span>{selected.label.toUpperCase()}</span>
            <strong>{formatScore(taskScore(selected, task))} {task === "collision" ? "AUROC" : "R²"}</strong>
          </div>
          <dl className={styles.probeTargets}>
            {TARGETS[task].map((target) => {
              const value = task === "collision"
                ? selected.classification[target.id]?.auc
                : selected.continuousR2[target.id];
              return <div key={target.id}><dt>{target.label}</dt><dd>{value === undefined ? "—" : formatScore(value)}</dd></div>;
            })}
          </dl>
          <div className={styles.probeControls}>
            <span>CONTROLS</span>
            {controls.map((control) => (
              <div key={control.label}>
                <span>{control.label}</span>
                <i><i style={{ width: `${Math.max(0, Math.min(1, (control.value - baseline) / (1 - baseline))) * 100}%` }} /></i>
                <strong>{formatScore(control.value)}</strong>
              </div>
            ))}
          </div>
        </aside>
      </div>

      {interventionValues.length > 0 && <div className={styles.causalAudit}>
        <div>
          <span>CAUSAL AUDIT · {manifest.intervention.layer.toUpperCase()}</span>
          <h3>Readable does not yet mean writable.</h3>
          <p>
            We pushed each velocity readout by ±{manifest.intervention.probeStandardDeviations.toFixed(0)} probe σ,
            then ran the frozen flow sampler with matched noise. The average decoded displacement changed by only
            {` ${meanIntervention.toFixed(3)} px`}. This edit is not causal evidence.
          </p>
        </div>
        <div className={styles.causalRows}>
          {Object.entries(manifest.intervention.targets).map(([name, value]) => (
            <div key={name}>
              <span>{name.replace("_", " ")}</span>
              <strong>{value.plusMinusDisplacementPx >= 0 ? "+" : ""}{value.plusMinusDisplacementPx.toFixed(3)} px</strong>
              <small>{(value.expectedSignRate * 100).toFixed(0)}% expected sign</small>
            </div>
          ))}
        </div>
      </div>}

      <div className={styles.causalControl}>
        <div className={styles.activationLocatorStrip}>
          <div><span>ACTIVATION LOCATOR</span><strong>{(causal.activationLocator.visualCellAccuracy * 100).toFixed(1)}%</strong><small>held-out spatial-cell accuracy</small></div>
          <div><span>ROLLOUT TRACKING</span><strong>{(causal.activationLocator.rolloutCellAgreementWithDecodedPosition * 100).toFixed(1)}%</strong><small>agreement with decoded position</small></div>
          <div><span>COORDINATE LABELS AT WRITE</span><strong>ZERO</strong><small>{causal.write.layer} activations choose the cell</small></div>
        </div>
        <div className={styles.causalControlHeader}>
          <div>
            <span>DOWNSTREAM-AVERAGED WRITE · {causal.write.layer.toUpperCase()}</span>
            <h3>The model’s puck state is writable.</h3>
            <p>
              We averaged the activation-to-render Jacobian over {causal.lens.contexts} worlds. A separate token scorer
              finds the puck from {causal.write.layer} activations, then places the causal template at every {isDirect ? "autoregressive transition" : "denoising evaluation"} for {causal.write.rolloutFrames} frames.
            </p>
          </div>
          <div className={styles.causalMetricPicker} role="group" aria-label="Choose causal write metric">
            <button type="button" aria-pressed={causalMetric === "velocity"} onClick={() => setCausalMetric("velocity")}>X velocity</button>
            <button type="button" aria-pressed={causalMetric === "speed"} onClick={() => setCausalMetric("speed")}>Speed</button>
          </div>
        </div>

        <div className={styles.causalControlGrid}>
          <div className={styles.causalTrajectoryChart}>
            <div className={styles.causalLegend}>
              {causalSeries.map((series) => <span key={series.id} data-series={series.id}><i />{series.label}</span>)}
            </div>
            <svg viewBox="0 0 760 248" role="img" aria-label={`Average puck ${causalMetric} over a 12-frame causal intervention rollout.`}>
              {[0, 0.5, 1].map((fraction) => {
                const value = causalMin + causalRange * fraction;
                const y = causalY(value);
                return <g key={fraction}><line x1="52" x2="732" y1={y} y2={y} /><text x="43" y={y + 4} textAnchor="end">{value.toFixed(1)}</text></g>;
              })}
              {causalSeries.map((series) => (
                <polyline
                  key={series.id}
                  data-series={series.id}
                  points={series.values.map((value, index) => `${causalX(index)},${causalY(value)}`).join(" ")}
                />
              ))}
              {[0, 3, 7, 11].map((frame) => <text key={frame} x={causalX(frame)} y="232" textAnchor="middle">{frame + 1}</text>)}
              <text x="732" y="244" textAnchor="end">PREDICTED FRAME</text>
            </svg>
          </div>

          <aside className={styles.causalResult}>
            <span>{causalMetric === "velocity" ? "SIGNED VELOCITY EFFECT" : "SPEED EFFECT"}</span>
            <strong>{causalEffect >= 0 ? "+" : ""}{causalEffect.toFixed(2)} <small>px / frame</small></strong>
            <p>{causalMetric === "velocity"
              ? `${(causal.effects.downstreamJacobian.expectedSignRate * 100).toFixed(0)}% of held-out rollouts move in the intended direction.`
              : `${(causal.effects.downstreamSpeed.expectedSignRate * 100).toFixed(0)}% of held-out rollouts become faster under the positive write.`}</p>
            <dl>
              <div><dt>averaged Jacobian</dt><dd>{Math.abs(causalEffect).toFixed(3)}</dd></div>
              {causalMetric === "velocity" && <div><dt>linear probe</dt><dd>{causalProbe.toFixed(3)}</dd></div>}
              <div><dt>random direction</dt><dd>{causalRandom.toFixed(3)}</dd></div>
            </dl>
          </aside>
        </div>

        <div className={styles.causalMethodLine}>
          <span>FROZEN WEIGHTS</span><span>{isDirect ? "DETERMINISTIC PAIRS" : "MATCHED NOISE"}</span><span>±{causal.write.strengthProjectionSigmas.toFixed(0)}σ WRITE</span>
          <span>ACTIVATION-LOCATED</span><span>{causal.write.samples} HELD-OUT ROLLOUTS</span>
        </div>
      </div>

      {isDirect && <CausalInterventionViewer />}

      <p className={styles.probeFootnote}>
        Frozen checkpoint {manifest.checkpointStep.toLocaleString()} · train/validation/test trajectories never overlap ·
        simulator state supplies post-hoc probe labels only · linear probes tune regularization on validation only ·
        {isDirect ? " the next latent is directly regressed" : " the future token begins as noise"}.
      </p>
    </div>
  );
}
