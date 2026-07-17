"use client";

import { useEffect, useState } from "react";

import styles from "./blocket-league-lab.module.css";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

type Probe = { block: number; position_r2: number; velocity_r2: number };
type Effect = {
  release_x_delta_px?: number;
  final_x_delta_px?: number;
  release_y_delta_px?: number;
  final_y_delta_px?: number;
  post_release_growth_px: number;
  expected_sign_fraction?: number;
  positive_fraction?: number;
  samples: number;
};
type PixelManifest = {
  checkpointStep: number;
  parameters: number;
  fitSamples: number;
  testSamples: number;
  labelSource: string;
  actionConditioning: boolean;
  probes: Probe[];
  causal: {
    block: number;
    strength: number;
    writeFrames: number;
    rolloutFrames: number;
    effects: Record<string, Effect>;
  };
};

function score(value: number) {
  return value.toFixed(2);
}

export function PixelInterpretabilityViewer() {
  const [manifest, setManifest] = useState<PixelManifest | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetch(`${BASE_PATH}/blocket-league/interpretability/passive-pixel-manifest.json`)
      .then((response) => {
        if (!response.ok) throw new Error("The passive-pixel study is unavailable");
        return response.json() as Promise<PixelManifest>;
      })
      .then((value) => { if (!cancelled) setManifest(value); })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => { cancelled = true; };
  }, []);

  if (error) return <p className={styles.interpretabilityLoading}>Study unavailable: {error}</p>;
  if (!manifest) return <p className={styles.interpretabilityLoading}>Loading passive-pixel study…</p>;

  const plotLeft = 54;
  const plotRight = 746;
  const plotTop = 24;
  const plotBottom = 214;
  const x = (index: number) => plotLeft + index * (plotRight - plotLeft) / (manifest.probes.length - 1);
  const y = (value: number) => plotBottom - Math.max(0, Math.min(1, value)) * (plotBottom - plotTop);
  const positionPoints = manifest.probes.map((probe, index) => `${x(index)},${y(probe.position_r2)}`).join(" ");
  const velocityPoints = manifest.probes.map((probe, index) => `${x(index)},${y(probe.velocity_r2)}`).join(" ");
  const bestVelocity = Math.max(...manifest.probes.map((probe) => probe.velocity_r2));
  const bestProbe = manifest.probes.reduce((best, probe) => (
    probe.velocity_r2 > best.velocity_r2 ? probe : best
  ));
  const plus = manifest.causal.effects.x_plus;
  const minus = manifest.causal.effects.x_minus;
  const yPlus = manifest.causal.effects.y_plus;
  const yMinus = manifest.causal.effects.y_minus;
  const random = manifest.causal.effects.random;
  const effectY = (value: number) => 122 - value * 54;

  return (
    <div className={styles.interpretabilityShell}>
      <div className={styles.probeProtocol}>
        <div><span>TRAINING SIGNAL</span><strong>passive rendered pixels only</strong></div>
        <div><span>MODEL</span><strong>3.67M parameters · 6 blocks</strong></div>
        <div><span>HELD-OUT PROTOCOL</span><strong>{manifest.fitSamples} fit · {manifest.testSamples} test</strong></div>
      </div>

      <div className={styles.evidenceStrip}>
        <div><span>PREDICTION</span><strong>0.93 px</strong><small>mean entity error · frames 1–12</small></div>
        <div><span>READOUT</span><strong>{score(bestVelocity)} R²</strong><small>green-circle velocity · block {bestProbe.block}</small></div>
        <div className={styles.evidenceCausal}><span>CAUSAL WRITE</span><strong>+{plus.final_x_delta_px?.toFixed(2)} px</strong><small>four writes · measured at frame 12</small></div>
      </div>

      <div className={styles.probeGrid}>
        <div className={styles.probeChart}>
          <div className={styles.probeChartHeader}>
            <div><span>WHAT BECOMES LINEAR?</span><strong>R² ON UNSEEN PASSIVE CLIPS</strong></div>
            <div>POSITION — GOLD · VELOCITY — MINT</div>
          </div>
          <svg viewBox="0 0 800 258" role="img" aria-label="Position and velocity probe scores across six transformer blocks.">
            {[0, 0.25, 0.5, 0.75, 1].map((tick) => <g key={tick}><line x1={plotLeft} x2={plotRight} y1={y(tick)} y2={y(tick)} /><text x="42" y={y(tick) + 3} textAnchor="end">{tick.toFixed(2)}</text></g>)}
            <polyline points={positionPoints} style={{ fill: "none", stroke: "var(--gold)", strokeWidth: 2 }} />
            <polyline className={styles.probeLine} points={velocityPoints} />
            {manifest.probes.map((probe, index) => <g key={probe.block}>
              <circle cx={x(index)} cy={y(probe.velocity_r2)} r="5" />
              <circle cx={x(index)} cy={y(probe.position_r2)} r="4" style={{ stroke: "var(--gold)" }} />
              <text x={x(index)} y="242" textAnchor="middle">B{probe.block}</text>
            </g>)}
          </svg>
        </div>
        <aside className={styles.probeDetail}>
          <div className={styles.probeDetailTitle}><span>BLOCK {bestProbe.block}</span><strong>{score(bestVelocity)} R²</strong></div>
          <dl className={styles.probeTargets}>
            <div><dt>position</dt><dd>{score(bestProbe.position_r2)}</dd></div>
            <div><dt>velocity</dt><dd>{score(bestProbe.velocity_r2)}</dd></div>
            <div><dt>action features</dt><dd>none</dd></div>
            <div><dt>label source</dt><dd>pixels</dd></div>
          </dl>
          <div className={styles.probeControls}><span>VELOCITY EMERGES WITH DEPTH</span>{manifest.probes.map((probe) => <div key={probe.block}><span>B{probe.block}</span><i><i style={{ width: `${Math.max(0, probe.velocity_r2) * 100}%` }} /></i><strong>{score(probe.velocity_r2)}</strong></div>)}</div>
        </aside>
      </div>

      <div className={styles.causalControl}>
        <div className={styles.causalControlHeader}>
          <div><span>DOWNSTREAM-AVERAGED DIRECTION</span><h3>Write briefly. Watch the new velocity persist.</h3><p>A single +x vector is averaged on the fit split, frozen, and written only during frames 1–4. The write then stops. If it merely redrew position, the gap would stop growing; instead it widens through frame 12.</p></div>
        </div>
        <div className={styles.causalControlGrid}>
          <div className={styles.causalTrajectoryChart}>
            <div className={styles.causalLegend}><span data-series="jacobianPlus"><i /> +x direction</span><span data-series="jacobianMinus"><i /> −x direction</span><span><i /> random direction</span></div>
            <svg viewBox="0 0 760 180" role="img" aria-label="Measured displacement at write release and final rollout frame.">
              <line x1="80" x2="700" y1={effectY(0)} y2={effectY(0)} />
              <line x1="250" x2="250" y1="24" y2="158" />
              <text x="250" y="172" textAnchor="middle">WRITE STOPS · FRAME 4</text>
              <text x="650" y="172" textAnchor="middle">FRAME 12</text>
              <polyline data-series="jacobianPlus" points={`250,${effectY(plus.release_x_delta_px ?? 0)} 650,${effectY(plus.final_x_delta_px ?? 0)}`} />
              <polyline data-series="jacobianMinus" points={`250,${effectY(minus.release_x_delta_px ?? 0)} 650,${effectY(minus.final_x_delta_px ?? 0)}`} />
              <polyline points={`250,${effectY(random.release_x_delta_px ?? 0)} 650,${effectY(random.final_x_delta_px ?? 0)}`} />
              <text x="660" y={effectY(plus.final_x_delta_px ?? 0) + 3}>+{plus.final_x_delta_px?.toFixed(2)} px</text>
              <text x="660" y={effectY(minus.final_x_delta_px ?? 0) + 3}>{minus.final_x_delta_px?.toFixed(2)} px</text>
            </svg>
          </div>
          <aside className={styles.causalResult}>
            <span>POST-RELEASE EFFECT</span>
            <strong>+{plus.post_release_growth_px.toFixed(2)} <small>px after writes stop</small></strong>
            <p>{((plus.expected_sign_fraction ?? 0) * 100).toFixed(1)}% of unseen rollouts move in the intended +x direction.</p>
            <dl><div><dt>+x / −x final</dt><dd>+{plus.final_x_delta_px?.toFixed(2)} / {minus.final_x_delta_px?.toFixed(2)} px</dd></div><div><dt>+y / −y final</dt><dd>+{yPlus?.final_y_delta_px?.toFixed(2)} / {yMinus?.final_y_delta_px?.toFixed(2)} px</dd></div><div><dt>random final</dt><dd>{random.final_x_delta_px?.toFixed(2)} px</dd></div></dl>
          </aside>
        </div>
        <div className={styles.causalMethodLine}><span>FROZEN WEIGHTS</span><span>NO ACTION CHANNEL</span><span>BLOCK {manifest.causal.block}</span><span>{manifest.causal.writeFrames} WRITE FRAMES</span><span>{manifest.testSamples} UNSEEN WORLDS</span></div>
      </div>

      <p className={styles.probeFootnote}>Recovery fine-tune {manifest.checkpointStep.toLocaleString()} steps · all readouts are derived from rendered pixels · simulator state is used only for trajectory-error evaluation · the intervention direction is global across rollouts.</p>
    </div>
  );
}
