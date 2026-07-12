"use client";

import {
	AudioLines,
	ExternalLink,
	Gauge,
	Radio,
	ShieldCheck,
	TriangleAlert,
	Volume2,
	VolumeX,
	Waves,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import analysis from "@/lib/acoustics/chainsaw-analysis.json";
import {
	pressureFieldColor,
	WAVE_FIELD_BACKGROUND,
} from "@/lib/acoustics/wave-palette";

type Complex = { re: number; im: number };
type Point = { x: number; y: number };
type WeightedPoint = Point & { weight: number };

type FrequencyController = {
	frequency: number;
	waveNumber: number;
	weights: Complex[];
	reductionDb: number;
};

const WORLD = { width: 12, height: 8 };
const SOURCE = { x: 4, y: 4 };
const HUMAN = { x: 9.2, y: 4.05 };
const SOUND_SPEED = 343;
const BUBBLE_RADIUS = 0.48;
const SPEAKER_COUNT = 12;
const VISUAL_TIME_SCALE = 0.035;

const AURA = {
	orange: "#ffc247",
	blue: "#168bd2",
	red: "#ff3b24",
};

function cAdd(left: Complex, right: Complex): Complex {
	return { re: left.re + right.re, im: left.im + right.im };
}

function cMul(left: Complex, right: Complex): Complex {
	return {
		re: left.re * right.re - left.im * right.im,
		im: left.re * right.im + left.im * right.re,
	};
}

function cConjugate(value: Complex): Complex {
	return { re: value.re, im: -value.im };
}

function magnitudeSquared(value: Complex) {
	return value.re * value.re + value.im * value.im;
}

function distance(left: Point, right: Point) {
	return Math.hypot(left.x - right.x, left.y - right.y);
}

function green(from: Point, to: Point, waveNumber: number): Complex {
	const radius = Math.max(0.09, distance(from, to));
	const amplitude = 1 / Math.sqrt(radius);
	const phase = -waveNumber * radius;
	return {
		re: amplitude * Math.cos(phase),
		im: amplitude * Math.sin(phase),
	};
}

function buildRing(count: number, radius = 1.55) {
	return Array.from({ length: count }, (_, index) => {
		const angle = (index / count) * Math.PI * 2;
		return {
			x: SOURCE.x + Math.cos(angle) * radius,
			y: SOURCE.y + Math.sin(angle) * radius,
		};
	});
}

function pointsOnDisk(center: Point, radius: number): WeightedPoint[] {
	const points: WeightedPoint[] = [{ ...center, weight: 2.5 }];
	for (const ring of [
		{ radius: radius * 0.5, count: 8, weight: 1.5 },
		{ radius, count: 14, weight: 1 },
	]) {
		for (let index = 0; index < ring.count; index += 1) {
			const angle = (index / ring.count) * Math.PI * 2;
			points.push({
				x: center.x + Math.cos(angle) * ring.radius,
				y: center.y + Math.sin(angle) * ring.radius,
				weight: ring.weight,
			});
		}
	}
	return points;
}

function guardPoints(center: Point): WeightedPoint[] {
	return Array.from({ length: 20 }, (_, index) => {
		const angle = (index / 20) * Math.PI * 2;
		return {
			x: center.x + Math.cos(angle) * 1.1,
			y: center.y + Math.sin(angle) * 1.1,
			weight: 0.025,
		};
	});
}

function solveLinearSystem(matrix: number[][], vector: number[]) {
	const size = vector.length;
	const rows = matrix.map((row, index) => [...row, vector[index]]);
	for (let column = 0; column < size; column += 1) {
		let pivot = column;
		for (let row = column + 1; row < size; row += 1) {
			if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) {
				pivot = row;
			}
		}
		if (Math.abs(rows[pivot][column]) < 1e-10) return Array(size).fill(0);
		[rows[column], rows[pivot]] = [rows[pivot], rows[column]];
		const divisor = rows[column][column];
		for (let entry = column; entry <= size; entry += 1) {
			rows[column][entry] /= divisor;
		}
		for (let row = 0; row < size; row += 1) {
			if (row === column) continue;
			const factor = rows[row][column];
			for (let entry = column; entry <= size; entry += 1) {
				rows[row][entry] -= factor * rows[column][entry];
			}
		}
	}
	return rows.map((row) => row[size]);
}

const SPEAKERS = buildRing(SPEAKER_COUNT);
const TARGETS = pointsOnDisk(HUMAN, BUBBLE_RADIUS);
const GUARDS = guardPoints(HUMAN);

function fieldAt(
	point: Point,
	controller: FrequencyController,
	controlEnabled: boolean,
) {
	let pressure = green(SOURCE, point, controller.waveNumber);
	if (!controlEnabled) return pressure;
	for (let index = 0; index < SPEAKERS.length; index += 1) {
		pressure = cAdd(
			pressure,
			cMul(
				controller.weights[index] ?? { re: 0, im: 0 },
				green(SPEAKERS[index], point, controller.waveNumber),
			),
		);
	}
	return pressure;
}

function solveFrequencyController(frequency: number): FrequencyController {
	const waveNumber = (Math.PI * 2 * frequency) / SOUND_SPEED;
	const size = SPEAKERS.length;
	const normal = Array.from({ length: size }, () =>
		Array.from({ length: size }, () => ({ re: 0, im: 0 })),
	);
	const target = Array.from({ length: size }, () => ({ re: 0, im: 0 }));

	const addRow = (point: WeightedPoint, cancelSource: boolean) => {
		const columns = SPEAKERS.map((speaker) => green(speaker, point, waveNumber));
		const desired = cancelSource
			? cMul(green(SOURCE, point, waveNumber), { re: -1, im: 0 })
			: { re: 0, im: 0 };
		for (let left = 0; left < size; left += 1) {
			const conjugate = cConjugate(columns[left]);
			target[left] = cAdd(
				target[left],
				cMul(cMul(conjugate, desired), { re: point.weight, im: 0 }),
			);
			for (let right = 0; right < size; right += 1) {
				normal[left][right] = cAdd(
					normal[left][right],
					cMul(
						cMul(conjugate, columns[right]),
						{ re: point.weight, im: 0 },
					),
				);
			}
		}
	};

	for (const point of TARGETS) addRow(point, true);
	for (const point of GUARDS) addRow(point, false);
	const diagonalMean =
		normal.reduce((sum, row, index) => sum + row[index].re, 0) / size;
	for (let index = 0; index < size; index += 1) {
		normal[index][index].re += diagonalMean * 0.012 + 1e-6;
	}

	const realSize = size * 2;
	const realMatrix = Array.from({ length: realSize }, () =>
		Array(realSize).fill(0),
	);
	const realTarget = Array(realSize).fill(0);
	for (let row = 0; row < size; row += 1) {
		realTarget[row] = target[row].re;
		realTarget[row + size] = target[row].im;
		for (let column = 0; column < size; column += 1) {
			const value = normal[row][column];
			realMatrix[row][column] = value.re;
			realMatrix[row][column + size] = -value.im;
			realMatrix[row + size][column] = value.im;
			realMatrix[row + size][column + size] = value.re;
		}
	}
	const solved = solveLinearSystem(realMatrix, realTarget);
	const weights = Array.from({ length: size }, (_, index) => {
		const value = { re: solved[index], im: solved[index + size] };
		const magnitude = Math.hypot(value.re, value.im);
		const scale = magnitude > 4 ? 4 / magnitude : 1;
		return { re: value.re * scale, im: value.im * scale };
	});

	const temporary: FrequencyController = {
		frequency,
		waveNumber,
		weights,
		reductionDb: 0,
	};
	let sourcePower = 0;
	let controlledPower = 0;
	let totalWeight = 0;
	for (const point of TARGETS) {
		sourcePower +=
			point.weight * magnitudeSquared(green(SOURCE, point, waveNumber));
		controlledPower +=
			point.weight * magnitudeSquared(fieldAt(point, temporary, true));
		totalWeight += point.weight;
	}
	const ratio =
		(controlledPower / totalWeight) / (sourcePower / totalWeight);
	return {
		...temporary,
		reductionDb: 10 * Math.log10(Math.max(1e-10, ratio)),
	};
}

const COMPONENTS = analysis.harmonics.map((harmonic) => ({
	...harmonic,
	controller: solveFrequencyController(harmonic.frequency),
}));

const PERIODIC_POWER = COMPONENTS.reduce(
	(sum, component) => sum + component.powerShare,
	0,
);
const PERIODIC_RESIDUAL =
	COMPONENTS.reduce(
		(sum, component) =>
			sum +
			component.powerShare *
				10 ** (component.controller.reductionDb / 10),
		0,
	) / PERIODIC_POWER;
const PERIODIC_REDUCTION_DB = 10 * Math.log10(PERIODIC_RESIDUAL);
const PERIODIC_FRACTION = analysis.segment.periodicEnergyFraction;
const FULL_SIGNAL_RESIDUAL =
	1 - PERIODIC_FRACTION + PERIODIC_FRACTION * PERIODIC_RESIDUAL;
const FULL_SIGNAL_REDUCTION_DB = 10 * Math.log10(FULL_SIGNAL_RESIDUAL);
const VISUAL_COMPONENTS = [...COMPONENTS]
	.sort((left, right) => right.powerShare - left.powerShare)
	.slice(0, 5);

function resizeCanvas(canvas: HTMLCanvasElement) {
	const rect = canvas.getBoundingClientRect();
	const dpr = Math.min(window.devicePixelRatio || 1, 2);
	const width = Math.max(1, Math.round(rect.width * dpr));
	const height = Math.max(1, Math.round(rect.height * dpr));
	if (canvas.width !== width || canvas.height !== height) {
		canvas.width = width;
		canvas.height = height;
	}
	return { width, height, dpr };
}

function getTransform(width: number, height: number) {
	const padding = Math.min(width, height) * 0.035;
	const scale = Math.min(
		(width - padding * 2) / WORLD.width,
		(height - padding * 2) / WORLD.height,
	);
	return {
		scale,
		offsetX: (width - WORLD.width * scale) / 2,
		offsetY: (height - WORLD.height * scale) / 2,
	};
}

function formatDb(value: number) {
	return `${value > 0 ? "+" : ""}${value.toFixed(1)} dB`;
}

function WaveformPlot() {
	const width = 1000;
	const height = 150;
	const center = height / 2;
	const scale = height * 0.42;
	const top = analysis.waveformMax
		.map((value, index) => {
			const x = (index / (analysis.waveformMax.length - 1)) * width;
			return `${x.toFixed(2)},${(center - value * scale).toFixed(2)}`;
		})
		.join(" ");
	const bottom = [...analysis.waveformMin]
		.reverse()
		.map((value, reverseIndex) => {
			const index = analysis.waveformMin.length - 1 - reverseIndex;
			const x = (index / (analysis.waveformMin.length - 1)) * width;
			return `${x.toFixed(2)},${(center - value * scale).toFixed(2)}`;
		})
		.join(" ");
	return (
		<svg
			viewBox={`0 0 ${width} ${height}`}
			className="h-28 w-full"
			role="img"
			aria-label="Amplitude envelope of the extracted chainsaw recording"
		>
			<line x1="0" y1={center} x2={width} y2={center} stroke="rgba(242,238,228,.12)" />
			<polygon points={`${top} ${bottom}`} fill="rgba(255,194,71,.18)" stroke="#ffc247" strokeWidth="1.4" />
		</svg>
	);
}

export function ChainsawLab({
	running,
	active,
	showControls,
}: {
	running: boolean;
	active: boolean;
	showControls: boolean;
}) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const frameRef = useRef(0);
	const lastFrameRef = useRef(0);
	const timeRef = useRef(0);
	const [controlEnabled, setControlEnabled] = useState(true);

	const draw = useCallback(
		(timestamp: number) => {
			if (!active) return;
			if (timestamp - lastFrameRef.current < 50) return;
			const delta = Math.min(0.08, (timestamp - lastFrameRef.current) / 1000);
			lastFrameRef.current = timestamp;
			if (running) timeRef.current += delta;
			const canvas = canvasRef.current;
			if (!canvas) return;
			const { width, height, dpr } = resizeCanvas(canvas);
			const context = canvas.getContext("2d");
			if (!context) return;
			const transform = getTransform(width, height);
			const toCanvas = (point: Point) => ({
				x: transform.offsetX + point.x * transform.scale,
				y: transform.offsetY + point.y * transform.scale,
			});

			context.fillStyle = WAVE_FIELD_BACKGROUND;
			context.fillRect(0, 0, width, height);
			const step = Math.max(9, Math.round(8 * dpr));
			for (
				let pixelY = transform.offsetY;
				pixelY < transform.offsetY + WORLD.height * transform.scale;
				pixelY += step
			) {
				for (
					let pixelX = transform.offsetX;
					pixelX < transform.offsetX + WORLD.width * transform.scale;
					pixelX += step
				) {
					const point = {
						x: (pixelX - transform.offsetX) / transform.scale,
						y: (pixelY - transform.offsetY) / transform.scale,
					};
					let instantaneous = 0;
					let amplitudeSum = 0;
					for (const component of VISUAL_COMPONENTS) {
						const pressure = fieldAt(
							point,
							component.controller,
							controlEnabled,
						);
						const phase =
							Math.PI * 2 * component.frequency * timeRef.current * VISUAL_TIME_SCALE +
							component.phase;
						instantaneous +=
							component.amplitude *
							(pressure.re * Math.cos(phase) - pressure.im * Math.sin(phase));
						amplitudeSum += component.amplitude;
					}
					const signedStrength = Math.tanh(
						(instantaneous / Math.max(amplitudeSum, 0.1)) * 1.8,
					);
					context.fillStyle = pressureFieldColor(signedStrength, 0.78);
					context.fillRect(pixelX, pixelY, step + 1, step + 1);
				}
			}

			context.lineWidth = dpr;
			context.strokeStyle = "rgba(242,238,228,.08)";
			for (let x = 0; x <= WORLD.width; x += 1) {
				const start = toCanvas({ x, y: 0 });
				const end = toCanvas({ x, y: WORLD.height });
				context.beginPath();
				context.moveTo(start.x, start.y);
				context.lineTo(end.x, end.y);
				context.stroke();
			}
			for (let y = 0; y <= WORLD.height; y += 1) {
				const start = toCanvas({ x: 0, y });
				const end = toCanvas({ x: WORLD.width, y });
				context.beginPath();
				context.moveTo(start.x, start.y);
				context.lineTo(end.x, end.y);
				context.stroke();
			}

			for (const speaker of SPEAKERS) {
				const point = toCanvas(speaker);
				context.fillStyle = "#0b0e12";
				context.strokeStyle = controlEnabled ? "#2f6df6" : "rgba(47,109,246,.38)";
				context.lineWidth = 2 * dpr;
				context.beginPath();
				context.arc(point.x, point.y, 8 * dpr, 0, Math.PI * 2);
				context.fill();
				context.stroke();
			}

			const source = toCanvas(SOURCE);
			context.fillStyle = AURA.orange;
			context.beginPath();
			context.arc(source.x, source.y, 8 * dpr, 0, Math.PI * 2);
			context.fill();
			context.strokeStyle = "rgba(255,194,71,.35)";
			context.lineWidth = 9 * dpr;
			context.stroke();

			const human = toCanvas(HUMAN);
			context.fillStyle = controlEnabled
				? "rgba(59,185,232,.06)"
				: "rgba(255,59,36,.04)";
			context.strokeStyle = controlEnabled ? "#3bb9e8" : AURA.red;
			context.lineWidth = 2 * dpr;
			context.beginPath();
			context.arc(human.x, human.y, BUBBLE_RADIUS * transform.scale, 0, Math.PI * 2);
			context.fill();
			context.stroke();
			context.fillStyle = "#0b0e12";
			context.beginPath();
			context.arc(human.x, human.y, 11 * dpr, 0, Math.PI * 2);
			context.fill();
			context.stroke();
			context.beginPath();
			context.moveTo(human.x - 15 * dpr, human.y);
			context.lineTo(human.x + 15 * dpr, human.y);
			context.moveTo(human.x, human.y - 15 * dpr);
			context.lineTo(human.x, human.y + 15 * dpr);
			context.stroke();
		},
		[active, controlEnabled, running],
	);

	useEffect(() => {
		if (!active) return;
		const animate = (timestamp: number) => {
			draw(timestamp);
			frameRef.current = requestAnimationFrame(animate);
		};
		frameRef.current = requestAnimationFrame(animate);
		return () => cancelAnimationFrame(frameRef.current);
	}, [active, draw]);

	const shownPeriodicDb = controlEnabled ? PERIODIC_REDUCTION_DB : 0;
	const shownFullDb = controlEnabled ? FULL_SIGNAL_REDUCTION_DB : 0;

	return (
		<section className={`mx-auto grid max-w-[1500px] gap-4 p-4 sm:p-6 ${showControls ? "lg:grid-cols-[minmax(0,1fr)_340px]" : ""}`}>
			<div className="space-y-4">
				<div className="relative aspect-[3/2] min-h-[420px] max-h-[720px] overflow-hidden rounded-2xl border border-[#ffc247]/20 bg-[#070a0d] shadow-2xl shadow-black/30">
					<canvas
						ref={canvasRef}
						data-testid="chainsaw-canvas"
						className="absolute inset-0 size-full"
						role="img"
						aria-label="Multi-frequency chainsaw cancellation field around one listener"
					/>
					<div className="pointer-events-none absolute left-4 top-4 flex flex-wrap gap-2">
						<span className="rounded-md border border-[#ffc247]/25 bg-[#0b0e12]/90 px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wider text-[#ffc247] backdrop-blur">
							Continuous 12 s chainsaw excerpt
						</span>
						<span className="rounded-md border border-[#3bb9e8]/20 bg-[#0b0e12]/90 px-2.5 py-1.5 font-mono text-[10px] text-[#3bb9e8] backdrop-blur">
							Periodic core {formatDb(shownPeriodicDb)}
						</span>
					</div>
					<div className="pointer-events-none absolute bottom-4 right-4 rounded-md border border-white/10 bg-[#0b0e12]/90 px-2.5 py-2 font-mono text-[9px] uppercase tracking-wider text-white/40 backdrop-blur">
						wave motion slowed ×29
					</div>
				</div>

				<div className="rounded-2xl border border-white/10 bg-[#111820] p-5">
					<div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
						<div>
							<p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em] text-[#ffc247]">
								<AudioLines className="size-3.5" /> Recorded waveform
							</p>
							<p className="mt-1 text-xs text-white/40">
								12.0 s continuous run · analyzed window {analysis.segment.startSeconds.toFixed(3)}–{analysis.segment.endSeconds.toFixed(3)} s
							</p>
						</div>
						<audio
							controls
							preload="metadata"
							src="/audio/chainsaw-steady.ogg"
							className="h-9 w-full max-w-[320px] opacity-80"
						>
							<track
								kind="captions"
								src="/audio/chainsaw-steady.vtt"
								srcLang="en"
								label="Sound description"
							/>
						</audio>
					</div>
					<div className="mt-4 overflow-hidden rounded-xl border border-white/8 bg-[#070a0d] px-3">
						<WaveformPlot />
					</div>
					<div className="mt-4 flex h-16 items-end gap-px" aria-label="Frequency spectrum of extracted chainsaw window">
						{analysis.spectrum.map((bin) => (
							<span
								key={`spectrum-${bin.frequency}`}
								className="min-w-0 flex-1 rounded-t-sm bg-[#168bd2]/55"
								style={{ height: `${Math.max(4, 64 + bin.relativeDb)}px` }}
								title={`${Math.round(bin.frequency)} Hz · ${bin.relativeDb.toFixed(1)} dB`}
							/>
						))}
					</div>
					<div className="mt-2 flex justify-between font-mono text-[9px] uppercase tracking-wider text-white/25">
						<span>40 Hz</span><span>log-frequency spectrum</span><span>5 kHz</span>
					</div>
				</div>
			</div>

			{showControls ? <aside className="space-y-4">
				<section className="overflow-hidden rounded-2xl border border-[#3bb9e8]/20 bg-[#111820]">
					<div className="border-b border-white/8 p-5">
						<p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em] text-[#3bb9e8]">
							<Gauge className="size-3.5" /> Harmonic-model result
						</p>
						<div className="mt-4 grid grid-cols-2 gap-3">
							<div>
								<p className="text-[10px] uppercase tracking-wider text-white/35">Periodic core</p>
								<p className="mt-1 font-mono text-2xl font-semibold text-[#3bb9e8]">{formatDb(shownPeriodicDb)}</p>
							</div>
							<div>
								<p className="text-[10px] uppercase tracking-wider text-white/35">Whole signal</p>
								<p className="mt-1 font-mono text-2xl font-semibold text-[#ffc247]">{formatDb(shownFullDb)}</p>
							</div>
						</div>
					</div>
					<div className="p-5">
						<button
							type="button"
							onClick={() => setControlEnabled((enabled) => !enabled)}
							className="flex w-full items-center justify-center gap-2 rounded-lg border border-white/10 px-3 py-2.5 text-xs text-white/65 transition hover:border-white/20 hover:text-white"
						>
							{controlEnabled ? <VolumeX className="size-4 text-[#3bb9e8]" /> : <Volume2 className="size-4" />}
							{controlEnabled ? "Disable speaker array" : "Enable speaker array"}
						</button>
						<p className="mt-3 text-xs leading-5 text-white/40">
							The array solves a separate complex weight for every extracted harmonic, then plays all twelve at once.
						</p>
					</div>
				</section>

				<section className="rounded-2xl border border-[#168bd2]/18 bg-[#111820] p-5">
					<p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em] text-[#168bd2]">
						<Radio className="size-3.5" /> Extracted periodic core
					</p>
					<div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 text-xs">
						<div><p className="text-white/30">Fundamental</p><p className="mt-1 font-mono text-[#168bd2]">{analysis.segment.fundamentalHz.toFixed(1)} Hz</p></div>
						<div><p className="text-white/30">Cycle length</p><p className="mt-1 font-mono text-[#168bd2]">{analysis.segment.periodMilliseconds.toFixed(2)} ms</p></div>
						<div><p className="text-white/30">Repeat score</p><p className="mt-1 font-mono text-[#168bd2]">{analysis.segment.autocorrelation.toFixed(3)}</p></div>
						<div><p className="text-white/30">Phase-locked energy</p><p className="mt-1 font-mono text-[#168bd2]">{Math.round(PERIODIC_FRACTION * 100)}%</p></div>
					</div>
					<div className="mt-4 space-y-2">
						{COMPONENTS.slice(0, 4).map((component) => (
							<div key={`harmonic-${component.order}`} className="grid grid-cols-[42px_1fr_58px] items-center gap-2 font-mono text-[10px]">
								<span className="text-white/30">H{component.order}</span>
								<div className="h-1.5 overflow-hidden rounded-full bg-white/6"><div className="h-full rounded-full bg-[#2f6df6]" style={{ width: `${Math.max(3, component.powerShare * 100)}%` }} /></div>
								<span className="text-right text-white/50">{Math.round(component.frequency)} Hz</span>
							</div>
						))}
					</div>
				</section>

				<section className="rounded-2xl border border-[#ffc247]/18 bg-[#ffc247]/[0.035] p-5">
					<p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em] text-[#ffc247]">
						<TriangleAlert className="size-3.5" /> The honest limitation
					</p>
					<p className="mt-3 text-xs leading-5 text-white/45">
						The remaining {Math.round((1 - PERIODIC_FRACTION) * 100)}% is blade noise, airflow, reflections, and engine drift. This periodic controller leaves it untouched, so it sets a hard floor near {formatDb(FULL_SIGNAL_REDUCTION_DB)} for the whole recording.
					</p>
				</section>

				<section className="rounded-2xl border border-[#ff6a2a]/18 bg-[#ff6a2a]/[0.035] p-5">
					<p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em] text-[#ff6a2a]">
						<Waves className="size-3.5" /> What broadband means
					</p>
					<p className="mt-3 text-xs leading-5 text-white/45">
						A pure tone puts nearly all its energy at one frequency. This chainsaw spreads energy across a wide band—from the low engine rhythm through kilohertz blade and airflow noise. Every narrow frequency slice needs its own cancellation amplitude and phase.
					</p>
				</section>

				<section className="rounded-2xl border border-white/10 bg-[#111820] p-5">
					<p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em] text-white/45">
						<ShieldCheck className="size-3.5" /> Recording provenance
					</p>
					<p className="mt-3 text-xs leading-5 text-white/40">
						“Chainsaw 12” by ezwa · public domain. This player uses seconds {analysis.source.originalStartSeconds.toFixed(1)}–{analysis.source.originalEndSeconds.toFixed(1)} of the 87.3 s original.
					</p>
					<a href={analysis.source.url} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1.5 text-[11px] text-[#168bd2] hover:underline">
						Wikimedia Commons source <ExternalLink className="size-3" />
					</a>
				</section>
			</aside> : null}
		</section>
	);
}
