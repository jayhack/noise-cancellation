"use client";

import {
	Activity,
	CircleHelp,
	Gauge,
	Mic2,
	MousePointer2,
	Pause,
	Play,
	Radio,
	RotateCcw,
	Sparkles,
	Trash2,
	Volume2,
	Waves,
} from "lucide-react";
import {
	type PointerEvent as ReactPointerEvent,
	useCallback,
	useDeferredValue,
	useEffect,
	useMemo,
	useRef,
	useState,
} from "react";
import { ChainsawLab } from "@/components/sound-lab/chainsaw-lab";
import { ContinuousRingStory } from "@/components/sound-lab/continuous-ring-story";
import { HEstimationLab } from "@/components/sound-lab/h-estimation-lab";
import { MultiHumanLab } from "@/components/sound-lab/multi-human-lab";
import { ObstacleLab } from "@/components/sound-lab/obstacle-lab";

type Point = { id: number; x: number; y: number };
type Complex = { re: number; im: number };
type Tool = "move" | "sensor" | "speaker" | "delete";
type ControllerMode = "boundary" | "human";
type DragTarget =
	| { kind: "sensor" | "speaker"; id: number }
	| { kind: "observer" }
	| null;

const WORLD = { width: 12, height: 8 };
const SOURCE = { x: 4, y: 4 };
const SOUND_SPEED = 343;
const CONTROL_SAMPLES = 56;
const MAX_SPEAKER_STRENGTH = 4;

type ControllerResult = {
	mode: ControllerMode;
	pairings: number[];
	weights: Complex[];
	waveNumber: number;
	targetPoints: Point[];
	guardPoints: Point[];
	reductionDb: number;
	boundaryDb: number;
	bubbleDb: number;
	centerDb: number;
	worstGuardDb: number;
	speakerEffort: number;
};

const AURA = {
	background: [21, 20, 27] as const,
	foreground: "#edecee",
	purple: [162, 119, 255] as const,
	green: [97, 255, 202] as const,
	orange: "#ffca85",
	blue: "#82e2ff",
	red: "#ff6767",
};

function cAdd(a: Complex, b: Complex): Complex {
	return { re: a.re + b.re, im: a.im + b.im };
}

function cMul(a: Complex, b: Complex): Complex {
	return {
		re: a.re * b.re - a.im * b.im,
		im: a.re * b.im + a.im * b.re,
	};
}

function cScale(a: Complex, scale: number): Complex {
	return { re: a.re * scale, im: a.im * scale };
}

function magnitude(value: Complex) {
	return Math.hypot(value.re, value.im);
}

function distance(a: Pick<Point, "x" | "y">, b: Pick<Point, "x" | "y">) {
	return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Far-field form of the outgoing 2D Helmholtz Green function. */
function green(
	from: Pick<Point, "x" | "y">,
	to: Pick<Point, "x" | "y">,
	waveNumber: number,
): Complex {
	const radius = Math.max(0.09, distance(from, to));
	const amplitude = 1 / Math.sqrt(radius);
	const phase = -waveNumber * radius;
	return { re: amplitude * Math.cos(phase), im: amplitude * Math.sin(phase) };
}

function buildRing(count: number, radius = 1.55, angleOffset = 0) {
	return Array.from({ length: count }, (_, index) => {
		const angle = (index / count) * Math.PI * 2 + angleOffset;
		return {
			id: index + 1,
			x: SOURCE.x + Math.cos(angle) * radius,
			y: SOURCE.y + Math.sin(angle) * radius,
		};
	});
}

function controlPoints(radius: number) {
	return Array.from({ length: CONTROL_SAMPLES }, (_, index) => {
		const angle = (index / CONTROL_SAMPLES) * Math.PI * 2;
		return {
			id: index,
			x: SOURCE.x + Math.cos(angle) * radius,
			y: SOURCE.y + Math.sin(angle) * radius,
		};
	});
}

function pointsOnDisk(center: Point, radius: number) {
	const points: Array<Point & { weight: number }> = [
		{ id: 0, x: center.x, y: center.y, weight: 2.4 },
	];
	const rings = [
		{ radius: radius * 0.35, count: 8, weight: 1.6 },
		{ radius: radius * 0.7, count: 12, weight: 1.15 },
		{ radius, count: 16, weight: 0.85 },
	];
	for (const ring of rings) {
		for (let index = 0; index < ring.count; index += 1) {
			const angle = (index / ring.count) * Math.PI * 2;
			points.push({
				id: points.length,
				x: center.x + Math.cos(angle) * ring.radius,
				y: center.y + Math.sin(angle) * ring.radius,
				weight: ring.weight,
			});
		}
	}
	return points;
}

function pointsOnGuardRing(center: Point, bubbleRadius: number) {
	const radius = Math.max(0.85, bubbleRadius * 1.9);
	return Array.from({ length: 24 }, (_, index) => {
		const angle = (index / 24) * Math.PI * 2;
		return {
			id: index,
			x: center.x + Math.cos(angle) * radius,
			y: center.y + Math.sin(angle) * radius,
		};
	});
}

function pairSpeakersToSensors(speakers: Point[], sensors: Point[]) {
	return speakers.map((speaker) => {
		if (sensors.length === 0) return -1;
		let nearest = 0;
		for (let index = 1; index < sensors.length; index += 1) {
			if (distance(speaker, sensors[index]) < distance(speaker, sensors[nearest])) {
				nearest = index;
			}
		}
		return nearest;
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
		if (Math.abs(rows[pivot][column]) < 1e-9) return Array(size).fill(0);
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

function calculateBoundaryController({
	sensors,
	speakers,
	frequency,
	delayMs,
	boundaryRadius,
}: {
	sensors: Point[];
	speakers: Point[];
	frequency: number;
	delayMs: number;
	boundaryRadius: number;
}): ControllerResult {
	const omega = Math.PI * 2 * frequency;
	const waveNumber = omega / SOUND_SPEED;
	const boundary = controlPoints(boundaryRadius);
	const sourceField = boundary.map((point) => green(SOURCE, point, waveNumber));
	const delay: Complex = {
		re: Math.cos(-omega * (delayMs / 1000)),
		im: Math.sin(-omega * (delayMs / 1000)),
	};

	const pairings = pairSpeakersToSensors(speakers, sensors);

	// Each speaker replays its nearest sensor signal after θ, with an inverted
	// polarity. The optimizer may choose only a real gain; it cannot secretly
	// remove the physical delay with an arbitrary single-frequency phase shift.
	const drives = speakers.map((_, index) => {
		const sensorIndex = pairings[index];
		if (sensorIndex < 0) return { re: 0, im: 0 };
		return cScale(
			cMul(green(SOURCE, sensors[sensorIndex], waveNumber), delay),
			-1,
		);
	});

	const columns = speakers.map((speaker, speakerIndex) =>
		boundary.map((point) =>
			cMul(drives[speakerIndex], green(speaker, point, waveNumber)),
		),
	);

	const size = speakers.length;
	const normal = Array.from({ length: size }, () => Array(size).fill(0));
	const target = Array(size).fill(0);

	for (let left = 0; left < size; left += 1) {
		for (let right = 0; right < size; right += 1) {
			for (let sample = 0; sample < boundary.length; sample += 1) {
				normal[left][right] +=
					columns[left][sample].re * columns[right][sample].re +
					columns[left][sample].im * columns[right][sample].im;
			}
		}
		for (let sample = 0; sample < boundary.length; sample += 1) {
			target[left] -=
				columns[left][sample].re * sourceField[sample].re +
				columns[left][sample].im * sourceField[sample].im;
		}
	}

	const diagonalMean =
		size > 0
			? normal.reduce((sum, row, index) => sum + row[index], 0) / size
			: 0;
	for (let index = 0; index < size; index += 1) {
		normal[index][index] += diagonalMean * 0.006 + 1e-6;
	}

	const gains = solveLinearSystem(normal, target).map((gain) =>
		Math.max(-8, Math.min(8, gain)),
	);
	let controlledEnergy = 0;
	let sourceEnergy = 0;
	for (let sample = 0; sample < boundary.length; sample += 1) {
		let total = sourceField[sample];
		for (let speaker = 0; speaker < size; speaker += 1) {
			total = cAdd(total, cScale(columns[speaker][sample], gains[speaker]));
		}
		controlledEnergy += magnitude(total) ** 2;
		sourceEnergy += magnitude(sourceField[sample]) ** 2;
	}

	const boundaryDb =
		10 * Math.log10(Math.max(1e-12, controlledEnergy / sourceEnergy));
	const weights = speakers.map((_, index) => cScale(drives[index], gains[index]));

	return {
		mode: "boundary",
		pairings,
		weights,
		waveNumber,
		targetPoints: boundary,
		guardPoints: [],
		reductionDb: boundaryDb,
		boundaryDb,
		bubbleDb: boundaryDb,
		centerDb: boundaryDb,
		worstGuardDb: 0,
		speakerEffort: weights.reduce((sum, weight) => sum + magnitude(weight) ** 2, 0),
	};
}

function pressureWithWeights(
	point: Pick<Point, "x" | "y">,
	waveNumber: number,
	speakers: Point[],
	weights: Complex[],
) {
	let pressure = green(SOURCE, point, waveNumber);
	for (let index = 0; index < speakers.length; index += 1) {
		pressure = cAdd(
			pressure,
			cMul(weights[index] ?? { re: 0, im: 0 }, green(speakers[index], point, waveNumber)),
		);
	}
	return pressure;
}

function calculateHumanController({
	sensors,
	speakers,
	frequency,
	delayMs,
	human,
	bubbleRadius,
}: {
	sensors: Point[];
	speakers: Point[];
	frequency: number;
	delayMs: number;
	human: Point;
	bubbleRadius: number;
}): ControllerResult {
	const omega = Math.PI * 2 * frequency;
	const waveNumber = omega / SOUND_SPEED;
	const targets = pointsOnDisk(human, bubbleRadius);
	const guards = pointsOnGuardRing(human, bubbleRadius);
	const pairings = pairSpeakersToSensors(speakers, sensors);
	const delay: Complex = {
		re: Math.cos(-omega * (delayMs / 1000)),
		im: Math.sin(-omega * (delayMs / 1000)),
	};
	const references = pairings.map((sensorIndex) =>
		sensorIndex < 0
			? { re: 0, im: 0 }
			: cMul(green(SOURCE, sensors[sensorIndex], waveNumber), delay),
	);

	const unknowns = speakers.length * 2;
	const normal = Array.from({ length: unknowns }, () => Array(unknowns).fill(0));
	const target = Array(unknowns).fill(0);

	const addConstraint = (
		point: Point,
		desiredSpeakerField: Complex,
		constraintWeight: number,
	) => {
		const realRow = Array(unknowns).fill(0);
		const imaginaryRow = Array(unknowns).fill(0);
		for (let index = 0; index < speakers.length; index += 1) {
			const column = cMul(
				references[index],
				green(speakers[index], point, waveNumber),
			);
			realRow[index * 2] = column.re;
			realRow[index * 2 + 1] = -column.im;
			imaginaryRow[index * 2] = column.im;
			imaginaryRow[index * 2 + 1] = column.re;
		}

		for (let left = 0; left < unknowns; left += 1) {
			target[left] +=
				constraintWeight *
				(realRow[left] * desiredSpeakerField.re +
					imaginaryRow[left] * desiredSpeakerField.im);
			for (let right = 0; right < unknowns; right += 1) {
				normal[left][right] +=
					constraintWeight *
					(realRow[left] * realRow[right] +
						imaginaryRow[left] * imaginaryRow[right]);
			}
		}
	};

	for (const point of targets) {
		addConstraint(point, cScale(green(SOURCE, point, waveNumber), -1), point.weight);
	}
	// The weak guard constraints ask the speaker contribution to approach zero
	// outside the bubble. They do not hide hot spots; they merely discourage the
	// optimizer from buying a quiet center with unlimited nearby amplification.
	for (const point of guards) {
		addConstraint(point, { re: 0, im: 0 }, 0.075);
	}

	const diagonalMean =
		unknowns > 0
			? normal.reduce((sum, row, index) => sum + row[index], 0) / unknowns
			: 0;
	for (let index = 0; index < unknowns; index += 1) {
		normal[index][index] += diagonalMean * 0.003 + 1e-6;
	}

	const solution = solveLinearSystem(normal, target);
	const filters = speakers.map((_, index) => ({
		re: solution[index * 2] ?? 0,
		im: solution[index * 2 + 1] ?? 0,
	}));
	let weights = filters.map((filter, index) => cMul(references[index], filter));
	const strongest = weights.reduce((maximum, weight) => Math.max(maximum, magnitude(weight)), 0);
	if (strongest > MAX_SPEAKER_STRENGTH) {
		weights = weights.map((weight) => cScale(weight, MAX_SPEAKER_STRENGTH / strongest));
	}

	let sourceEnergy = 0;
	let controlledEnergy = 0;
	for (const point of targets) {
		const source = green(SOURCE, point, waveNumber);
		const controlled = pressureWithWeights(point, waveNumber, speakers, weights);
		sourceEnergy += point.weight * magnitude(source) ** 2;
		controlledEnergy += point.weight * magnitude(controlled) ** 2;
	}
	const bubbleDb = 10 * Math.log10(Math.max(1e-12, controlledEnergy / sourceEnergy));
	const centerSource = green(SOURCE, human, waveNumber);
	const centerControlled = pressureWithWeights(human, waveNumber, speakers, weights);
	const centerDb =
		20 * Math.log10(Math.max(1e-7, magnitude(centerControlled) / magnitude(centerSource)));
	let worstGuardDb = -Infinity;
	for (const point of guards) {
		const source = green(SOURCE, point, waveNumber);
		const controlled = pressureWithWeights(point, waveNumber, speakers, weights);
		worstGuardDb = Math.max(
			worstGuardDb,
			20 * Math.log10(Math.max(1e-7, magnitude(controlled) / magnitude(source))),
		);
	}

	return {
		mode: "human",
		pairings,
		weights,
		waveNumber,
		targetPoints: targets,
		guardPoints: guards,
		reductionDb: bubbleDb,
		boundaryDb: bubbleDb,
		bubbleDb,
		centerDb,
		worstGuardDb: Number.isFinite(worstGuardDb) ? worstGuardDb : 0,
		speakerEffort: weights.reduce((sum, weight) => sum + magnitude(weight) ** 2, 0),
	};
}

function rankHumanFrequencies({
	sensors,
	speakers,
	delayMs,
	human,
	bubbleRadius,
}: {
	sensors: Point[];
	speakers: Point[];
	delayMs: number;
	human: Point;
	bubbleRadius: number;
}) {
	const candidates = [];
	for (let frequency = 100; frequency <= 1000; frequency += 25) {
		const result = calculateHumanController({
			sensors,
			speakers,
			frequency,
			delayMs,
			human,
			bubbleRadius,
		});
		// Prefer a deep average bubble, but reject solutions that purchase it
		// with a severe nearby hot spot or excessive speaker effort.
		const score =
			result.bubbleDb +
			Math.max(0, result.worstGuardDb - 1) * 0.75 +
			Math.log10(1 + result.speakerEffort) * 1.5;
		candidates.push({
			frequency,
			bubbleDb: result.bubbleDb,
			worstGuardDb: result.worstGuardDb,
			speakerEffort: result.speakerEffort,
			score,
		});
	}
	return candidates.sort((left, right) => left.score - right.score).slice(0, 3);
}

function fieldAt(
	point: Pick<Point, "x" | "y">,
	controller: ControllerResult,
	speakers: Point[],
	controlEnabled: boolean,
) {
	if (!controlEnabled) return green(SOURCE, point, controller.waveNumber);
	return pressureWithWeights(point, controller.waveNumber, speakers, controller.weights);
}

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
	if (!Number.isFinite(value)) return "—";
	return `${value > 0 ? "+" : ""}${value.toFixed(1)} dB`;
}

function Waveform({ ratio }: { ratio: number }) {
	const controlledAmplitude = Math.min(16, Math.max(1, ratio * 16));
	const points = Array.from({ length: 90 }, (_, index) => {
		const x = (index / 89) * 240;
		const y = 30 + Math.sin((index / 89) * Math.PI * 8) * controlledAmplitude;
		return `${x.toFixed(1)},${y.toFixed(1)}`;
	}).join(" ");
	const sourcePoints = Array.from({ length: 90 }, (_, index) => {
		const x = (index / 89) * 240;
		const y = 30 + Math.sin((index / 89) * Math.PI * 8) * 16;
		return `${x.toFixed(1)},${y.toFixed(1)}`;
	}).join(" ");

	return (
		<svg viewBox="0 0 240 60" className="h-14 w-full" aria-hidden>
			<line x1="0" y1="30" x2="240" y2="30" stroke="rgba(255,255,255,.08)" />
			<polyline
				points={sourcePoints}
				fill="none"
				stroke="rgba(255,202,133,.28)"
				strokeWidth="1.2"
			/>
			<polyline
				points={points}
				fill="none"
				stroke="#61ffca"
				strokeWidth="1.8"
			/>
		</svg>
	);
}

function useSectionActive(ref: { current: HTMLElement | null }) {
	const [active, setActive] = useState(false);
	useEffect(() => {
		const element = ref.current;
		if (!element || typeof IntersectionObserver === "undefined") {
			setActive(true);
			return;
		}
		const observer = new IntersectionObserver(
			([entry]) => setActive(entry.isIntersecting),
			{ rootMargin: "240px 0px", threshold: 0.01 },
		);
		observer.observe(element);
		return () => observer.disconnect();
	}, [ref]);
	return active;
}

function ExperimentHeading({
	number,
	eyebrow,
	title,
	body,
	color,
}: {
	number: string;
	eyebrow: string;
	title: string;
	body: string;
	color: string;
}) {
	return (
		<div className="mx-auto max-w-[1500px] px-5 pb-1 pt-14 sm:px-8 sm:pt-20">
			<div className="flex items-center gap-3 font-mono text-[10px] uppercase tracking-[0.18em]" style={{ color }}>
				<span className="grid size-7 place-items-center rounded-full border" style={{ borderColor: `${color}55` }}>
					{number}
				</span>
				{eyebrow}
			</div>
			<h2 className="mt-4 max-w-4xl text-3xl font-semibold tracking-tight sm:text-4xl">
				{title}
			</h2>
			<p className="mt-3 max-w-3xl text-sm leading-6 text-white/48 sm:text-base">
				{body}
			</p>
		</div>
	);
}

export function WaveLab() {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const idealSectionRef = useRef<HTMLDivElement>(null);
	const discreteSectionRef = useRef<HTMLDivElement>(null);
	const multipleSectionRef = useRef<HTMLDivElement>(null);
	const obstacleSectionRef = useRef<HTMLDivElement>(null);
	const estimationSectionRef = useRef<HTMLDivElement>(null);
	const recoverySectionRef = useRef<HTMLDivElement>(null);
	const chainsawSectionRef = useRef<HTMLDivElement>(null);
	const frameRef = useRef<number>(0);
	const lastFrameRef = useRef(0);
	const timeRef = useRef(0);
	const dragRef = useRef<DragTarget>(null);
	const nextIdRef = useRef(100);
	const orbitFrameRef = useRef(0);
	const orbitLastFrameRef = useRef(0);
	const orbitPhaseRef = useRef(-1.128);
	const idealActive = useSectionActive(idealSectionRef);
	const discreteActive = useSectionActive(discreteSectionRef);
	const multipleActive = useSectionActive(multipleSectionRef);
	const obstacleActive = useSectionActive(obstacleSectionRef);
	const estimationActive = useSectionActive(estimationSectionRef);
	const recoveryActive = useSectionActive(recoverySectionRef);
	const chainsawActive = useSectionActive(chainsawSectionRef);
	const [running, setRunning] = useState(true);
	const [storyResetKey, setStoryResetKey] = useState(0);
	const [multiResetKey, setMultiResetKey] = useState(0);
	const [obstacleResetKey, setObstacleResetKey] = useState(0);
	const [estimationResetKey, setEstimationResetKey] = useState(0);
	const [recoveryResetKey, setRecoveryResetKey] = useState(0);
	const [chainsawResetKey, setChainsawResetKey] = useState(0);
	const [autoTrack, setAutoTrack] = useState(true);
	const [frequency, setFrequency] = useState(440);
	const [delay, setDelay] = useState(0);
	const [boundaryRadius, setBoundaryRadius] = useState(3.15);
	const [bubbleRadius, setBubbleRadius] = useState(0.45);
	const [controllerMode, setControllerMode] = useState<ControllerMode>("human");
	const [controlEnabled, setControlEnabled] = useState(true);
	const [tool, setTool] = useState<Tool>("move");
	const [sensors, setSensors] = useState<Point[]>(() => buildRing(8, 1.55, Math.PI / 8));
	const [speakers, setSpeakers] = useState<Point[]>(() => buildRing(8));
	const [observer, setObserver] = useState<Point>({ id: 1, x: 9.55, y: 3.15 });
	const deferredObserver = useDeferredValue(observer);

	const controller = useMemo(
		() =>
			controllerMode === "human"
				? calculateHumanController({
						sensors,
						speakers,
						frequency,
						delayMs: delay,
						human: observer,
						bubbleRadius,
					})
				: calculateBoundaryController({
						sensors,
						speakers,
						frequency,
						delayMs: delay,
						boundaryRadius,
					}),
		[
			controllerMode,
			sensors,
			speakers,
			frequency,
			delay,
			observer,
			bubbleRadius,
			boundaryRadius,
		],
	);

	const sourceAtObserver = green(SOURCE, observer, controller.waveNumber);
	const totalAtObserver = fieldAt(observer, controller, speakers, controlEnabled);
	const observerRatio = controlEnabled
		? magnitude(totalAtObserver) / magnitude(sourceAtObserver)
		: 1;
	const observerDb = 20 * Math.log10(Math.max(1e-7, observerRatio));
	const rankedFrequencies = useMemo(
		() =>
			controllerMode === "human"
				? rankHumanFrequencies({
						sensors,
						speakers,
						delayMs: delay,
						human: deferredObserver,
						bubbleRadius,
					})
				: [],
		[
			controllerMode,
			sensors,
			speakers,
			delay,
			deferredObserver,
			bubbleRadius,
		],
	);

	const draw = useCallback(
		(timestamp: number) => {
			if (!discreteActive) return;
			if (timestamp - lastFrameRef.current < 32) return;
			const delta = Math.min(0.05, (timestamp - lastFrameRef.current) / 1000);
			lastFrameRef.current = timestamp;
			if (running) timeRef.current += delta;

			const canvas = canvasRef.current;
			if (!canvas) return;
			const { width, height, dpr } = resizeCanvas(canvas);
			const context = canvas.getContext("2d");
			if (!context) return;
			const transform = getTransform(width, height);
			const toCanvas = (point: Pick<Point, "x" | "y">) => ({
				x: transform.offsetX + point.x * transform.scale,
				y: transform.offsetY + point.y * transform.scale,
			});

			context.fillStyle = "#100f15";
			context.fillRect(0, 0, width, height);
			const step = Math.max(7, Math.round(6 * dpr));
			const visualPhase = timeRef.current * Math.PI * 1.35;
			const cosTime = Math.cos(visualPhase);
			const sinTime = Math.sin(visualPhase);

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
					const phasor = fieldAt(point, controller, speakers, controlEnabled);
					const instantaneous = phasor.re * cosTime - phasor.im * sinTime;
					const signedStrength = Math.tanh(instantaneous * 0.72);
					const mix = Math.abs(signedStrength) * 0.76;
					const target = signedStrength >= 0 ? AURA.purple : AURA.green;
					const red = Math.round(
						AURA.background[0] + (target[0] - AURA.background[0]) * mix,
					);
					const greenValue = Math.round(
						AURA.background[1] + (target[1] - AURA.background[1]) * mix,
					);
					const blue = Math.round(
						AURA.background[2] + (target[2] - AURA.background[2]) * mix,
					);
					context.fillStyle = `rgb(${red} ${greenValue} ${blue})`;
					context.fillRect(pixelX, pixelY, step + 1, step + 1);
				}
			}

			context.lineWidth = dpr;
			context.strokeStyle = "rgba(237,236,238,.09)";
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

			const center = toCanvas(SOURCE);
			if (controller.mode === "boundary") {
				context.save();
				context.setLineDash([6 * dpr, 7 * dpr]);
				context.strokeStyle = "rgba(246,148,255,.55)";
				context.lineWidth = 1.25 * dpr;
				context.beginPath();
				context.arc(center.x, center.y, boundaryRadius * transform.scale, 0, Math.PI * 2);
				context.stroke();
				context.restore();
			} else {
				const humanPoint = toCanvas(observer);
				context.save();
				context.fillStyle = "rgba(97,255,202,.055)";
				context.strokeStyle = "rgba(97,255,202,.8)";
				context.lineWidth = 1.5 * dpr;
				context.beginPath();
				context.arc(humanPoint.x, humanPoint.y, bubbleRadius * transform.scale, 0, Math.PI * 2);
				context.fill();
				context.stroke();

				for (const targetPoint of controller.targetPoints) {
					const sample = toCanvas(targetPoint);
					context.fillStyle = "rgba(97,255,202,.38)";
					context.beginPath();
					context.arc(sample.x, sample.y, 1.25 * dpr, 0, Math.PI * 2);
					context.fill();
				}

				const guardRadius = controller.guardPoints[0]
					? distance(observer, controller.guardPoints[0]) * transform.scale
					: 0;
				if (guardRadius > 0) {
					context.setLineDash([3 * dpr, 8 * dpr]);
					context.strokeStyle = "rgba(255,103,103,.36)";
					context.lineWidth = dpr;
					context.beginPath();
					context.arc(humanPoint.x, humanPoint.y, guardRadius, 0, Math.PI * 2);
					context.stroke();
				}
				context.restore();
			}

			for (let index = 0; index < speakers.length; index += 1) {
				const sensorIndex = controller.pairings[index];
				if (sensorIndex < 0) continue;
				const start = toCanvas(sensors[sensorIndex]);
				const end = toCanvas(speakers[index]);
				context.strokeStyle = "rgba(130,226,255,.2)";
				context.beginPath();
				context.moveTo(start.x, start.y);
				context.lineTo(end.x, end.y);
				context.stroke();
			}

			for (const sensor of sensors) {
				const point = toCanvas(sensor);
				context.save();
				context.translate(point.x, point.y);
				context.rotate(Math.PI / 4);
				context.fillStyle = "#15141b";
				context.strokeStyle = AURA.blue;
				context.lineWidth = 1.6 * dpr;
				context.fillRect(-5 * dpr, -5 * dpr, 10 * dpr, 10 * dpr);
				context.strokeRect(-5 * dpr, -5 * dpr, 10 * dpr, 10 * dpr);
				context.restore();
			}

			for (const speaker of speakers) {
				const point = toCanvas(speaker);
				context.fillStyle = "#15141b";
				context.strokeStyle = controlEnabled ? "#a277ff" : "rgba(162,119,255,.4)";
				context.lineWidth = 1.8 * dpr;
				context.beginPath();
				context.arc(point.x, point.y, 8 * dpr, 0, Math.PI * 2);
				context.fill();
				context.stroke();
				context.beginPath();
				context.arc(point.x, point.y, 3 * dpr, -0.8, 0.8);
				context.stroke();
			}

			context.fillStyle = AURA.orange;
			context.beginPath();
			context.arc(center.x, center.y, 7 * dpr, 0, Math.PI * 2);
			context.fill();
			context.strokeStyle = "rgba(255,202,133,.35)";
			context.lineWidth = 8 * dpr;
			context.stroke();

			const observerPoint = toCanvas(observer);
			context.fillStyle = "#15141b";
			context.strokeStyle = "#61ffca";
			context.lineWidth = 2 * dpr;
			context.beginPath();
			context.arc(observerPoint.x, observerPoint.y, 11 * dpr, 0, Math.PI * 2);
			context.fill();
			context.stroke();
			context.beginPath();
			context.moveTo(observerPoint.x - 15 * dpr, observerPoint.y);
			context.lineTo(observerPoint.x + 15 * dpr, observerPoint.y);
			context.moveTo(observerPoint.x, observerPoint.y - 15 * dpr);
			context.lineTo(observerPoint.x, observerPoint.y + 15 * dpr);
			context.stroke();

			const drawCallout = ({
				anchor,
				text,
				color,
				offset,
				verticalOffset,
			}: {
				anchor: { x: number; y: number };
				text: string;
				color: string;
				offset: number;
				verticalOffset: number;
			}) => {
				const fontSize = 17 * dpr;
				const horizontalPadding = 10 * dpr;
				const height = 34 * dpr;
				context.font = `600 ${fontSize}px ui-monospace, monospace`;
				context.textBaseline = "middle";
				const textWidth = context.measureText(text).width;
				const plateWidth = textWidth + horizontalPadding * 2;
				const preferredX = anchor.x + offset * dpr;
				const x = Math.max(
					12 * dpr,
					Math.min(preferredX, width - plateWidth - 12 * dpr),
				);
				const labelCenterY = anchor.y + verticalOffset * dpr;
				const y = labelCenterY - height / 2;

				context.strokeStyle = color;
				context.globalAlpha = 0.52;
				context.lineWidth = 1.5 * dpr;
				context.beginPath();
				context.moveTo(anchor.x + 12 * dpr, anchor.y);
				context.lineTo(x - 5 * dpr, labelCenterY);
				context.stroke();

				context.globalAlpha = 1;
				context.fillStyle = "rgba(16,15,21,.9)";
				context.strokeStyle = `${color}66`;
				context.lineWidth = dpr;
				context.beginPath();
				context.roundRect(x, y, plateWidth, height, 8 * dpr);
				context.fill();
				context.stroke();

				context.fillStyle = color;
				context.fillText(text, x + horizontalPadding, labelCenterY + dpr);
			};

			drawCallout({
				anchor: center,
				text: "SOURCE",
				color: "#ffca85",
				offset: 30,
				verticalOffset: 0,
			});
			const observerLabel = `${controller.mode === "human" ? "TRACKED HUMAN" : "BYSTANDER"}  ${formatDb(observerDb)}`;
			drawCallout({
				anchor: observerPoint,
				text: observerLabel,
				color: "#61ffca",
				offset: 42,
				verticalOffset: -52,
			});
			context.textBaseline = "alphabetic";
		},
		[
			boundaryRadius,
			bubbleRadius,
			controlEnabled,
			controller,
			discreteActive,
			observer,
			observerDb,
			running,
			sensors,
			speakers,
		],
	);

	useEffect(() => {
		if (!discreteActive) return;
		const animate = (timestamp: number) => {
			draw(timestamp);
			frameRef.current = requestAnimationFrame(animate);
		};
		frameRef.current = requestAnimationFrame(animate);
		return () => cancelAnimationFrame(frameRef.current);
	}, [discreteActive, draw]);

	useEffect(() => {
		if (
			!discreteActive ||
			controllerMode !== "human" ||
			!autoTrack ||
			!running
		) {
			orbitLastFrameRef.current = 0;
			return;
		}

		const center = { x: 9.1, y: 4.4 };
		const radii = { x: 1.05, y: 1.38 };
		const metersPerSecond = 0.42;
		const animateOrbit = (timestamp: number) => {
			orbitFrameRef.current = requestAnimationFrame(animateOrbit);
			if (orbitLastFrameRef.current === 0) {
				orbitLastFrameRef.current = timestamp;
				return;
			}
			if (timestamp - orbitLastFrameRef.current < 64) return;

			const delta = Math.min(0.12, (timestamp - orbitLastFrameRef.current) / 1000);
			orbitLastFrameRef.current = timestamp;
			const phase = orbitPhaseRef.current;
			const tangentLength = Math.hypot(
				radii.x * Math.sin(phase),
				radii.y * Math.cos(phase),
			);
			orbitPhaseRef.current =
				(phase + (metersPerSecond * delta) / Math.max(0.01, tangentLength)) %
				(Math.PI * 2);
			setObserver((point) => ({
				...point,
				x: center.x + radii.x * Math.cos(orbitPhaseRef.current),
				y: center.y + radii.y * Math.sin(orbitPhaseRef.current),
			}));
		};

		orbitFrameRef.current = requestAnimationFrame(animateOrbit);
		return () => cancelAnimationFrame(orbitFrameRef.current);
	}, [autoTrack, controllerMode, discreteActive, running]);

	const reset = useCallback((count = 8) => {
		setSensors(buildRing(count, 1.55, Math.PI / count));
		setSpeakers(buildRing(count));
		setObserver({ id: 1, x: 9.55, y: 3.15 });
		setFrequency(440);
		setDelay(0);
		setBoundaryRadius(3.15);
		setBubbleRadius(0.45);
		setControllerMode("human");
		setControlEnabled(true);
		setAutoTrack(true);
		setTool("move");
		timeRef.current = 0;
		orbitPhaseRef.current = -1.128;
		orbitLastFrameRef.current = 0;
	}, []);

	const pointerToWorld = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
		const canvas = canvasRef.current;
		if (!canvas) return null;
		const rect = canvas.getBoundingClientRect();
		const transform = getTransform(rect.width, rect.height);
		return {
			x: Math.max(
				0.1,
				Math.min(WORLD.width - 0.1, (event.clientX - rect.left - transform.offsetX) / transform.scale),
			),
			y: Math.max(
				0.1,
				Math.min(WORLD.height - 0.1, (event.clientY - rect.top - transform.offsetY) / transform.scale),
			),
		};
	}, []);

	const findTarget = useCallback(
		(point: Pick<Point, "x" | "y">): DragTarget => {
			if (distance(point, observer) < 0.35) return { kind: "observer" };
			const speaker = speakers.find((item) => distance(point, item) < 0.3);
			if (speaker) return { kind: "speaker", id: speaker.id };
			const sensor = sensors.find((item) => distance(point, item) < 0.3);
			if (sensor) return { kind: "sensor", id: sensor.id };
			return null;
		},
		[observer, sensors, speakers],
	);

	const handlePointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
		const point = pointerToWorld(event);
		if (!point) return;
		const target = findTarget(point);

		if (tool === "delete") {
			if (target?.kind === "speaker") {
				setSpeakers((items) => items.filter((item) => item.id !== target.id));
			} else if (target?.kind === "sensor") {
				setSensors((items) => items.filter((item) => item.id !== target.id));
			}
			return;
		}

		if (tool === "sensor" || tool === "speaker") {
			const newPoint = { ...point, id: nextIdRef.current++ };
			if (tool === "sensor") setSensors((items) => [...items, newPoint]);
			else setSpeakers((items) => [...items, newPoint]);
			setTool("move");
			return;
		}

		if (target) {
			if (target.kind === "observer") setAutoTrack(false);
			dragRef.current = target;
			event.currentTarget.setPointerCapture(event.pointerId);
		}
	};

	const handlePointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
		const point = pointerToWorld(event);
		const target = dragRef.current;
		if (!point || !target) return;
		if (target.kind === "observer") setObserver((item) => ({ ...item, ...point }));
		if (target.kind === "speaker") {
			setSpeakers((items) =>
				items.map((item) => (item.id === target.id ? { ...item, ...point } : item)),
			);
		}
		if (target.kind === "sensor") {
			setSensors((items) =>
				items.map((item) => (item.id === target.id ? { ...item, ...point } : item)),
			);
		}
	};

	const handlePointerUp = () => {
		dragRef.current = null;
	};

	const toolOptions: Array<{ id: Tool; label: string; icon: typeof MousePointer2 }> = [
		{ id: "move", label: "Move", icon: MousePointer2 },
		{ id: "sensor", label: "Add sensor", icon: Mic2 },
		{ id: "speaker", label: "Add speaker", icon: Volume2 },
		{ id: "delete", label: "Delete", icon: Trash2 },
	];

	return (
		<main className="min-h-screen bg-[#15141b] text-[#edecee]">
			<header className="border-b border-white/10 px-5 py-4 sm:px-8">
				<div className="mx-auto flex max-w-[1500px] items-center justify-between gap-4">
					<div className="flex items-center gap-3">
						<div className="grid size-9 place-items-center rounded-lg border border-[#a277ff]/40 bg-[#a277ff]/10">
							<Waves className="size-[18px] text-[#a277ff]" />
						</div>
						<div>
							<h1 className="text-sm font-semibold tracking-[-0.01em] sm:text-base">
								Active sound control lab
							</h1>
							<p className="font-mono text-[10px] uppercase tracking-[0.15em] text-white/40">
								Seven experiments · scroll to compare
							</p>
						</div>
					</div>
					<div className="flex items-center gap-2">
						<button
							type="button"
							onClick={() => {
								reset();
								setStoryResetKey((value) => value + 1);
								setMultiResetKey((value) => value + 1);
								setObstacleResetKey((value) => value + 1);
								setEstimationResetKey((value) => value + 1);
								setRecoveryResetKey((value) => value + 1);
								setChainsawResetKey((value) => value + 1);
							}}
							className="grid size-9 place-items-center rounded-lg border border-white/10 text-white/55 transition hover:border-white/20 hover:text-white"
							aria-label="Reset simulation"
						>
							<RotateCcw className="size-4" />
						</button>
						<button
							type="button"
							onClick={() => setRunning((value) => !value)}
							className="inline-flex h-9 items-center gap-2 rounded-lg bg-[#edecee] px-3.5 text-xs font-medium text-[#15141b] transition hover:bg-white"
						>
							{running ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
							{running ? "Pause" : "Run"}
						</button>
					</div>
				</div>
			</header>

			<nav className="sticky top-0 z-30 border-b border-white/10 bg-[#15141b]/92 px-4 py-2 backdrop-blur-xl sm:px-6" aria-label="Experiment sequence">
				<div className="mx-auto grid max-w-[1500px] grid-cols-4 gap-1 lg:grid-cols-7">
					{[
						["#experiment-1", "01", "Ideal shell"],
						["#experiment-2", "02", "One person"],
						["#experiment-3", "03", "Three people"],
						["#experiment-4", "04", "Wrong H"],
						["#experiment-5", "05", "Estimate H"],
						["#experiment-6", "06", "Recover"],
						["#experiment-7", "07", "Real audio"],
					].map(([href, number, label]) => (
						<a
							key={href}
							href={href}
							className="rounded-lg border border-transparent px-3 py-2 font-mono text-[10px] uppercase tracking-[0.12em] text-white/40 transition hover:border-white/10 hover:bg-white/[0.035] hover:text-white/75"
						>
							<span className="mr-2 text-[#82e2ff]">{number}</span>{label}
						</a>
					))}
				</div>
			</nav>

			<div id="experiment-1" ref={idealSectionRef} className="scroll-mt-14 border-b border-white/10">
				<ExperimentHeading
					number="1"
					eyebrow="Ideal mathematical limit"
					title="Can a perfect shell erase an outgoing wave?"
					body="Start with the exception: a continuous, zero-delay ring around a centered pure tone can exactly cancel the exterior field."
					color="#82e2ff"
				/>
				<ContinuousRingStory key={storyResetKey} running={running} active={idealActive} />
			</div>

			<div id="experiment-2" ref={discreteSectionRef} className="scroll-mt-14 border-b border-white/10">
				<ExperimentHeading
					number="2"
					eyebrow="Finite array · open field"
					title="Can discrete speakers follow one moving person?"
					body="Replace the perfect shell with a finite sensor-speaker ring and optimize a small quiet bubble around a tracked listener."
					color="#61ffca"
				/>
			<section className="mx-auto grid max-w-[1500px] gap-4 p-4 sm:p-6 lg:grid-cols-[minmax(0,1fr)_340px]">
				<div className="space-y-3">
					<div className="relative aspect-[3/2] min-h-[420px] max-h-[720px] overflow-hidden rounded-2xl border border-white/10 bg-[#100f15] shadow-2xl shadow-black/30">
						<canvas
							ref={canvasRef}
							data-testid="sound-canvas"
							className={`absolute inset-0 h-full w-full touch-none ${tool === "move" ? "cursor-grab active:cursor-grabbing" : "cursor-crosshair"}`}
							onPointerDown={handlePointerDown}
							onPointerMove={handlePointerMove}
							onPointerUp={handlePointerUp}
							onPointerCancel={handlePointerUp}
							role="img"
							aria-label="Interactive pressure field. Drag the sensors, speakers, and tracked human."
						/>

						<div className="pointer-events-none absolute left-4 top-4 flex flex-wrap items-center gap-2">
							<span className="rounded-md border border-white/10 bg-[#15141b]/85 px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wider text-white/50 backdrop-blur">
								Total pressure
							</span>
							<span className="rounded-md border border-[#f694ff]/20 bg-[#15141b]/85 px-2.5 py-1.5 font-mono text-[10px] text-[#f694ff] backdrop-blur">
								{controller.mode === "human" ? "Bubble avg" : "Γ fit"}{" "}
								{formatDb(controller.reductionDb)}
							</span>
							{controller.mode === "human" ? (
								<span className="rounded-md border border-[#ff6767]/20 bg-[#15141b]/85 px-2.5 py-1.5 font-mono text-[10px] text-[#ff6767] backdrop-blur">
									Guard max {formatDb(controller.worstGuardDb)}
								</span>
							) : null}
						</div>

						<div className="absolute bottom-4 left-4 flex flex-wrap gap-1.5 rounded-lg border border-white/10 bg-[#15141b]/85 p-1.5 backdrop-blur">
							{toolOptions.map((option) => (
								<button
									key={option.id}
									type="button"
									onClick={() => setTool(option.id)}
									className={`inline-flex h-8 items-center gap-1.5 rounded-md px-2.5 text-[11px] transition ${
										tool === option.id
											? "bg-white/10 text-white"
											: "text-white/45 hover:bg-white/5 hover:text-white/80"
									}`}
									aria-pressed={tool === option.id}
								>
									<option.icon className="size-3.5" />
									<span className="hidden sm:inline">{option.label}</span>
								</button>
							))}
						</div>

						<div className="pointer-events-none absolute bottom-4 right-4 hidden rounded-md border border-white/10 bg-[#15141b]/85 px-2.5 py-2 font-mono text-[9px] uppercase leading-5 tracking-wider text-white/40 backdrop-blur sm:block">
							<span className="text-[#82e2ff]">◇</span> sensor &nbsp;
							<span className="text-[#a277ff]">○</span> speaker &nbsp;
							<span className="text-[#61ffca]">⊕</span>{" "}
							{controller.mode === "human" ? "tracked human" : "bystander"}
						</div>
					</div>
					<p className="px-1 text-xs leading-5 text-white/40">
						{controller.mode === "human"
							? autoTrack
								? "The controller follows the human around a slow ellipse. Grab the green crosshair to take over manually."
								: "Drag any node. The green samples define the quiet bubble; the red ring measures nearby amplification."
							: "Drag any node. Speakers pair to the nearest sensor; the dashed ring is the exterior boundary being fitted."}
					</p>
				</div>

				<aside className="space-y-4">
					<section
						data-testid="bystander-meter"
						className="overflow-hidden rounded-2xl border border-[#61ffca]/20 bg-[#1b1924]"
					>
						<div className="flex items-start justify-between gap-4 p-5 pb-2">
							<div>
								<p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.17em] text-[#61ffca]">
									<Gauge className="size-3.5" />{" "}
									{controller.mode === "human" ? "Tracked human" : "Bystander meter"}
								</p>
								<p className="mt-2 text-xs text-white/45">
									{controller.mode === "human"
										? `Optimizing a ${bubbleRadius.toFixed(2)} m bubble around this point.`
										: "Drag the green crosshair anywhere outside."}
								</p>
								{controller.mode === "human" ? (
									<button
										type="button"
										onClick={() => setAutoTrack((value) => !value)}
										className={`mt-3 inline-flex items-center gap-2 rounded-full border px-2.5 py-1 font-mono text-[9px] uppercase tracking-[0.12em] transition ${
											autoTrack
												? "border-[#61ffca]/30 bg-[#61ffca]/8 text-[#61ffca]"
												: "border-white/10 text-white/40 hover:border-white/20 hover:text-white/70"
										}`}
										aria-pressed={autoTrack}
									>
										<span
											className={`size-1.5 rounded-full ${autoTrack ? "bg-[#61ffca] shadow-[0_0_8px_#61ffca]" : "bg-white/25"}`}
										/>
										{autoTrack ? "Auto orbit on" : "Manual position"}
									</button>
								) : null}
							</div>
							<span
								className={`font-mono text-xl font-semibold ${observerDb <= 0 ? "text-[#61ffca]" : "text-[#ff6767]"}`}
							>
								{formatDb(observerDb)}
							</span>
						</div>
						<div className="px-5">
							<Waveform ratio={observerRatio} />
						</div>
						<div className="grid grid-cols-2 border-t border-white/8">
							<div className="border-r border-white/8 px-5 py-3">
								<p className="text-[10px] uppercase tracking-wider text-white/35">Source only</p>
								<p className="mt-1 font-mono text-xs text-[#ffca85]">{magnitude(sourceAtObserver).toFixed(3)} p</p>
							</div>
							<div className="px-5 py-3">
								<p className="text-[10px] uppercase tracking-wider text-white/35">With array</p>
								<p className="mt-1 font-mono text-xs text-[#61ffca]">{magnitude(totalAtObserver).toFixed(3)} p</p>
							</div>
						</div>
					</section>

					<section className="rounded-2xl border border-white/10 bg-[#1b1924] p-5">
						<div className="grid grid-cols-2 rounded-lg border border-white/10 bg-[#15141b] p-1">
							<button
								type="button"
								onClick={() => setControllerMode("human")}
								className={`rounded-md px-2 py-2 text-[11px] font-medium transition ${
									controllerMode === "human"
										? "bg-[#61ffca]/12 text-[#61ffca]"
										: "text-white/40 hover:text-white/70"
								}`}
								aria-pressed={controllerMode === "human"}
							>
								Human bubble
							</button>
							<button
								type="button"
								onClick={() => setControllerMode("boundary")}
								className={`rounded-md px-2 py-2 text-[11px] font-medium transition ${
									controllerMode === "boundary"
										? "bg-[#a277ff]/15 text-[#a277ff]"
										: "text-white/40 hover:text-white/70"
								}`}
								aria-pressed={controllerMode === "boundary"}
							>
								Exterior ring
							</button>
						</div>

						<div className="flex items-center justify-between gap-4">
							<div className="mt-5">
								<p className="font-mono text-[10px] uppercase tracking-[0.17em] text-[#a277ff]">Controller</p>
								<p className="mt-1 text-sm font-medium">
									{controller.mode === "human"
										? "Human-aware phase optimizer"
										: "Delayed sensor inversion"}
								</p>
							</div>
							<button
								type="button"
								onClick={() => setControlEnabled((value) => !value)}
								className={`relative h-6 w-11 rounded-full transition ${controlEnabled ? "bg-[#a277ff]" : "bg-white/15"}`}
								aria-label="Toggle control speakers"
								aria-pressed={controlEnabled}
							>
								<span className={`absolute top-1 size-4 rounded-full bg-white transition ${controlEnabled ? "left-6" : "left-1"}`} />
							</button>
						</div>

						<label className="mt-6 block text-xs text-white/55" htmlFor="frequency">
							<span className="flex items-center justify-between">
								Frequency <output className="font-mono text-[#ffca85]">{frequency} Hz</output>
							</span>
							<input
								id="frequency"
								type="range"
								min="100"
								max="1000"
								step="10"
								value={frequency}
								onChange={(event) => setFrequency(Number(event.target.value))}
								className="mt-3 w-full accent-[#a277ff]"
							/>
						</label>

						{controller.mode === "human" ? (
							<div className="mt-5 rounded-xl border border-[#61ffca]/15 bg-[#61ffca]/[0.035] p-3.5">
								<div className="flex items-center justify-between gap-3">
									<p className="font-mono text-[9px] uppercase tracking-[0.16em] text-[#61ffca]">
										Best controllable bands
									</p>
									<span className="font-mono text-[9px] text-white/30">100–1000 Hz scan</span>
								</div>
								<div className="mt-3 grid grid-cols-3 gap-1.5" data-testid="frequency-rankings">
									{rankedFrequencies.map((candidate, index) => (
										<button
											key={candidate.frequency}
											type="button"
											onClick={() => setFrequency(candidate.frequency)}
											className={`rounded-lg border px-2 py-2 text-left transition ${
												frequency === candidate.frequency
													? "border-[#61ffca]/40 bg-[#61ffca]/10"
													: "border-white/8 bg-[#15141b] hover:border-white/20"
											}`}
										>
											<span className="block font-mono text-[10px] text-white/75">
												{index + 1}. {candidate.frequency} Hz
											</span>
											<span className="mt-1 block font-mono text-[9px] text-[#61ffca]">
												{formatDb(candidate.bubbleDb)}
											</span>
										</button>
									))}
								</div>
								<p className="mt-3 text-[10px] leading-4 text-white/38">
									These are source bands this geometry controls best. Each anti-tone must match the source frequency.
								</p>
							</div>
						) : null}

						<label className="mt-5 block text-xs text-white/55" htmlFor="delay">
							<span className="flex items-center justify-between">
								{controller.mode === "human" ? "Reference delay θ" : "Processing delay θ"}{" "}
								<output className="font-mono text-[#82e2ff]">{delay.toFixed(1)} ms</output>
							</span>
							<input
								id="delay"
								type="range"
								min="0"
								max="12"
								step="0.1"
								value={delay}
								onChange={(event) => setDelay(Number(event.target.value))}
								className="mt-3 w-full accent-[#82e2ff]"
							/>
						</label>

						{controller.mode === "human" ? (
							<label className="mt-5 block text-xs text-white/55" htmlFor="bubble-radius">
								<span className="flex items-center justify-between">
									Quiet bubble radius{" "}
									<output className="font-mono text-[#61ffca]">{bubbleRadius.toFixed(2)} m</output>
								</span>
								<input
									id="bubble-radius"
									type="range"
									min="0.15"
									max="0.9"
									step="0.05"
									value={bubbleRadius}
									onChange={(event) => setBubbleRadius(Number(event.target.value))}
									className="mt-3 w-full accent-[#61ffca]"
								/>
							</label>
						) : (
							<label className="mt-5 block text-xs text-white/55" htmlFor="boundary">
								<span className="flex items-center justify-between">
									Control boundary Γ{" "}
									<output className="font-mono text-[#f694ff]">{boundaryRadius.toFixed(2)} m</output>
								</span>
								<input
									id="boundary"
									type="range"
									min="2.4"
									max="3.7"
									step="0.05"
									value={boundaryRadius}
									onChange={(event) => setBoundaryRadius(Number(event.target.value))}
									className="mt-3 w-full accent-[#f694ff]"
								/>
							</label>
						)}
					</section>

					<section className="rounded-2xl border border-white/10 bg-[#1b1924] p-5">
						<div className="flex items-center justify-between">
							<div>
								<p className="font-mono text-[10px] uppercase tracking-[0.17em] text-[#82e2ff]">Array geometry</p>
								<p className="mt-1 text-xs text-white/45">
									<span data-testid="sensor-count">{sensors.length}</span> sensors · <span data-testid="speaker-count">{speakers.length}</span> speakers
								</p>
							</div>
							<Radio className="size-4 text-[#82e2ff]" />
						</div>
						<div className="mt-4 grid grid-cols-3 gap-2">
							{[4, 8, 16].map((count) => (
								<button
									key={count}
									type="button"
									onClick={() => reset(count)}
									className={`rounded-lg border px-2 py-2 font-mono text-[11px] transition ${
										speakers.length === count && sensors.length === count
											? "border-[#82e2ff]/35 bg-[#82e2ff]/10 text-[#82e2ff]"
											: "border-white/10 text-white/45 hover:border-white/20 hover:text-white"
									}`}
								>
									{count} × {count}
								</button>
							))}
						</div>
						<button
							type="button"
							onClick={() => {
								const ring = buildRing(12);
								setSensors(buildRing(12, 1.55, Math.PI / 12));
								setSpeakers(ring.filter((_, index) => index < 9));
							}}
							className="mt-2 w-full rounded-lg border border-white/10 px-3 py-2 text-left text-[11px] text-white/45 transition hover:border-[#ff6767]/30 hover:text-[#ff6767]"
						>
							Make a gap in the array →
						</button>
					</section>
					</aside>
			</section>
			</div>

			<div id="experiment-3" ref={multipleSectionRef} className="scroll-mt-14 border-b border-white/10">
				<ExperimentHeading
					number="3"
					eyebrow="Shared degrees of freedom"
					title="Can the same array protect three moving people?"
					body="Add every person’s bubble to one objective. At low frequencies, many sample constraints share the same few spatial modes."
					color="#f694ff"
				/>
				<MultiHumanLab key={multiResetKey} running={running} active={multipleActive} />
			</div>

			<div id="experiment-4" ref={obstacleSectionRef} className="scroll-mt-14 border-b border-white/10">
				<ExperimentHeading
					number="4"
					eyebrow="Model mismatch · two buildings"
					title="What happens when the controller uses the wrong H?"
					body="Keep the open-field algorithm, but add two small reflective squares beside a wide direct path. The wrong phases can make the listener’s level increase."
					color="#ff6767"
				/>
				<ObstacleLab key={obstacleResetKey} running={running} active={obstacleActive} mode="open" />
			</div>

			<div id="experiment-5" ref={estimationSectionRef} className="scroll-mt-14 border-b border-white/10">
				<ExperimentHeading
					number="5"
					eyebrow="Sparse acoustic observations"
					title="Can a moving microphone estimate the missing H?"
					body="Fly a microphone through four nearby points. Orthogonal speaker probes measure one row of the transfer matrix at each stop, while the camera supplies scene geometry."
					color="#82e2ff"
				/>
				<HEstimationLab key={estimationResetKey} running={running} active={estimationActive} />
			</div>

			<div id="experiment-6" ref={recoverySectionRef} className="scroll-mt-14 border-b border-white/10">
				<ExperimentHeading
					number="6"
					eyebrow="Re-optimized with estimated H"
					title="The same speaker ring cancels the reflected field."
					body="Use the camera-conditioned transfer estimate instead of free space, then solve the speaker phases again. The obstacles stay; the model mismatch does not."
					color="#61ffca"
				/>
				<ObstacleLab key={recoveryResetKey} running={running} active={recoveryActive} mode="estimated" />
			</div>

			<div id="experiment-7" ref={chainsawSectionRef} className="scroll-mt-14 border-b border-white/10">
				<ExperimentHeading
					number="7"
					eyebrow="Measured waveform · broadband source"
					title="Replace the sine wave with an actual chainsaw recording."
					body="Extract the repeating engine cycle and its first twelve harmonics, optimize every frequency independently, then account for the non-periodic blade noise the harmonic model cannot cancel."
					color="#ffca85"
				/>
				<ChainsawLab key={chainsawResetKey} running={running} active={chainsawActive} />
			</div>

			<section className="border-t border-white/10 px-5 py-16 sm:px-8 sm:py-20">
				<div className="mx-auto max-w-5xl">
					<div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-[#ffca85]">
						<CircleHelp className="size-3.5" /> The mathematical answer
					</div>
					<h2 className="mt-4 max-w-3xl text-3xl font-semibold tracking-tight sm:text-4xl">
						There isn’t a blanket impossibility proof.
					</h2>
					<p className="mt-4 max-w-3xl text-base leading-7 text-white/55 sm:text-lg">
						An ideal continuous shell can cancel an ideal wave outside it. What fails in practice is the finite, delayed, broadband version: a handful of point speakers cannot usually satisfy an infinite boundary of constraints at once.
					</p>

					<div className="mt-10 rounded-2xl border border-[#61ffca]/15 bg-[#61ffca]/[0.035] p-5 sm:p-7">
						<div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr] lg:items-center">
							<div>
								<p className="font-mono text-[10px] uppercase tracking-[0.17em] text-[#61ffca]">
									Human-aware objective
								</p>
								<h3 className="mt-3 text-xl font-semibold">Optimize a region, not one fragile point.</h3>
								<p className="mt-3 text-sm leading-6 text-white/50">
									The green disk is sampled at the center and across three rings. The controller minimizes their combined pressure while lightly penalizing speaker effort and disturbance on the red guard ring.
								</p>
							</div>
							<div className="overflow-x-auto rounded-xl border border-white/8 bg-[#15141b] p-4 font-mono text-xs leading-7 text-white/65">
								<span className="text-[#61ffca]">w*₍f₎</span> = arg min Σ<span className="text-white/35">x∈bubble</span>
								 |d₍f₎(x) + H₍f₎(x)w₍f₎|²
								<br />
								<span className="text-white/35">+ λ‖w₍f₎‖² + guard penalty</span>
								<br />
								<span className="text-[#ffca85]">anti-frequency f = source frequency f</span>
							</div>
						</div>
						<p className="mt-5 border-t border-white/8 pt-5 text-xs leading-5 text-white/40">
							For additional people, we add each person’s bubble samples to the same objective. The available speaker degrees of freedom are then shared across all protected regions.
						</p>
					</div>

					<div className="mt-4 grid gap-4 lg:grid-cols-2">
						<div className="rounded-2xl border border-white/10 bg-[#100f15] p-5 sm:p-7">
							<div className="flex items-center gap-2 text-sm font-medium">
								<Activity className="size-4 text-[#a277ff]" /> Finite point sources
							</div>
							<div className="mt-5 overflow-x-auto rounded-xl border border-white/8 bg-[#15141b] p-4 font-mono text-xs leading-7 text-white/65">
								<span className="text-[#edecee]">p(x)</span> = q₀ G(x,x₀) + <span className="text-[#a277ff]">Σ aⱼ G(x,xⱼ)</span>
								<br />
								<span className="text-[#82e2ff]">p(xₘ) ≈ 0</span>, m = 1…M
							</div>
							<p className="mt-5 text-sm leading-6 text-white/50">
								Our N speaker gains are fitted to M sampled points on Γ. Exact silence outside would require p = 0 at every point of Γ—infinitely many conditions. A finite displaced array generally leaves angular modes uncancelled.
							</p>
						</div>

						<div className="rounded-2xl border border-[#61ffca]/15 bg-[#100f15] p-5 sm:p-7">
							<div className="flex items-center gap-2 text-sm font-medium">
								<Sparkles className="size-4 text-[#61ffca]" /> The ideal exception
							</div>
							<div className="mt-5 overflow-x-auto rounded-xl border border-white/8 bg-[#15141b] p-4 font-mono text-xs leading-7 text-white/65">
								<span className="text-[#edecee]">∫ring G(x,y) σ ds</span>
								<br />
								= C · <span className="text-[#61ffca]">J₀(ka) H₀⁽¹⁾(kr)</span>, r &gt; a
							</div>
							<p className="mt-5 text-sm leading-6 text-white/50">
								For a centered, single-frequency radial source, a continuous ring of in-phase monopoles produces the same exterior radial shape. Choose its density with the opposite coefficient and the outside field is exactly zero (except at special J₀ zeros).
							</p>
						</div>
					</div>

					<div className="mt-4 rounded-2xl border border-[#82e2ff]/15 bg-[#82e2ff]/[0.035] p-5 sm:p-7">
						<p className="font-mono text-[10px] uppercase tracking-[0.17em] text-[#82e2ff]">Why θ matters</p>
						<p className="mt-3 max-w-4xl text-sm leading-6 text-white/55">
							At one steady pure tone, the human-aware controller can compensate a fixed reference delay with complex phase weights. Exterior-ring mode intentionally uses only real gains on delayed sensor copies, so θ remains visible there. For an unpredictable broadband saw waveform, causality is stricter: the sensor must hear the disturbance early enough that the secondary wave can still reach the listener at the same instant.
						</p>
					</div>
				</div>
			</section>
		</main>
	);
}
