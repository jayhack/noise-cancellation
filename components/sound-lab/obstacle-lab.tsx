"use client";

import { Gauge, ShieldAlert } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
	pressureFieldColor,
	WAVE_FIELD_BACKGROUND,
} from "@/lib/acoustics/wave-palette";

type Complex = { re: number; im: number };
type Point = { x: number; y: number };
type WeightedPoint = Point & { weight: number };
type Rect = { x0: number; x1: number; y0: number; y1: number; label: string };

type OpenFieldController = {
	waveNumber: number;
	weights: Complex[];
	targets: WeightedPoint[];
	guards: Point[];
	predictedBubbleDb: number;
	predictedCenterDb: number;
};

type ActualResult = {
	bubbleDb: number;
	centerDb: number;
	worstGuardDb: number;
};

const WORLD = { width: 12, height: 8 };
const SOURCE = { x: 4, y: 4 };
const HUMAN = { x: 9.35, y: 4 };
const SOUND_SPEED = 343;
const SPEAKER_COUNT = 8;
const BUBBLE_RADIUS = 0.45;
const MAX_SPEAKER_STRENGTH = 4;

const OBSTACLES: Rect[] = [
	{ x0: 6.05, x1: 7.2, y0: 1.35, y1: 2.5, label: "BUILDING A" },
	{ x0: 6.65, x1: 7.8, y0: 5.05, y1: 6.2, label: "BUILDING B" },
];

const PROBE_POINTS: Point[] = [
	{ x: 8.25, y: 2.65 },
	{ x: 9.05, y: 2.25 },
	{ x: 9.75, y: 2.8 },
	{ x: 9.9, y: 3.65 },
];

const SEQUENCE_STAGES = [
	{
		title: "Solve in an open field",
		result: "−23.0 dB predicted",
		body: "The naïve controller assumes every sound path is direct.",
		color: "#168bd2",
	},
	{
		title: "Add reflective obstacles",
		result: "+4.2 dB actual",
		body: "Buildings add delayed arrivals, so the old speaker phases now make the target louder.",
		color: "#ff3b24",
	},
	{
		title: "Measure the missing paths",
		result: "4 probe readings",
		body: "Sparse microphones anchor an environment-aware estimate of the transfer matrix H.",
		color: "#ffc247",
	},
	{
		title: "Re-solve with known H",
		result: "−10.2 dB actual",
		body: "The obstacles stay, but the corrected controller now phases the same array against the reflections.",
		color: "#3bb9e8",
	},
] as const;

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

function cScale(value: Complex, scale: number): Complex {
	return { re: value.re * scale, im: value.im * scale };
}

function magnitude(value: Complex) {
	return Math.hypot(value.re, value.im);
}

function distance(left: Point, right: Point) {
	return Math.hypot(left.x - right.x, left.y - right.y);
}

function phasorForPath(pathLength: number, waveNumber: number, gain = 1): Complex {
	const safeLength = Math.max(0.09, pathLength);
	const amplitude = gain / Math.sqrt(safeLength);
	const phase = -waveNumber * safeLength;
	return {
		re: amplitude * Math.cos(phase),
		im: amplitude * Math.sin(phase),
	};
}

function openFieldGreen(from: Point, to: Point, waveNumber: number) {
	return phasorForPath(distance(from, to), waveNumber);
}

function segmentIntersectsRect(from: Point, to: Point, rect: Rect) {
	let start = 0;
	let end = 1;
	const dx = to.x - from.x;
	const dy = to.y - from.y;
	const boundaries = [
		[-dx, from.x - rect.x0],
		[dx, rect.x1 - from.x],
		[-dy, from.y - rect.y0],
		[dy, rect.y1 - from.y],
	] as const;

	for (const [direction, offset] of boundaries) {
		if (Math.abs(direction) < 1e-9) {
			if (offset < 0) return false;
			continue;
		}
		const ratio = offset / direction;
		if (direction < 0) {
			if (ratio > end) return false;
			if (ratio > start) start = ratio;
		} else {
			if (ratio < start) return false;
			if (ratio < end) end = ratio;
		}
	}
	return end > 0 && start < 1;
}

function reflectedPath(
	from: Point,
	to: Point,
	rect: Rect,
	axis: "x" | "y",
	boundary: number,
	segmentMin: number,
	segmentMax: number,
	waveNumber: number,
	gain: number,
) {
	const mirrored =
		axis === "x"
			? { x: boundary * 2 - from.x, y: from.y }
			: { x: from.x, y: boundary * 2 - from.y };
	const denominator = axis === "x" ? to.x - mirrored.x : to.y - mirrored.y;
	if (Math.abs(denominator) < 1e-6) return { re: 0, im: 0 };
	const interpolation =
		(boundary - (axis === "x" ? mirrored.x : mirrored.y)) / denominator;
	if (interpolation <= 0 || interpolation >= 1) return { re: 0, im: 0 };
	const crossCoordinate =
		axis === "x"
			? mirrored.y + interpolation * (to.y - mirrored.y)
			: mirrored.x + interpolation * (to.x - mirrored.x);
	if (crossCoordinate < segmentMin || crossCoordinate > segmentMax) {
		return { re: 0, im: 0 };
	}
	const reflectionPoint =
		axis === "x"
			? { x: boundary, y: crossCoordinate }
			: { x: crossCoordinate, y: boundary };
	return phasorForPath(
		distance(from, reflectionPoint) + distance(reflectionPoint, to),
		waveNumber,
		gain,
	);
}

/**
 * Lightweight 2D scene model: direct sound, first-order specular reflections,
 * attenuation through buildings, and a two-corner diffraction approximation.
 * The controller intentionally never sees this function.
 */
function environmentGreen(
	from: Point,
	to: Point,
	waveNumber: number,
	reflectionStrength: number,
) {
	const blocked = OBSTACLES.some((rect) => segmentIntersectsRect(from, to, rect));
	let field = cScale(
		openFieldGreen(from, to, waveNumber),
		blocked ? 0.12 : 1,
	);

	for (const rect of OBSTACLES) {
		const reflections = [
			reflectedPath(
				from,
				to,
				rect,
				"x",
				rect.x0,
				rect.y0,
				rect.y1,
				waveNumber,
				0.58 * reflectionStrength,
			),
			reflectedPath(
				from,
				to,
				rect,
				"x",
				rect.x1,
				rect.y0,
				rect.y1,
				waveNumber,
				0.46 * reflectionStrength,
			),
			reflectedPath(
				from,
				to,
				rect,
				"y",
				rect.y0,
				rect.x0,
				rect.x1,
				waveNumber,
				0.58 * reflectionStrength,
			),
			reflectedPath(
				from,
				to,
				rect,
				"y",
				rect.y1,
				rect.x0,
				rect.x1,
				waveNumber,
				0.58 * reflectionStrength,
			),
		];
		for (const reflection of reflections) field = cAdd(field, reflection);

		if (segmentIntersectsRect(from, to, rect)) {
			const corners = [
				{ x: rect.x0, y: rect.y0 },
				{ x: rect.x0, y: rect.y1 },
				{ x: rect.x1, y: rect.y0 },
				{ x: rect.x1, y: rect.y1 },
			]
				.sort(
					(left, right) =>
						distance(from, left) + distance(left, to) -
						(distance(from, right) + distance(right, to)),
				)
				.slice(0, 2);
			for (const corner of corners) {
				field = cAdd(
					field,
					phasorForPath(
						distance(from, corner) + distance(corner, to),
						waveNumber,
						0.28 * reflectionStrength,
					),
				);
			}
		}
	}
	return field;
}

function buildRing(count: number, radius = 1.55, angleOffset = 0) {
	return Array.from({ length: count }, (_, index) => {
		const angle = (index / count) * Math.PI * 2 + angleOffset;
		return {
			x: SOURCE.x + Math.cos(angle) * radius,
			y: SOURCE.y + Math.sin(angle) * radius,
		};
	});
}

function pointsOnDisk(center: Point, radius: number) {
	const points: WeightedPoint[] = [{ x: center.x, y: center.y, weight: 2.4 }];
	for (const ring of [
		{ radius: radius * 0.35, count: 8, weight: 1.6 },
		{ radius: radius * 0.7, count: 12, weight: 1.15 },
		{ radius, count: 16, weight: 0.85 },
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

function pointsOnGuardRing(center: Point, bubbleRadius: number) {
	const radius = Math.max(0.85, bubbleRadius * 1.9);
	return Array.from({ length: 24 }, (_, index) => {
		const angle = (index / 24) * Math.PI * 2;
		return {
			x: center.x + Math.cos(angle) * radius,
			y: center.y + Math.sin(angle) * radius,
		};
	});
}

function solveLinearSystem(matrix: number[][], vector: number[]) {
	const size = vector.length;
	const rows = matrix.map((row, index) => [...row, vector[index]]);
	for (let column = 0; column < size; column += 1) {
		let pivot = column;
		for (let row = column + 1; row < size; row += 1) {
			if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) pivot = row;
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

function pressureWith(
	point: Point,
	waveNumber: number,
	speakers: Point[],
	weights: Complex[],
	propagator: (from: Point, to: Point) => Complex,
) {
	let pressure = propagator(SOURCE, point);
	for (let index = 0; index < speakers.length; index += 1) {
		pressure = cAdd(
			pressure,
			cMul(weights[index] ?? { re: 0, im: 0 }, propagator(speakers[index], point)),
		);
	}
	return pressure;
}

function calculateController({
	human,
	speakers,
	frequency,
	reflectionStrength,
	model,
}: {
	human: Point;
	speakers: Point[];
	frequency: number;
	reflectionStrength: number;
	model: "open" | "estimated";
}): OpenFieldController {
	const waveNumber = (Math.PI * 2 * frequency) / SOUND_SPEED;
	const modelPropagator = (from: Point, to: Point) =>
		model === "estimated"
			? environmentGreen(from, to, waveNumber, reflectionStrength)
			: openFieldGreen(from, to, waveNumber);
	const targets = pointsOnDisk(human, BUBBLE_RADIUS);
	const guards = pointsOnGuardRing(human, BUBBLE_RADIUS);
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
			const column = modelPropagator(speakers[index], point);
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
		addConstraint(
			point,
			cScale(modelPropagator(SOURCE, point), -1),
			point.weight,
		);
	}
	for (const point of guards) {
		addConstraint(
			point,
			{ re: 0, im: 0 },
			model === "estimated" ? 0.02 : 0.075,
		);
	}

	const diagonalMean =
		unknowns > 0
			? normal.reduce((sum, row, index) => sum + row[index], 0) / unknowns
			: 0;
	for (let index = 0; index < unknowns; index += 1) {
		normal[index][index] += diagonalMean * 0.003 + 1e-6;
	}
	const solution = solveLinearSystem(normal, target);
	let weights = speakers.map((_, index) => ({
		re: solution[index * 2] ?? 0,
		im: solution[index * 2 + 1] ?? 0,
	}));
	const strongest = weights.reduce(
		(maximum, weight) => Math.max(maximum, magnitude(weight)),
		0,
	);
	if (strongest > MAX_SPEAKER_STRENGTH) {
		weights = weights.map((weight) =>
			cScale(weight, MAX_SPEAKER_STRENGTH / strongest),
		);
	}

	let sourceEnergy = 0;
	let controlledEnergy = 0;
	for (const point of targets) {
		const source = modelPropagator(SOURCE, point);
		const controlled = pressureWith(
			point,
			waveNumber,
			speakers,
			weights,
			modelPropagator,
		);
		sourceEnergy += point.weight * magnitude(source) ** 2;
		controlledEnergy += point.weight * magnitude(controlled) ** 2;
	}
	const centerSource = modelPropagator(SOURCE, human);
	const centerControlled = pressureWith(
		human,
		waveNumber,
		speakers,
		weights,
		modelPropagator,
	);
	return {
		waveNumber,
		weights,
		targets,
		guards,
		predictedBubbleDb:
			10 * Math.log10(Math.max(1e-12, controlledEnergy / sourceEnergy)),
		predictedCenterDb:
			20 *
			Math.log10(
				Math.max(1e-7, magnitude(centerControlled) / magnitude(centerSource)),
			),
	};
}

function evaluateActualScene({
	human,
	speakers,
	controller,
	reflectionStrength,
}: {
	human: Point;
	speakers: Point[];
	controller: OpenFieldController;
	reflectionStrength: number;
}): ActualResult {
	const propagator = (from: Point, to: Point) =>
		environmentGreen(
			from,
			to,
			controller.waveNumber,
			reflectionStrength,
		);
	let sourceEnergy = 0;
	let controlledEnergy = 0;
	for (const point of controller.targets) {
		const source = propagator(SOURCE, point);
		const controlled = pressureWith(
			point,
			controller.waveNumber,
			speakers,
			controller.weights,
			propagator,
		);
		sourceEnergy += point.weight * magnitude(source) ** 2;
		controlledEnergy += point.weight * magnitude(controlled) ** 2;
	}
	const centerSource = propagator(SOURCE, human);
	const centerControlled = pressureWith(
		human,
		controller.waveNumber,
		speakers,
		controller.weights,
		propagator,
	);
	let worstGuardDb = -Infinity;
	for (const point of controller.guards) {
		const source = propagator(SOURCE, point);
		const controlled = pressureWith(
			point,
			controller.waveNumber,
			speakers,
			controller.weights,
			propagator,
		);
		worstGuardDb = Math.max(
			worstGuardDb,
			20 * Math.log10(Math.max(1e-7, magnitude(controlled) / magnitude(source))),
		);
	}
	return {
		bubbleDb: 10 * Math.log10(Math.max(1e-12, controlledEnergy / sourceEnergy)),
		centerDb:
			20 *
			Math.log10(
				Math.max(1e-7, magnitude(centerControlled) / magnitude(centerSource)),
			),
		worstGuardDb: Number.isFinite(worstGuardDb) ? worstGuardDb : 0,
	};
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

export function ObstacleLab({
	running,
	active,
	mode,
	showControls,
}: {
	running: boolean;
	active: boolean;
	mode: "open" | "estimated" | "sequence";
	showControls: boolean;
}) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const frameRef = useRef(0);
	const lastFrameRef = useRef(0);
	const timeRef = useRef(0);
	const [frequency, setFrequency] = useState(240);
	const [reflectionStrength, setReflectionStrength] = useState(1.2);
	const [controlEnabled, setControlEnabled] = useState(true);
	const [sequenceStage, setSequenceStage] = useState(0);
	const sequenceMode = mode === "sequence";

	useEffect(() => {
		if (!sequenceMode || !active || !running) return;
		const interval = window.setInterval(() => {
			setSequenceStage((stage) => (stage + 1) % SEQUENCE_STAGES.length);
		}, 3400);
		return () => window.clearInterval(interval);
	}, [active, running, sequenceMode]);
	const speakers = useMemo(() => buildRing(SPEAKER_COUNT), []);
	const sensors = useMemo(
		() => buildRing(SPEAKER_COUNT, 1.55, Math.PI / SPEAKER_COUNT),
		[],
	);
	const openController = useMemo(
		() =>
			calculateController({
				human: HUMAN,
				speakers,
				frequency,
				reflectionStrength,
				model: "open",
			}),
		[frequency, reflectionStrength, speakers],
	);
	const estimatedController = useMemo(
		() =>
			calculateController({
				human: HUMAN,
				speakers,
				frequency,
				reflectionStrength,
				model: "estimated",
			}),
		[frequency, reflectionStrength, speakers],
	);
	const estimatedMode = mode === "estimated" || (sequenceMode && sequenceStage === 3);
	const worldHasObstacles = !sequenceMode || sequenceStage >= 1;
	const measurementsVisible = sequenceMode && sequenceStage === 2;
	const controller = estimatedMode ? estimatedController : openController;
	const openActual = useMemo(
		() =>
			evaluateActualScene({
				human: HUMAN,
				speakers,
				controller: openController,
				reflectionStrength,
			}),
		[openController, reflectionStrength, speakers],
	);
	const actual = useMemo(
		() =>
			evaluateActualScene({
				human: HUMAN,
				speakers,
				controller,
				reflectionStrength,
			}),
		[controller, reflectionStrength, speakers],
	);
	const attenuationLost = Math.max(
		0,
		openActual.bubbleDb - openController.predictedBubbleDb,
	);
	const shownAttenuationLost = controlEnabled ? attenuationLost : 0;
	const recoveredDb = Math.max(0, openActual.bubbleDb - actual.bubbleDb);
	const shownRecoveredDb = controlEnabled ? recoveredDb : 0;
	const shownActualDb = worldHasObstacles
		? actual.bubbleDb
		: openController.predictedBubbleDb;

	const draw = useCallback(
		(timestamp: number) => {
			if (!active) return;
			if (timestamp - lastFrameRef.current < 50) return;
			const delta = Math.min(0.05, (timestamp - lastFrameRef.current) / 1000);
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
			const propagator = worldHasObstacles
				? (from: Point, to: Point) =>
						environmentGreen(
							from,
							to,
							controller.waveNumber,
							reflectionStrength,
						)
				: (from: Point, to: Point) =>
						openFieldGreen(from, to, controller.waveNumber);

			context.fillStyle = WAVE_FIELD_BACKGROUND;
			context.fillRect(0, 0, width, height);
			const step = Math.max(9, Math.round(8 * dpr));
			const phase = timeRef.current * Math.PI * 1.35;
			const cosTime = Math.cos(phase);
			const sinTime = Math.sin(phase);
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
					const phasor = controlEnabled
						? pressureWith(
								point,
								controller.waveNumber,
								speakers,
								controller.weights,
								propagator,
							)
						: propagator(SOURCE, point);
					const instantaneous = phasor.re * cosTime - phasor.im * sinTime;
					const signedStrength = Math.tanh(instantaneous * 0.62);
					context.fillStyle = pressureFieldColor(signedStrength, 0.76);
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

			const sourcePoint = toCanvas(SOURCE);
			const humanPoint = toCanvas(HUMAN);
			context.save();
			context.setLineDash([6 * dpr, 7 * dpr]);
			context.strokeStyle = "rgba(255,194,71,.44)";
			context.lineWidth = 1.3 * dpr;
			context.beginPath();
			context.moveTo(sourcePoint.x, sourcePoint.y);
			context.lineTo(humanPoint.x, humanPoint.y);
			context.stroke();
			if (worldHasObstacles) {
				for (const corner of [
					{ x: OBSTACLES[0].x1, y: OBSTACLES[0].y1 },
					{ x: OBSTACLES[1].x1, y: OBSTACLES[1].y0 },
				]) {
					const cornerPoint = toCanvas(corner);
					context.strokeStyle = "rgba(22,139,210,.55)";
					context.beginPath();
					context.moveTo(sourcePoint.x, sourcePoint.y);
					context.lineTo(cornerPoint.x, cornerPoint.y);
					context.lineTo(humanPoint.x, humanPoint.y);
					context.stroke();
				}
			}
			context.restore();

			for (let index = 0; index < speakers.length; index += 1) {
				const sensor = toCanvas(sensors[index]);
				const speaker = toCanvas(speakers[index]);
				context.strokeStyle = "rgba(22,139,210,.18)";
				context.beginPath();
				context.moveTo(sensor.x, sensor.y);
				context.lineTo(speaker.x, speaker.y);
				context.stroke();
				context.save();
				context.translate(sensor.x, sensor.y);
				context.rotate(Math.PI / 4);
				context.fillStyle = "#0b0e12";
				context.strokeStyle = AURA.blue;
				context.lineWidth = 1.5 * dpr;
				context.fillRect(-4.5 * dpr, -4.5 * dpr, 9 * dpr, 9 * dpr);
				context.strokeRect(-4.5 * dpr, -4.5 * dpr, 9 * dpr, 9 * dpr);
				context.restore();
				context.fillStyle = "#0b0e12";
				context.strokeStyle = controlEnabled ? "#2f6df6" : "rgba(47,109,246,.4)";
				context.lineWidth = 1.8 * dpr;
				context.beginPath();
				context.arc(speaker.x, speaker.y, 7.5 * dpr, 0, Math.PI * 2);
				context.fill();
				context.stroke();
			}

			if (worldHasObstacles) for (const rect of OBSTACLES) {
				const topLeft = toCanvas({ x: rect.x0, y: rect.y0 });
				const bottomRight = toCanvas({ x: rect.x1, y: rect.y1 });
				const rectWidth = bottomRight.x - topLeft.x;
				const rectHeight = bottomRight.y - topLeft.y;
				context.fillStyle = "rgba(18,24,34,.96)";
				context.strokeStyle = "rgba(22,139,210,.62)";
				context.lineWidth = 2 * dpr;
				context.fillRect(topLeft.x, topLeft.y, rectWidth, rectHeight);
				context.strokeRect(topLeft.x, topLeft.y, rectWidth, rectHeight);
				context.save();
				context.beginPath();
				context.rect(topLeft.x, topLeft.y, rectWidth, rectHeight);
				context.clip();
				context.strokeStyle = "rgba(22,139,210,.09)";
				for (
					let offset = -rectHeight;
					offset < rectWidth + rectHeight;
					offset += 12 * dpr
				) {
					context.beginPath();
					context.moveTo(topLeft.x + offset, topLeft.y);
					context.lineTo(topLeft.x + offset - rectHeight, bottomRight.y);
					context.stroke();
				}
				context.restore();
				context.fillStyle = "rgba(22,139,210,.72)";
				context.font = `600 ${9 * dpr}px ui-monospace, monospace`;
				context.textAlign = "center";
				context.fillText(
					rect.label,
					topLeft.x + rectWidth / 2,
					topLeft.y + rectHeight / 2,
				);
			}

			if (measurementsVisible) {
				context.save();
				context.setLineDash([5 * dpr, 8 * dpr]);
				context.strokeStyle = "rgba(59,185,232,.7)";
				context.lineWidth = 1.5 * dpr;
				context.beginPath();
				PROBE_POINTS.forEach((probe, index) => {
					const point = toCanvas(probe);
					if (index === 0) context.moveTo(point.x, point.y);
					else context.lineTo(point.x, point.y);
				});
				context.stroke();
				context.setLineDash([]);
				PROBE_POINTS.forEach((probe, index) => {
					const point = toCanvas(probe);
					const pulse = (10 + Math.sin(timeRef.current * 5 + index) * 2.5) * dpr;
					context.fillStyle = "rgba(7,10,13,.92)";
					context.strokeStyle = index % 2 === 0 ? "#ffc247" : "#3bb9e8";
					context.lineWidth = 2 * dpr;
					context.beginPath();
					context.arc(point.x, point.y, pulse, 0, Math.PI * 2);
					context.fill();
					context.stroke();
					context.fillStyle = context.strokeStyle;
					context.font = `600 ${9 * dpr}px ui-monospace, monospace`;
					context.textAlign = "center";
					context.textBaseline = "middle";
					context.fillText(String(index + 1), point.x, point.y + dpr);
				});
				context.restore();
			}

			context.textAlign = "start";
			context.fillStyle = AURA.orange;
			context.beginPath();
			context.arc(sourcePoint.x, sourcePoint.y, 7 * dpr, 0, Math.PI * 2);
			context.fill();
			context.strokeStyle = "rgba(255,194,71,.35)";
			context.lineWidth = 8 * dpr;
			context.stroke();

			context.fillStyle = !worldHasObstacles
				? "rgba(22,139,210,.06)"
				: estimatedMode
					? "rgba(59,185,232,.06)"
					: "rgba(255,59,36,.06)";
			context.strokeStyle = !worldHasObstacles
				? AURA.blue
				: estimatedMode
					? "#3bb9e8"
					: AURA.red;
			context.lineWidth = 1.8 * dpr;
			context.beginPath();
			context.arc(
				humanPoint.x,
				humanPoint.y,
				BUBBLE_RADIUS * transform.scale,
				0,
				Math.PI * 2,
			);
			context.fill();
			context.stroke();
			context.fillStyle = "#0b0e12";
			context.lineWidth = 2 * dpr;
			context.beginPath();
			context.arc(humanPoint.x, humanPoint.y, 11 * dpr, 0, Math.PI * 2);
			context.fill();
			context.stroke();
			context.beginPath();
			context.moveTo(humanPoint.x - 15 * dpr, humanPoint.y);
			context.lineTo(humanPoint.x + 15 * dpr, humanPoint.y);
			context.moveTo(humanPoint.x, humanPoint.y - 15 * dpr);
			context.lineTo(humanPoint.x, humanPoint.y + 15 * dpr);
			context.stroke();

			const drawCallout = (
				anchor: { x: number; y: number },
				text: string,
				color: string,
				offsetY: number,
			) => {
				const fontSize = 15 * dpr;
				const paddingX = 9 * dpr;
				const boxHeight = 31 * dpr;
				context.font = `600 ${fontSize}px ui-monospace, monospace`;
				context.textBaseline = "middle";
				const boxWidth = context.measureText(text).width + paddingX * 2;
				const x = Math.min(anchor.x + 34 * dpr, width - boxWidth - 10 * dpr);
				const centerY = anchor.y + offsetY * dpr;
				context.strokeStyle = `${color}88`;
				context.beginPath();
				context.moveTo(anchor.x + 10 * dpr, anchor.y);
				context.lineTo(x - 4 * dpr, centerY);
				context.stroke();
				context.fillStyle = "rgba(7,10,13,.94)";
				context.beginPath();
				context.roundRect(x, centerY - boxHeight / 2, boxWidth, boxHeight, 7 * dpr);
				context.fill();
				context.strokeStyle = `${color}66`;
				context.stroke();
				context.fillStyle = color;
				context.fillText(text, x + paddingX, centerY + dpr);
			};

			drawCallout(sourcePoint, "SOURCE", AURA.orange, 0);
			drawCallout(
				humanPoint,
				`${!worldHasObstacles ? "OPEN FIELD" : estimatedMode ? "RECOVERED" : "ACTUAL"}  ${formatDb(controlEnabled ? shownActualDb : 0)}`,
				!worldHasObstacles ? AURA.blue : estimatedMode ? "#3bb9e8" : AURA.red,
				-46,
			);
			context.textBaseline = "alphabetic";
		},
		[
			active,
			controlEnabled,
			controller,
			estimatedMode,
			measurementsVisible,
			reflectionStrength,
			running,
			sensors,
			shownActualDb,
			speakers,
			worldHasObstacles,
		],
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

	const activeSequenceStage = SEQUENCE_STAGES[sequenceStage];

	return (
		<section className={`mx-auto grid max-w-[1500px] gap-4 p-4 sm:p-6 ${showControls ? "lg:grid-cols-[minmax(0,1fr)_340px]" : ""}`}>
				<div className="space-y-3">
					<div
						className={`relative aspect-[3/2] min-h-[420px] max-h-[720px] overflow-hidden rounded-2xl border bg-[#070a0d] shadow-2xl shadow-black/30 ${sequenceMode ? "" : estimatedMode ? "border-[#3bb9e8]/20" : "border-[#ff3b24]/20"}`}
						style={sequenceMode ? { borderColor: `${activeSequenceStage.color}44` } : undefined}
					>
						<canvas
							ref={canvasRef}
							data-testid={`obstacle-canvas-${mode}`}
							className="absolute inset-0 h-full w-full"
							role="img"
							aria-label={sequenceMode ? "Animated sequence from naïve open-field control through obstacle measurement and corrected cancellation" : estimatedMode ? "Environment-aware sound controller recovering cancellation around two reflective buildings" : "Open-field sound controller failing around two reflective buildings"}
						/>
						<div className="pointer-events-none absolute left-4 top-4 flex flex-wrap items-center gap-2">
							<span
								className={`rounded-md border bg-[#0b0e12]/88 px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wider backdrop-blur ${sequenceMode ? "" : estimatedMode ? "border-[#3bb9e8]/20 text-[#3bb9e8]" : "border-[#ff3b24]/20 text-[#ff3b24]"}`}
								style={sequenceMode ? { borderColor: `${activeSequenceStage.color}55`, color: activeSequenceStage.color } : undefined}
							>
								{sequenceMode ? `${sequenceStage + 1}/4 · ${activeSequenceStage.title} · ${activeSequenceStage.result}` : `${estimatedMode ? "Environment-aware" : "Naïve open-field"} · actual ${formatDb(controlEnabled ? actual.bubbleDb : 0)}`}
							</span>
						</div>
						{sequenceMode ? (
							<div className="absolute bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-2 rounded-full border border-white/10 bg-[#0b0e12]/88 px-3 py-2 backdrop-blur">
								{SEQUENCE_STAGES.map((stage, index) => (
									<button
										key={stage.title}
										type="button"
										onClick={() => setSequenceStage(index)}
										className={`h-2 rounded-full transition-all ${index === sequenceStage ? "w-9" : "w-4 bg-white/15 hover:bg-white/30"}`}
										style={index === sequenceStage ? { backgroundColor: stage.color } : undefined}
										aria-label={`Show obstacle stage ${index + 1}: ${stage.title}`}
										aria-pressed={index === sequenceStage}
									/>
								))}
							</div>
						) : (
							<div className="pointer-events-none absolute bottom-4 right-4 hidden rounded-md border border-white/10 bg-[#0b0e12]/88 px-2.5 py-2 font-mono text-[9px] uppercase tracking-wider text-white/40 backdrop-blur sm:block">
								cyan paths = reflected / diffracted arrivals
							</div>
						)}
					</div>
					<p className="px-1 text-base leading-7 text-white/50">
						{sequenceMode
							? activeSequenceStage.body
							: estimatedMode
							? "The same array now uses the estimated environmental transfer paths. Reflections are included when the speaker weights are solved."
							: "The optimizer still assumes direct free-space paths. The canvas evaluates those same speaker weights in a different world containing two reflective buildings."}
					</p>
				</div>

				{showControls ? <aside className="space-y-4">
					<section className={`overflow-hidden rounded-2xl border bg-[#111820] ${estimatedMode ? "border-[#3bb9e8]/20" : "border-[#ff3b24]/20"}`}>
						<div className="flex items-start justify-between gap-4 p-5">
							<div>
								<p className={`flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.17em] ${estimatedMode ? "text-[#3bb9e8]" : "text-[#ff3b24]"}`}>
									<Gauge className="size-3.5" /> {estimatedMode ? "Cancellation recovered" : "Model mismatch"}
								</p>
								<p className="mt-2 text-xs leading-5 text-white/45">
									{estimatedMode ? "Improvement after solving against the estimated transfer matrix." : "Attenuation lost because the controller uses the wrong transfer matrix."}
								</p>
							</div>
							<span className={`font-mono text-xl font-semibold ${estimatedMode ? "text-[#3bb9e8]" : "text-[#ff3b24]"}`}>
								{estimatedMode ? shownRecoveredDb.toFixed(1) : shownAttenuationLost.toFixed(1)} dB
							</span>
						</div>
						<div className="grid grid-cols-2 border-t border-white/8">
							<div className="border-r border-white/8 px-5 py-4">
								<p className="text-[9px] uppercase tracking-wider text-white/30">{estimatedMode ? "Before estimation" : "Open-field prediction"}</p>
								<p className={`mt-1.5 font-mono text-sm ${estimatedMode ? "text-[#ff3b24]" : "text-[#3bb9e8]"}`}>
									{formatDb(controlEnabled ? (estimatedMode ? openActual.bubbleDb : controller.predictedBubbleDb) : 0)}
								</p>
							</div>
							<div className="px-5 py-4">
								<p className="text-[9px] uppercase tracking-wider text-white/30">{estimatedMode ? "After re-optimization" : "Actual scene"}</p>
								<p className={`mt-1.5 font-mono text-sm ${estimatedMode ? "text-[#3bb9e8]" : "text-[#ff3b24]"}`}>
									{formatDb(controlEnabled ? actual.bubbleDb : 0)}
								</p>
							</div>
						</div>
					</section>

					<section className="rounded-2xl border border-white/10 bg-[#111820] p-5">
						<div className="flex items-center justify-between gap-4">
							<div>
								<p className="font-mono text-[10px] uppercase tracking-[0.17em] text-[#2f6df6]">
									{estimatedMode ? "Environment-aware algorithm" : "Open-field algorithm"}
								</p>
								<p className="mt-1 text-sm font-medium">{estimatedMode ? "Measured paths included in H" : "Buildings excluded from H"}</p>
							</div>
							<button
								type="button"
								onClick={() => setControlEnabled((value) => !value)}
								className={`relative h-6 w-11 rounded-full transition ${controlEnabled ? "bg-[#2f6df6]" : "bg-white/15"}`}
								aria-label={estimatedMode ? "Toggle estimated-H obstacle control" : "Toggle open-field obstacle control"}
								aria-pressed={controlEnabled}
							>
								<span className={`absolute top-1 size-4 rounded-full bg-white transition ${controlEnabled ? "left-6" : "left-1"}`} />
							</button>
						</div>

						<label className="mt-6 block text-xs text-white/55" htmlFor={`obstacle-frequency-${mode}`}>
							<span className="flex items-center justify-between">
								Frequency <output className="font-mono text-[#ffc247]">{frequency} Hz</output>
							</span>
							<input
								id={`obstacle-frequency-${mode}`}
								type="range"
								min="160"
								max="700"
								step="20"
								value={frequency}
								onChange={(event) => setFrequency(Number(event.target.value))}
								className="mt-3 w-full accent-[#2f6df6]"
							/>
						</label>

						<label className="mt-5 block text-xs text-white/55" htmlFor={`reflection-strength-${mode}`}>
							<span className="flex items-center justify-between">
								Reflection strength <output className="font-mono text-[#168bd2]">{reflectionStrength.toFixed(1)}×</output>
							</span>
							<input
								id={`reflection-strength-${mode}`}
								type="range"
								min="0"
								max="1.8"
								step="0.1"
								value={reflectionStrength}
								onChange={(event) => setReflectionStrength(Number(event.target.value))}
								className="mt-3 w-full accent-[#168bd2]"
							/>
						</label>

						<div className={`mt-5 rounded-xl border p-3.5 text-xs leading-5 text-white/42 ${estimatedMode ? "border-[#3bb9e8]/15 bg-[#3bb9e8]/[0.035]" : "border-[#ff3b24]/15 bg-[#ff3b24]/[0.035]"}`}>
							<p><span className="font-mono text-[#3bb9e8]">controller:</span> {estimatedMode ? "H ≈ direct + measured reflections" : "H = direct paths only"}</p>
							<p className="mt-1"><span className="font-mono text-[#ff3b24]">world:</span> H = direct + reflections + diffraction</p>
							<p className="mt-1"><span className="font-mono text-[#ff6a2a]">guard max:</span> {formatDb(controlEnabled ? actual.worstGuardDb : 0)}</p>
						</div>
					</section>

					<section className="rounded-2xl border border-[#168bd2]/15 bg-[#168bd2]/[0.035] p-5">
						<p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.17em] text-[#168bd2]">
							<ShieldAlert className="size-3.5" /> {estimatedMode ? "What changed" : "Why the ring mics are not enough"}
						</p>
						<p className="mt-3 text-xs leading-5 text-white/45">
							{estimatedMode ? "Four remote microphone readings anchor the camera-conditioned estimate near the listener. The deterministic optimizer can now phase the same speakers against the reflected arrivals." : "They observe reflected sound back at the array, which is valuable, but not the pressure inside a remote human’s bubble. One external error signal—or a camera-conditioned propagation model—is still needed to anchor that part of H."}
						</p>
					</section>
				</aside> : null}
			</section>
	);
}
