"use client";

import {
	BrainCircuit,
	Building2,
	Cpu,
	Gauge,
	Orbit,
	Radio,
	TriangleAlert,
	UsersRound,
} from "lucide-react";
import {
	type PointerEvent as ReactPointerEvent,
	useCallback,
	useEffect,
	useMemo,
	useRef,
	useState,
} from "react";

type Complex = { re: number; im: number };
type Point = { x: number; y: number };
type Human = Point & { id: number; label: string; color: string };
type WeightedPoint = Point & { weight: number };

type MultiController = {
	waveNumber: number;
	weights: Complex[];
	targetGroups: WeightedPoint[][];
	guardGroups: Point[][];
	bubbleDbs: number[];
	centerDbs: number[];
	aggregateDb: number;
	worstPersonDb: number;
	worstGuardDb: number;
	speakerEffort: number;
};

const WORLD = { width: 12, height: 8 };
const SOURCE = { x: 4, y: 4 };
const SOUND_SPEED = 343;
const MAX_SPEAKER_STRENGTH = 4;

const HUMAN_BASES: Human[] = [
	{ id: 1, label: "A", color: "#61ffca", x: 8.1, y: 2.2 },
	{ id: 2, label: "B", color: "#f694ff", x: 9.35, y: 4.05 },
	{ id: 3, label: "C", color: "#82e2ff", x: 8.2, y: 5.9 },
];

const AURA = {
	background: [21, 20, 27] as const,
	purple: [162, 119, 255] as const,
	green: [97, 255, 202] as const,
	orange: "#ffca85",
	blue: "#82e2ff",
	red: "#ff6767",
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

function green(from: Point, to: Point, waveNumber: number): Complex {
	const radius = Math.max(0.09, distance(from, to));
	const amplitude = 1 / Math.sqrt(radius);
	const phase = -waveNumber * radius;
	return {
		re: amplitude * Math.cos(phase),
		im: amplitude * Math.sin(phase),
	};
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

function pointsOnDisk(center: Human, radius: number) {
	const points: WeightedPoint[] = [{ x: center.x, y: center.y, weight: 2.4 }];
	const rings = [
		{ radius: radius * 0.35, count: 8, weight: 1.6 },
		{ radius: radius * 0.7, count: 12, weight: 1.15 },
		{ radius, count: 16, weight: 0.85 },
	];
	for (const ring of rings) {
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

function pointsOnGuardRing(center: Human, bubbleRadius: number) {
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

function pressureAt(
	point: Point,
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

function calculateMultiHumanController({
	humans,
	speakers,
	frequency,
	bubbleRadius,
}: {
	humans: Human[];
	speakers: Point[];
	frequency: number;
	bubbleRadius: number;
}): MultiController {
	const waveNumber = (Math.PI * 2 * frequency) / SOUND_SPEED;
	const targetGroups = humans.map((human) => pointsOnDisk(human, bubbleRadius));
	const guardGroups = humans.map((human) => pointsOnGuardRing(human, bubbleRadius));
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
			const column = green(speakers[index], point, waveNumber);
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

	for (const points of targetGroups) {
		for (const point of points) {
			addConstraint(
				point,
				cScale(green(SOURCE, point, waveNumber), -1),
				point.weight,
			);
		}
	}
	for (const points of guardGroups) {
		for (const point of points) addConstraint(point, { re: 0, im: 0 }, 0.05);
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

	let totalSourceEnergy = 0;
	let totalControlledEnergy = 0;
	const bubbleDbs = targetGroups.map((points) => {
		let sourceEnergy = 0;
		let controlledEnergy = 0;
		for (const point of points) {
			const source = green(SOURCE, point, waveNumber);
			const controlled = pressureAt(point, waveNumber, speakers, weights);
			sourceEnergy += point.weight * magnitude(source) ** 2;
			controlledEnergy += point.weight * magnitude(controlled) ** 2;
		}
		totalSourceEnergy += sourceEnergy;
		totalControlledEnergy += controlledEnergy;
		return 10 * Math.log10(Math.max(1e-12, controlledEnergy / sourceEnergy));
	});
	const centerDbs = humans.map((human) => {
		const source = green(SOURCE, human, waveNumber);
		const controlled = pressureAt(human, waveNumber, speakers, weights);
		return 20 * Math.log10(Math.max(1e-7, magnitude(controlled) / magnitude(source)));
	});
	let worstGuardDb = -Infinity;
	for (const points of guardGroups) {
		for (const point of points) {
			const source = green(SOURCE, point, waveNumber);
			const controlled = pressureAt(point, waveNumber, speakers, weights);
			worstGuardDb = Math.max(
				worstGuardDb,
				20 * Math.log10(Math.max(1e-7, magnitude(controlled) / magnitude(source))),
			);
		}
	}

	return {
		waveNumber,
		weights,
		targetGroups,
		guardGroups,
		bubbleDbs,
		centerDbs,
		aggregateDb:
			10 * Math.log10(Math.max(1e-12, totalControlledEnergy / totalSourceEnergy)),
		worstPersonDb: Math.max(...bubbleDbs),
		worstGuardDb: Number.isFinite(worstGuardDb) ? worstGuardDb : 0,
		speakerEffort: weights.reduce(
			(sum, weight) => sum + magnitude(weight) ** 2,
			0,
		),
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

function resetHumans() {
	return HUMAN_BASES.map((human) => ({ ...human }));
}

export function MultiHumanLab({
	running,
	active,
}: {
	running: boolean;
	active: boolean;
}) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const frameRef = useRef(0);
	const lastFrameRef = useRef(0);
	const timeRef = useRef(0);
	const orbitFrameRef = useRef(0);
	const orbitLastFrameRef = useRef(0);
	const orbitPhaseRef = useRef(0);
	const dragHumanRef = useRef<number | null>(null);
	const [humans, setHumans] = useState<Human[]>(resetHumans);
	const [frequency, setFrequency] = useState(280);
	const [speakerCount, setSpeakerCount] = useState(16);
	const [bubbleRadius, setBubbleRadius] = useState(0.45);
	const [controlEnabled, setControlEnabled] = useState(true);
	const [autoTrack, setAutoTrack] = useState(true);
	const speakers = useMemo(() => buildRing(speakerCount), [speakerCount]);
	const sensors = useMemo(
		() => buildRing(speakerCount, 1.55, Math.PI / speakerCount),
		[speakerCount],
	);
	const controller = useMemo(
		() =>
			calculateMultiHumanController({
				humans,
				speakers,
				frequency,
				bubbleRadius,
			}),
		[humans, speakers, frequency, bubbleRadius],
	);
	const shownAggregateDb = controlEnabled ? controller.aggregateDb : 0;
	const shownWorstPersonDb = controlEnabled ? controller.worstPersonDb : 0;
	const shownBubbleDbs = controlEnabled
		? controller.bubbleDbs
		: controller.bubbleDbs.map(() => 0);

	useEffect(() => {
		if (!active || !running || !autoTrack) {
			orbitLastFrameRef.current = 0;
			return;
		}
		const animate = (timestamp: number) => {
			orbitFrameRef.current = requestAnimationFrame(animate);
			if (orbitLastFrameRef.current === 0) {
				orbitLastFrameRef.current = timestamp;
				return;
			}
			if (timestamp - orbitLastFrameRef.current < 72) return;
			const delta = Math.min(0.12, (timestamp - orbitLastFrameRef.current) / 1000);
			orbitLastFrameRef.current = timestamp;
			orbitPhaseRef.current =
				(orbitPhaseRef.current + delta * 0.24) % (Math.PI * 2);
			setHumans(
				HUMAN_BASES.map((human, index) => {
					const phase = orbitPhaseRef.current + index * 2.08;
					const xRadius = [0.34, 0.43, 0.36][index];
					const yRadius = [0.24, 0.31, 0.26][index];
					return {
						...human,
						x: human.x + Math.cos(phase) * xRadius,
						y: human.y + Math.sin(phase) * yRadius,
					};
				}),
			);
		};
		orbitFrameRef.current = requestAnimationFrame(animate);
		return () => cancelAnimationFrame(orbitFrameRef.current);
	}, [active, autoTrack, running]);

	const draw = useCallback(
		(timestamp: number) => {
			if (!active) return;
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
			const toCanvas = (point: Point) => ({
				x: transform.offsetX + point.x * transform.scale,
				y: transform.offsetY + point.y * transform.scale,
			});

			context.fillStyle = "#100f15";
			context.fillRect(0, 0, width, height);
			const step = Math.max(7, Math.round(6 * dpr));
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
						? pressureAt(point, controller.waveNumber, speakers, controller.weights)
						: green(SOURCE, point, controller.waveNumber);
					const instantaneous = phasor.re * cosTime - phasor.im * sinTime;
					const signedStrength = Math.tanh(instantaneous * 0.72);
					const mix = Math.abs(signedStrength) * 0.76;
					const targetColor = signedStrength >= 0 ? AURA.purple : AURA.green;
					context.fillStyle = `rgb(${Math.round(AURA.background[0] + (targetColor[0] - AURA.background[0]) * mix)} ${Math.round(AURA.background[1] + (targetColor[1] - AURA.background[1]) * mix)} ${Math.round(AURA.background[2] + (targetColor[2] - AURA.background[2]) * mix)})`;
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

			for (let index = 0; index < sensors.length; index += 1) {
				const sensor = toCanvas(sensors[index]);
				const speaker = toCanvas(speakers[index]);
				context.strokeStyle = "rgba(130,226,255,.16)";
				context.beginPath();
				context.moveTo(sensor.x, sensor.y);
				context.lineTo(speaker.x, speaker.y);
				context.stroke();
				context.save();
				context.translate(sensor.x, sensor.y);
				context.rotate(Math.PI / 4);
				context.fillStyle = "#15141b";
				context.strokeStyle = AURA.blue;
				context.lineWidth = 1.35 * dpr;
				context.fillRect(-4 * dpr, -4 * dpr, 8 * dpr, 8 * dpr);
				context.strokeRect(-4 * dpr, -4 * dpr, 8 * dpr, 8 * dpr);
				context.restore();
			}
			for (const speaker of speakers) {
				const point = toCanvas(speaker);
				context.fillStyle = "#15141b";
				context.strokeStyle = controlEnabled ? "#a277ff" : "rgba(162,119,255,.4)";
				context.lineWidth = 1.6 * dpr;
				context.beginPath();
				context.arc(point.x, point.y, 6.8 * dpr, 0, Math.PI * 2);
				context.fill();
				context.stroke();
			}

			const source = toCanvas(SOURCE);
			context.fillStyle = AURA.orange;
			context.beginPath();
			context.arc(source.x, source.y, 7 * dpr, 0, Math.PI * 2);
			context.fill();
			context.strokeStyle = "rgba(255,202,133,.35)";
			context.lineWidth = 8 * dpr;
			context.stroke();

			const drawCallout = (
				anchor: { x: number; y: number },
				text: string,
				color: string,
				offsetY: number,
			) => {
				const fontSize = 14 * dpr;
				const paddingX = 9 * dpr;
				const boxHeight = 30 * dpr;
				context.font = `600 ${fontSize}px ui-monospace, monospace`;
				context.textBaseline = "middle";
				const boxWidth = context.measureText(text).width + paddingX * 2;
				const x = Math.min(anchor.x + 32 * dpr, width - boxWidth - 10 * dpr);
				const centerY = anchor.y + offsetY * dpr;
				context.strokeStyle = `${color}88`;
				context.lineWidth = 1.25 * dpr;
				context.beginPath();
				context.moveTo(anchor.x + 10 * dpr, anchor.y);
				context.lineTo(x - 4 * dpr, centerY);
				context.stroke();
				context.fillStyle = "rgba(16,15,21,.92)";
				context.beginPath();
				context.roundRect(x, centerY - boxHeight / 2, boxWidth, boxHeight, 7 * dpr);
				context.fill();
				context.strokeStyle = `${color}66`;
				context.stroke();
				context.fillStyle = color;
				context.fillText(text, x + paddingX, centerY + dpr);
			};

			for (let index = 0; index < humans.length; index += 1) {
				const human = humans[index];
				const point = toCanvas(human);
				context.save();
				context.fillStyle = `${human.color}0e`;
				context.strokeStyle = human.color;
				context.lineWidth = 1.65 * dpr;
				context.beginPath();
				context.arc(point.x, point.y, bubbleRadius * transform.scale, 0, Math.PI * 2);
				context.fill();
				context.stroke();
				context.setLineDash([3 * dpr, 8 * dpr]);
				context.strokeStyle = "rgba(255,103,103,.24)";
				context.beginPath();
				context.arc(
					point.x,
					point.y,
					Math.max(0.85, bubbleRadius * 1.9) * transform.scale,
					0,
					Math.PI * 2,
				);
				context.stroke();
				context.setLineDash([]);
				for (let sample = 0; sample < 12; sample += 1) {
					const angle = (sample / 12) * Math.PI * 2;
					context.fillStyle = `${human.color}88`;
					context.beginPath();
					context.arc(
						point.x + Math.cos(angle) * bubbleRadius * transform.scale,
						point.y + Math.sin(angle) * bubbleRadius * transform.scale,
						1.15 * dpr,
						0,
						Math.PI * 2,
					);
					context.fill();
				}
				context.fillStyle = "#15141b";
				context.strokeStyle = human.color;
				context.lineWidth = 2 * dpr;
				context.beginPath();
				context.arc(point.x, point.y, 10 * dpr, 0, Math.PI * 2);
				context.fill();
				context.stroke();
				context.beginPath();
				context.moveTo(point.x - 14 * dpr, point.y);
				context.lineTo(point.x + 14 * dpr, point.y);
				context.moveTo(point.x, point.y - 14 * dpr);
				context.lineTo(point.x, point.y + 14 * dpr);
				context.stroke();
				context.restore();

				drawCallout(
					point,
					`HUMAN ${human.label}  ${formatDb(controlEnabled ? controller.bubbleDbs[index] : 0)}`,
					human.color,
					index === 0 ? -34 : index === 2 ? 34 : -42,
				);
			}

			drawCallout(source, "SOURCE", AURA.orange, 0);
			context.textBaseline = "alphabetic";
		},
		[
			active,
			bubbleRadius,
			controlEnabled,
			controller,
			humans,
			running,
			sensors,
			speakers,
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

	const pointerToWorld = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
		const canvas = canvasRef.current;
		if (!canvas) return null;
		const rect = canvas.getBoundingClientRect();
		const transform = getTransform(rect.width, rect.height);
		return {
			x: Math.max(
				6.2,
				Math.min(WORLD.width - 0.2, (event.clientX - rect.left - transform.offsetX) / transform.scale),
			),
			y: Math.max(
				0.2,
				Math.min(WORLD.height - 0.2, (event.clientY - rect.top - transform.offsetY) / transform.scale),
			),
		};
	}, []);

	const handlePointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
		const point = pointerToWorld(event);
		if (!point) return;
		const humanIndex = humans.findIndex((human) => distance(human, point) < 0.45);
		if (humanIndex < 0) return;
		setAutoTrack(false);
		dragHumanRef.current = humanIndex;
		event.currentTarget.setPointerCapture(event.pointerId);
	};

	const handlePointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
		const point = pointerToWorld(event);
		const humanIndex = dragHumanRef.current;
		if (!point || humanIndex === null) return;
		setHumans((current) =>
			current.map((human, index) =>
				index === humanIndex ? { ...human, ...point } : human,
			),
		);
	};

	const handlePointerUp = () => {
		dragHumanRef.current = null;
	};

	return (
		<>
			<section className="mx-auto grid max-w-[1500px] gap-4 p-4 sm:p-6 lg:grid-cols-[minmax(0,1fr)_340px]">
				<div className="space-y-3">
					<div className="relative aspect-[3/2] min-h-[420px] max-h-[720px] overflow-hidden rounded-2xl border border-white/10 bg-[#100f15] shadow-2xl shadow-black/30">
						<canvas
							ref={canvasRef}
							data-testid="multi-human-canvas"
							className="absolute inset-0 h-full w-full touch-none cursor-grab active:cursor-grabbing"
							onPointerDown={handlePointerDown}
							onPointerMove={handlePointerMove}
							onPointerUp={handlePointerUp}
							onPointerCancel={handlePointerUp}
							role="img"
							aria-label="Pressure field optimized around three moving people"
						/>
						<div className="pointer-events-none absolute left-4 top-4 flex flex-wrap items-center gap-2">
							<span className="rounded-md border border-white/10 bg-[#15141b]/88 px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wider text-white/50 backdrop-blur">
								Shared controller
							</span>
							<span className="rounded-md border border-[#61ffca]/20 bg-[#15141b]/88 px-2.5 py-1.5 font-mono text-[10px] text-[#61ffca] backdrop-blur">
								3-bubble avg {formatDb(shownAggregateDb)}
							</span>
							<span className="rounded-md border border-[#f694ff]/20 bg-[#15141b]/88 px-2.5 py-1.5 font-mono text-[10px] text-[#f694ff] backdrop-blur">
								Worst person {formatDb(shownWorstPersonDb)}
							</span>
						</div>
						<div className="pointer-events-none absolute bottom-4 right-4 hidden rounded-md border border-white/10 bg-[#15141b]/88 px-2.5 py-2 font-mono text-[9px] uppercase tracking-wider text-white/40 backdrop-blur sm:block">
							{speakerCount} complex controls · 3 moving targets
						</div>
					</div>
					<p className="px-1 text-xs leading-5 text-white/40">
						All three bubbles enter one least-squares objective. The controller is recomputed as the people move; grab any crosshair to take over manually.
					</p>
				</div>

				<aside className="space-y-4">
					<section className="overflow-hidden rounded-2xl border border-[#61ffca]/20 bg-[#1b1924]">
						<div className="flex items-start justify-between gap-4 p-5 pb-3">
							<div>
								<p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.17em] text-[#61ffca]">
									<Gauge className="size-3.5" /> Three-person result
								</p>
								<p className="mt-2 text-xs leading-5 text-white/45">
									Energy average across three {bubbleRadius.toFixed(2)} m bubbles.
								</p>
							</div>
							<span className="font-mono text-xl font-semibold text-[#61ffca]">
								{formatDb(shownAggregateDb)}
							</span>
						</div>
						<div className="border-t border-white/8">
							{humans.map((human, index) => (
								<div
									key={human.id}
									className="flex items-center justify-between border-b border-white/8 px-5 py-3 last:border-b-0"
								>
									<span className="flex items-center gap-2 text-xs text-white/55">
										<span className="size-2 rounded-full" style={{ background: human.color }} />
										Human {human.label}
									</span>
									<span className="font-mono text-sm" style={{ color: human.color }}>
										{formatDb(shownBubbleDbs[index])}
									</span>
								</div>
							))}
						</div>
						<div className="grid grid-cols-2 border-t border-white/8">
							<div className="border-r border-white/8 px-5 py-3">
								<p className="text-[9px] uppercase tracking-wider text-white/30">Guard max</p>
								<p className="mt-1 font-mono text-xs text-[#ff6767]">
									{formatDb(controlEnabled ? controller.worstGuardDb : 0)}
								</p>
							</div>
							<div className="px-5 py-3">
								<p className="text-[9px] uppercase tracking-wider text-white/30">Speaker effort</p>
								<p className="mt-1 font-mono text-xs text-[#a277ff]">
									{controller.speakerEffort.toFixed(2)}
								</p>
							</div>
						</div>
					</section>

					<section className="rounded-2xl border border-white/10 bg-[#1b1924] p-5">
						<div className="flex items-center justify-between gap-4">
							<div>
								<p className="font-mono text-[10px] uppercase tracking-[0.17em] text-[#a277ff]">
									Shared optimizer
								</p>
								<p className="mt-1 text-sm font-medium">One solution, three regions</p>
							</div>
							<button
								type="button"
								onClick={() => setControlEnabled((value) => !value)}
								className={`relative h-6 w-11 rounded-full transition ${controlEnabled ? "bg-[#a277ff]" : "bg-white/15"}`}
								aria-label="Toggle multi-human control speakers"
								aria-pressed={controlEnabled}
							>
								<span className={`absolute top-1 size-4 rounded-full bg-white transition ${controlEnabled ? "left-6" : "left-1"}`} />
							</button>
						</div>

						<button
							type="button"
							onClick={() => setAutoTrack((value) => !value)}
							className={`mt-5 flex w-full items-center justify-between rounded-lg border px-3 py-2.5 text-xs transition ${
								autoTrack
									? "border-[#61ffca]/25 bg-[#61ffca]/[0.055] text-[#61ffca]"
									: "border-white/10 text-white/45 hover:border-white/20"
							}`}
							aria-pressed={autoTrack}
						>
							<span className="flex items-center gap-2">
								<Orbit className="size-3.5" /> {autoTrack ? "Three paths moving" : "Manual positions"}
							</span>
							<span className="font-mono text-[9px] uppercase">{autoTrack ? "on" : "off"}</span>
						</button>

						<label className="mt-5 block text-xs text-white/55" htmlFor="multi-frequency">
							<span className="flex items-center justify-between">
								Frequency <output className="font-mono text-[#ffca85]">{frequency} Hz</output>
							</span>
							<input
								id="multi-frequency"
								type="range"
								min="100"
								max="700"
								step="20"
								value={frequency}
								onChange={(event) => setFrequency(Number(event.target.value))}
								className="mt-3 w-full accent-[#a277ff]"
							/>
						</label>

						<label className="mt-5 block text-xs text-white/55" htmlFor="multi-bubble-radius">
							<span className="flex items-center justify-between">
								Bubble radius <output className="font-mono text-[#61ffca]">{bubbleRadius.toFixed(2)} m</output>
							</span>
							<input
								id="multi-bubble-radius"
								type="range"
								min="0.25"
								max="0.7"
								step="0.05"
								value={bubbleRadius}
								onChange={(event) => setBubbleRadius(Number(event.target.value))}
								className="mt-3 w-full accent-[#61ffca]"
							/>
						</label>

						<div className="mt-5">
							<div className="flex items-center justify-between">
								<span className="text-xs text-white/55">Speaker controls</span>
								<Radio className="size-3.5 text-[#82e2ff]" />
							</div>
							<div className="mt-3 grid grid-cols-3 gap-2">
								{[8, 16, 24].map((count) => (
									<button
										key={count}
										type="button"
										onClick={() => setSpeakerCount(count)}
										className={`rounded-lg border px-2 py-2 font-mono text-[11px] transition ${
											speakerCount === count
												? "border-[#82e2ff]/35 bg-[#82e2ff]/10 text-[#82e2ff]"
												: "border-white/10 text-white/45 hover:border-white/20"
										}`}
									>
										{count}
									</button>
								))}
							</div>
						</div>
					</section>

					<section className="rounded-2xl border border-[#f694ff]/15 bg-[#f694ff]/[0.035] p-5">
						<p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.17em] text-[#f694ff]">
							<UsersRound className="size-3.5" /> Generalizing to N
						</p>
						<p className="mt-3 text-xs leading-5 text-white/45">
							Each new person adds a sampled quiet region, but not necessarily independent constraints. At low frequency the field has few spatial modes; at high frequency the required speaker count grows quickly.
						</p>
						<div className="mt-4 grid grid-cols-2 gap-2 font-mono text-[10px]">
							<div className="rounded-lg border border-white/8 bg-[#15141b] p-2.5 text-white/45">
								<span className="block text-[#82e2ff]">{speakerCount} complex</span> controls
							</div>
							<div className="rounded-lg border border-white/8 bg-[#15141b] p-2.5 text-white/45">
								<span className="block text-[#61ffca]">111 samples</span> across 3 bubbles
							</div>
						</div>
					</section>
				</aside>
			</section>

			<section className="border-t border-white/10 px-5 py-14 sm:px-8 sm:py-18">
				<div className="mx-auto max-w-[1200px]">
					<div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-[#ffca85]">
						<Building2 className="size-3.5" /> The next hard case · buildings and obstacles
					</div>
					<h2 className="mt-4 max-w-3xl text-3xl font-semibold tracking-tight sm:text-4xl">
						Obstacles change the transfer matrix—not the optimization problem.
					</h2>
					<p className="mt-4 max-w-4xl text-base leading-7 text-white/55">
						Replace the free-field Green function with an environment-aware impulse response from every source and speaker to every protected point. Reflections and diffraction can be modeled deterministically, measured in place, or learned as a fast residual correction.
					</p>
					<div className="mt-7 grid gap-4 rounded-2xl border border-[#61ffca]/15 bg-[#61ffca]/[0.035] p-5 sm:p-6 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
						<div>
							<p className="font-mono text-[10px] uppercase tracking-[0.17em] text-[#61ffca]">
								Deterministic verdict
							</p>
							<p className="mt-3 text-sm leading-6 text-white/55">
								If the environment transfer matrix is accurate and acoustics remain linear, the optimal narrowband speaker weights are still a small convex solve. A neural network is not needed to find the optimum.
							</p>
						</div>
						<div className="rounded-xl border border-white/8 bg-[#15141b] p-4 font-mono text-xs leading-7 text-white/60">
							<span className="text-[#edecee]">p(f)</span> = d(f) + <span className="text-[#82e2ff]">H<sub>environment</sub>(f)</span>w(f)
							<br />
							<span className="text-[#61ffca]">w*</span> = arg min ‖p<sub>bubbles</sub>‖² + λ‖w‖²
						</div>
					</div>

					<div className="mt-9 grid gap-4 md:grid-cols-3">
						<div className="rounded-2xl border border-[#82e2ff]/15 bg-[#100f15] p-5 sm:p-6">
							<Cpu className="size-5 text-[#82e2ff]" />
							<h3 className="mt-4 text-base font-semibold">1 · Deterministic first</h3>
							<p className="mt-3 text-sm leading-6 text-white/48">
								Use a wave solver near the array and people, geometric acoustics for distant reflections, then solve the same regularized least-squares controller.
							</p>
						</div>
						<div className="rounded-2xl border border-[#ff6767]/15 bg-[#100f15] p-5 sm:p-6">
							<TriangleAlert className="size-5 text-[#ff6767]" />
							<h3 className="mt-4 text-base font-semibold">2 · Calibration is the bottleneck</h3>
							<p className="mt-3 text-sm leading-6 text-white/48">
								Unknown wall impedance, open windows, wind, temperature gradients, and moving objects create more error than the optimizer itself.
							</p>
						</div>
						<div className="rounded-2xl border border-[#f694ff]/15 bg-[#100f15] p-5 sm:p-6">
							<BrainCircuit className="size-5 text-[#f694ff]" />
							<h3 className="mt-4 text-base font-semibold">3 · Learn the residual</h3>
							<p className="mt-3 text-sm leading-6 text-white/48">
								A learned surrogate can accelerate repeated solves or correct model mismatch. It should predict transfer functions—not replace the safety-constrained optimizer.
							</p>
						</div>
					</div>
				</div>
			</section>
		</>
	);
}
