"use client";

import { CircleDot, RadioTower, UsersRound, Waves } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

type Complex = { re: number; im: number };
type Point = { x: number; y: number };

const WORLD = { width: 12, height: 8 };
const SOURCE = { x: 4, y: 4 };
const RING_RADIUS = 1.65;
const FREQUENCY = 140;
const SOUND_SPEED = 343;
const WAVE_NUMBER = (Math.PI * 2 * FREQUENCY) / SOUND_SPEED;

const PEOPLE: Point[] = [
	{ x: 8.35, y: 2.15 },
	{ x: 9.45, y: 4.05 },
	{ x: 8.45, y: 6.15 },
];

const STAGES = [
	{
		title: "People outside the site",
		eyebrow: "01 · listeners",
		body: "Start with the people we want to protect beyond the control ring.",
		color: "#edecee",
		duration: 2.4,
		icon: UsersRound,
	},
	{
		title: "The chainsaw radiates",
		eyebrow: "02 · source field",
		body: "A centered tone launches the same outgoing circular wave in every direction.",
		color: "#ffca85",
		duration: 3.2,
		icon: CircleDot,
	},
	{
		title: "The ideal ring answers",
		eyebrow: "03 · anti-field",
		body: "A continuous ring emits the exact opposite outgoing field beyond its boundary.",
		color: "#82e2ff",
		duration: 3.2,
		icon: RadioTower,
	},
	{
		title: "Outside cancels; inside remains",
		eyebrow: "04 · combined field",
		body: "The two fields sum to zero outside. The interior still contains strong structure.",
		color: "#61ffca",
		duration: 5.2,
		icon: Waves,
	},
] as const;

function cAdd(left: Complex, right: Complex): Complex {
	return { re: left.re + right.re, im: left.im + right.im };
}

function cScale(value: Complex, scale: number): Complex {
	return { re: value.re * scale, im: value.im * scale };
}

function sourcePhasor(radius: number): Complex {
	const safeRadius = Math.max(0.09, radius);
	const amplitude = 1 / Math.sqrt(safeRadius);
	const phase = -WAVE_NUMBER * safeRadius;
	return {
		re: amplitude * Math.cos(phase),
		im: amplitude * Math.sin(phase),
	};
}

function besselJ0(value: number) {
	let sum = 1;
	let term = 1;
	const quarterSquared = (value * value) / 4;
	for (let order = 1; order <= 18; order += 1) {
		term *= -quarterSquared / (order * order);
		sum += term;
	}
	return sum;
}

/**
 * Idealized continuous-ring anti-field. Outside the ring it is exactly the
 * negative outgoing source field. Inside, the regular J0 solution keeps the
 * pressure finite and exposes the non-quiet interior structure.
 */
function antiRingPhasor(radius: number): Complex {
	if (radius >= RING_RADIUS) return cScale(sourcePhasor(radius), -1);
	const boundary = sourcePhasor(RING_RADIUS);
	const denominator = besselJ0(WAVE_NUMBER * RING_RADIUS);
	const radialShape = besselJ0(WAVE_NUMBER * radius) / denominator;
	return cScale(boundary, -radialShape);
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

function stageAtElapsed(elapsed: number) {
	const total = STAGES.reduce((sum, stage) => sum + stage.duration, 0);
	let cursor = ((elapsed % total) + total) % total;
	for (let index = 0; index < STAGES.length; index += 1) {
		if (cursor < STAGES[index].duration) return index;
		cursor -= STAGES[index].duration;
	}
	return STAGES.length - 1;
}

function elapsedAtStage(stageIndex: number) {
	return STAGES.slice(0, stageIndex).reduce((sum, stage) => sum + stage.duration, 0);
}

export function ContinuousRingStory({
	running,
	active,
}: {
	running: boolean;
	active: boolean;
}) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const frameRef = useRef(0);
	const lastFrameRef = useRef(0);
	const elapsedRef = useRef(0);
	const stageRef = useRef(0);
	const [stage, setStage] = useState(0);

	const jumpToStage = useCallback((index: number) => {
		elapsedRef.current = elapsedAtStage(index) + 0.02;
		stageRef.current = index;
		setStage(index);
	}, []);

	const draw = useCallback(
		(timestamp: number) => {
			if (!active) return;
			if (timestamp - lastFrameRef.current < 32) return;
			const delta = Math.min(0.08, (timestamp - lastFrameRef.current) / 1000);
			lastFrameRef.current = timestamp;
			if (running) elapsedRef.current += delta;
			const nextStage = stageAtElapsed(elapsedRef.current);
			if (nextStage !== stageRef.current) {
				stageRef.current = nextStage;
				setStage(nextStage);
			}

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
			const currentStage = stageRef.current;
			const visualPhase = elapsedRef.current * Math.PI * 1.35;
			const cosTime = Math.cos(visualPhase);
			const sinTime = Math.sin(visualPhase);

			context.fillStyle = "#100f15";
			context.fillRect(0, 0, width, height);
			const step = Math.max(7, Math.round(6 * dpr));
			if (currentStage > 0) {
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
						const radius = Math.hypot(point.x - SOURCE.x, point.y - SOURCE.y);
						const source = sourcePhasor(radius);
						const anti = antiRingPhasor(radius);
						const phasor =
							currentStage === 1
								? source
								: currentStage === 2
									? anti
									: cAdd(source, anti);
						const instantaneous = phasor.re * cosTime - phasor.im * sinTime;
						const signedStrength = Math.tanh(instantaneous * 0.68);
						const mix = Math.abs(signedStrength) * 0.78;
						const positive = currentStage === 2 ? [130, 226, 255] : [162, 119, 255];
						const negative = currentStage === 2 ? [246, 148, 255] : [97, 255, 202];
						const target = signedStrength >= 0 ? positive : negative;
						context.fillStyle = `rgb(${Math.round(21 + (target[0] - 21) * mix)} ${Math.round(20 + (target[1] - 20) * mix)} ${Math.round(27 + (target[2] - 27) * mix)})`;
						context.fillRect(pixelX, pixelY, step + 1, step + 1);
					}
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

			const sourcePoint = toCanvas(SOURCE);
			const ringActive = currentStage >= 2;
			context.save();
			context.shadowColor = ringActive ? "#82e2ff" : "transparent";
			context.shadowBlur = ringActive ? 16 * dpr : 0;
			context.strokeStyle = ringActive ? "#82e2ff" : "rgba(130,226,255,.24)";
			context.lineWidth = (ringActive ? 4 : 2) * dpr;
			context.beginPath();
			context.arc(sourcePoint.x, sourcePoint.y, RING_RADIUS * transform.scale, 0, Math.PI * 2);
			context.stroke();
			context.restore();

			const sourceActive = currentStage === 1 || currentStage === 3;
			context.globalAlpha = sourceActive ? 1 : 0.32;
			context.fillStyle = "#ffca85";
			context.beginPath();
			context.arc(sourcePoint.x, sourcePoint.y, 10 * dpr, 0, Math.PI * 2);
			context.fill();
			context.strokeStyle = "rgba(255,202,133,.35)";
			context.lineWidth = 10 * dpr;
			context.stroke();
			context.globalAlpha = 1;

			const peoplePulse = currentStage === 0
				? 0.72 + Math.sin(elapsedRef.current * Math.PI * 3) * 0.28
				: 1;
			for (const person of PEOPLE) {
				const point = toCanvas(person);
				context.save();
				context.translate(point.x, point.y);
				context.scale(peoplePulse, peoplePulse);
				context.strokeStyle = currentStage === 3 ? "#61ffca" : currentStage === 0 ? "#edecee" : "rgba(237,236,238,.55)";
				context.lineWidth = 2.6 * dpr;
				context.beginPath();
				context.arc(0, -11 * dpr, 5.5 * dpr, 0, Math.PI * 2);
				context.moveTo(0, -5 * dpr);
				context.lineTo(0, 10 * dpr);
				context.moveTo(-9 * dpr, 1 * dpr);
				context.lineTo(9 * dpr, 1 * dpr);
				context.moveTo(0, 10 * dpr);
				context.lineTo(-8 * dpr, 20 * dpr);
				context.moveTo(0, 10 * dpr);
				context.lineTo(8 * dpr, 20 * dpr);
				context.stroke();
				context.restore();
			}

			const drawLabel = (
				text: string,
				anchor: Point,
				offsetX: number,
				offsetY: number,
				color: string,
			) => {
				const point = toCanvas(anchor);
				const fontSize = 15 * dpr;
				const paddingX = 9 * dpr;
				const boxHeight = 31 * dpr;
				context.font = `600 ${fontSize}px ui-monospace, monospace`;
				context.textBaseline = "middle";
				const textWidth = context.measureText(text).width;
				const boxWidth = textWidth + paddingX * 2;
				const x = Math.max(10 * dpr, Math.min(point.x + offsetX * dpr, width - boxWidth - 10 * dpr));
				const centerY = point.y + offsetY * dpr;
				context.strokeStyle = `${color}88`;
				context.lineWidth = 1.3 * dpr;
				context.beginPath();
				context.moveTo(point.x, point.y);
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
				context.textBaseline = "alphabetic";
			};

			if (currentStage === 0) {
				drawLabel("PEOPLE OUTSIDE THE SITE", PEOPLE[1], 34, -54, "#edecee");
			}
			if (currentStage === 1) {
				drawLabel("CHAINSAW SOURCE", SOURCE, 32, 0, "#ffca85");
			}
			if (currentStage === 2) {
				drawLabel("CONTINUOUS ANTI-SOUND RING", { x: SOURCE.x, y: SOURCE.y - RING_RADIUS }, 30, -28, "#82e2ff");
			}
			if (currentStage === 3) {
				drawLabel("OUTSIDE FIELD = 0", PEOPLE[1], 34, -54, "#61ffca");
				drawLabel("FIELD REMAINS INSIDE", SOURCE, 32, 0, "#ffca85");
			}
		},
		[active, running],
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

	const activeStage = STAGES[stage];
	const ActiveIcon = activeStage.icon;

	return (
		<section className="mx-auto grid max-w-[1500px] gap-4 p-4 sm:p-6 lg:grid-cols-[minmax(0,1fr)_340px]">
			<div className="space-y-3">
				<div className="relative aspect-[3/2] min-h-[420px] max-h-[720px] overflow-hidden rounded-2xl border border-white/10 bg-[#100f15] shadow-2xl shadow-black/30">
					<canvas
						ref={canvasRef}
						className="absolute inset-0 h-full w-full"
						role="img"
						aria-label="Animated ideal continuous-ring sound cancellation story"
					/>
					<div className="pointer-events-none absolute left-4 top-4 flex items-center gap-2 rounded-lg border border-white/10 bg-[#15141b]/88 px-3 py-2 backdrop-blur">
						<span className="font-mono text-[10px] uppercase tracking-[0.16em] text-white/45">
							Ideal model
						</span>
						<span className="h-3 w-px bg-white/10" />
						<span className="font-mono text-[10px] text-[#ffca85]">{FREQUENCY} Hz</span>
					</div>
					<div className="pointer-events-none absolute bottom-4 right-4 rounded-lg border border-white/10 bg-[#15141b]/88 px-3 py-2 font-mono text-[10px] uppercase tracking-[0.13em] text-white/45 backdrop-blur">
						Stage {stage + 1} / {STAGES.length}
					</div>
				</div>
				<p className="px-1 text-xs leading-5 text-white/40">
					An ideal centered single tone and infinitely dense ring: exact outside cancellation, nonzero interior field.
				</p>
			</div>

			<aside className="space-y-4">
				<section className="rounded-2xl border border-white/10 bg-[#1b1924] p-5">
					<div className="flex items-center justify-between gap-3">
						<p className="font-mono text-[10px] uppercase tracking-[0.17em]" style={{ color: activeStage.color }}>
							{activeStage.eyebrow}
						</p>
						<ActiveIcon className="size-4" style={{ color: activeStage.color }} />
					</div>
					<h2 className="mt-3 text-xl font-semibold leading-tight">{activeStage.title}</h2>
					<p className="mt-3 text-sm leading-6 text-white/48">{activeStage.body}</p>
					<div className="mt-6 grid grid-cols-4 gap-1.5">
						{STAGES.map((item, index) => (
							<button
								key={item.title}
								type="button"
								onClick={() => jumpToStage(index)}
								className={`h-2 rounded-full transition ${index === stage ? "bg-[#61ffca]" : "bg-white/10 hover:bg-white/20"}`}
								aria-label={`Show stage ${index + 1}: ${item.title}`}
								aria-pressed={index === stage}
							/>
						))}
					</div>
				</section>

				<section className="rounded-2xl border border-[#61ffca]/15 bg-[#61ffca]/[0.035] p-5">
					<p className="font-mono text-[10px] uppercase tracking-[0.17em] text-[#61ffca]">
						What the final frame means
					</p>
					<div className="mt-4 space-y-3 text-sm leading-6 text-white/48">
						<p><span className="text-[#a277ff]">Source</span> + <span className="text-[#82e2ff]">anti-ring</span> = <span className="text-[#61ffca]">silence outside</span>.</p>
						<p>The ring still supports a regular standing field inside, so cancellation is not the same as destroying all acoustic energy.</p>
					</div>
				</section>

				<section className="rounded-2xl border border-white/10 bg-[#1b1924] p-5">
					<p className="font-mono text-[10px] uppercase tracking-[0.17em] text-white/35">Ideal assumptions</p>
					<ul className="mt-3 space-y-2 text-xs leading-5 text-white/38">
						<li>• Perfect radial symmetry</li>
						<li>• One matched frequency</li>
						<li>• Continuous, zero-delay ring</li>
						<li>• No wind or reflections</li>
					</ul>
				</section>
			</aside>
		</section>
	);
}
