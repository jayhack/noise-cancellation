"use client";

import { Camera, Check, Mic2, Plane, Radio, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";

const PATH_POINTS = [
	{ x: 760, y: 306, level: "−12.8 dB", phase: "+38°" },
	{ x: 846, y: 248, level: "−16.4 dB", phase: "−71°" },
	{ x: 930, y: 322, level: "−10.1 dB", phase: "+126°" },
	{ x: 950, y: 438, level: "−14.7 dB", phase: "+19°" },
] as const;

function ringPoints(count: number, radius: number, offset = 0) {
	return Array.from({ length: count }, (_, index) => {
		const angle = (index / count) * Math.PI * 2 + offset;
		return {
			x: Math.round((400 + Math.cos(angle) * radius) * 1000) / 1000,
			y: Math.round((400 + Math.sin(angle) * radius) * 1000) / 1000,
		};
	});
}

export function HEstimationLab({
	running,
	active,
	showControls,
}: {
	running: boolean;
	active: boolean;
	showControls: boolean;
}) {
	const [activePoint, setActivePoint] = useState(0);

	useEffect(() => {
		if (!active || !running) return;
		const interval = window.setInterval(() => {
			setActivePoint((point) => (point + 1) % PATH_POINTS.length);
		}, 1250);
		return () => window.clearInterval(interval);
	}, [active, running]);

	const current = PATH_POINTS[activePoint];
	const speakers = ringPoints(8, 112);
	const sensors = ringPoints(8, 112, Math.PI / 8);

	return (
		<section className={`mx-auto grid max-w-[1500px] gap-4 p-4 sm:p-6 ${showControls ? "lg:grid-cols-[minmax(0,1fr)_340px]" : ""}`}>
				<div className="space-y-3">
					<div className="relative aspect-[3/2] min-h-[420px] max-h-[720px] overflow-hidden rounded-2xl border border-[#168bd2]/20 bg-[#070a0d] shadow-2xl shadow-black/30">
						<svg
							viewBox="0 0 1200 800"
							className="absolute inset-0 size-full"
							role="img"
							aria-label="Drone carrying a microphone through four acoustic sampling points"
							data-testid="h-estimation-path"
						>
							<defs>
								<pattern id="estimation-grid" width="80" height="80" patternUnits="userSpaceOnUse">
									<path d="M 80 0 L 0 0 0 80" fill="none" stroke="rgba(242,238,228,.075)" strokeWidth="1" />
								</pattern>
								<filter id="reading-glow" x="-100%" y="-100%" width="300%" height="300%">
									<feGaussianBlur stdDeviation="8" result="blur" />
									<feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
								</filter>
							</defs>
							<rect width="1200" height="800" fill="#070a0d" />
							<rect x="30" y="28" width="1140" height="744" rx="18" fill="url(#estimation-grid)" />

							{[88, 170, 252].map((radius, index) => (
								<circle
									key={radius}
									cx="400"
									cy="400"
									r={radius}
									fill="none"
									stroke={index % 2 === 0 ? "rgba(47,109,246,.18)" : "rgba(59,185,232,.14)"}
									strokeWidth="18"
								/>
							))}

							<g>
								<rect x="605" y="135" width="115" height="115" fill="#121822" stroke="#168bd2" strokeOpacity=".58" strokeWidth="3" />
								<path d="M605 232 L702 135 M622 250 L720 152 M605 198 L668 135" stroke="#168bd2" strokeOpacity=".12" strokeWidth="7" />
								<text x="662" y="198" textAnchor="middle" fill="#168bd2" fillOpacity=".72" fontFamily="ui-monospace, monospace" fontSize="13">BUILDING A</text>
								<rect x="665" y="505" width="115" height="115" fill="#121822" stroke="#168bd2" strokeOpacity=".58" strokeWidth="3" />
								<path d="M665 602 L762 505 M682 620 L780 522 M665 568 L728 505" stroke="#168bd2" strokeOpacity=".12" strokeWidth="7" />
								<text x="722" y="568" textAnchor="middle" fill="#168bd2" fillOpacity=".72" fontFamily="ui-monospace, monospace" fontSize="13">BUILDING B</text>
							</g>

							{speakers.map((point, index) => (
								<g key={`speaker-${index}`}>
									<circle cx={point.x} cy={point.y} r="12" fill="#0b0e12" stroke="#2f6df6" strokeWidth="3" />
									<circle cx={point.x + 3} cy={point.y} r="3" fill="#2f6df6" />
								</g>
							))}
							{sensors.map((point, index) => (
								<rect
									key={`sensor-${index}`}
									x={point.x - 8}
									y={point.y - 8}
									width="16"
									height="16"
									transform={`rotate(45 ${point.x} ${point.y})`}
									fill="#0b0e12"
									stroke="#168bd2"
									strokeWidth="3"
								/>
							))}
							<circle cx="400" cy="400" r="13" fill="#ffc247" />
							<circle cx="400" cy="400" r="24" fill="none" stroke="#ffc247" strokeOpacity=".3" strokeWidth="11" />
							<rect x="434" y="377" width="118" height="46" rx="10" fill="#070a0d" stroke="#ffc247" strokeOpacity=".6" />
							<text x="493" y="406" textAnchor="middle" fill="#ffc247" fontFamily="ui-monospace, monospace" fontWeight="700" fontSize="19">SOURCE</text>

							<path
								d="M 760 306 C 792 270, 818 252, 846 248 C 886 244, 914 278, 930 322 C 945 363, 951 403, 950 438"
								fill="none"
								stroke="#3bb9e8"
								strokeOpacity=".55"
								strokeWidth="4"
								strokeDasharray="9 11"
							/>
							{PATH_POINTS.map((point, index) => {
								const captured = index <= activePoint;
								const selected = index === activePoint;
								return (
									<g key={`${point.x}-${point.y}`}>
										{selected ? (
											<circle cx={point.x} cy={point.y} r="31" fill="none" stroke="#3bb9e8" strokeOpacity=".28" strokeWidth="12" className="animate-pulse" />
										) : null}
										<circle cx={point.x} cy={point.y} r="13" fill="#0b0e12" stroke={captured ? "#3bb9e8" : "rgba(242,238,228,.3)"} strokeWidth="4" />
										<text x={point.x} y={point.y + 5} textAnchor="middle" fill={captured ? "#3bb9e8" : "rgba(242,238,228,.35)"} fontFamily="ui-monospace, monospace" fontWeight="700" fontSize="13">{index + 1}</text>
									</g>
								);
							})}

							<g transform={`translate(${current.x} ${current.y - 42})`} filter="url(#reading-glow)">
								<path d="M-24 0 H24 M0 -18 V18 M-19 -12 L19 12 M19 -12 L-19 12" stroke="#168bd2" strokeWidth="4" strokeLinecap="round" />
								<circle cx="0" cy="0" r="8" fill="#0b0e12" stroke="#168bd2" strokeWidth="3" />
								<circle cx="0" cy="0" r="3" fill="#3bb9e8" />
							</g>
							<g transform={`translate(${Math.min(current.x + 36, 880)} ${current.y - 94})`}>
								<rect width="260" height="62" rx="12" fill="rgba(7,10,13,.96)" stroke="#3bb9e8" strokeOpacity=".65" />
								<text x="18" y="25" fill="#3bb9e8" fontFamily="ui-monospace, monospace" fontWeight="700" fontSize="16">MICROPHONE READING</text>
								<text x="18" y="47" fill="rgba(242,238,228,.55)" fontFamily="ui-monospace, monospace" fontSize="13">{current.level} · phase {current.phase}</text>
							</g>

							<g transform="translate(918 488)">
								<circle cx="0" cy="0" r="50" fill="rgba(59,185,232,.045)" stroke="#3bb9e8" strokeWidth="3" />
								<circle cx="0" cy="0" r="15" fill="#0b0e12" stroke="#3bb9e8" strokeWidth="4" />
								<path d="M-24 0 H24 M0 -24 V24" stroke="#3bb9e8" strokeWidth="4" />
							</g>
							<rect x="948" y="516" width="198" height="48" rx="10" fill="#070a0d" stroke="#3bb9e8" strokeOpacity=".5" />
							<text x="1047" y="546" textAnchor="middle" fill="#3bb9e8" fontFamily="ui-monospace, monospace" fontWeight="700" fontSize="16">TARGET BUBBLE</text>
						</svg>

						<div className="pointer-events-none absolute left-4 top-4 flex items-center gap-2 rounded-lg border border-[#168bd2]/20 bg-[#0b0e12]/90 px-3 py-2 backdrop-blur">
							<Plane className="size-3.5 text-[#168bd2]" />
							<span className="font-mono text-[10px] uppercase tracking-[0.14em] text-[#168bd2]">Drone probe path</span>
						</div>
						<div className="pointer-events-none absolute bottom-4 right-4 rounded-lg border border-white/10 bg-[#0b0e12]/90 px-3 py-2 font-mono text-[10px] uppercase tracking-[0.13em] text-white/45 backdrop-blur">
							Reading {activePoint + 1} / {PATH_POINTS.length}
						</div>
					</div>
					<p className="px-1 text-xs leading-5 text-white/40">
						At each point the eight speakers emit orthogonal probe codes. The drone microphone recovers one measured row of the environment transfer matrix.
					</p>
				</div>

				{showControls ? <aside className="space-y-4">
					<section className="rounded-2xl border border-[#3bb9e8]/20 bg-[#111820] p-5">
						<div className="flex items-start justify-between gap-4">
							<div>
								<p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.17em] text-[#3bb9e8]">
									<Mic2 className="size-3.5" /> Sampling H
								</p>
								<p className="mt-2 text-xs leading-5 text-white/45">Each location measures all eight speaker-to-point responses.</p>
							</div>
							<span className="font-mono text-xl font-semibold text-[#3bb9e8]">{activePoint + 1}/4</span>
						</div>
						<div className="mt-5 grid grid-cols-8 gap-1" aria-label="Estimated transfer matrix rows">
							{Array.from({ length: 32 }, (_, index) => {
								const row = Math.floor(index / 8);
								const captured = row <= activePoint;
								return (
									<span
										key={`matrix-cell-${index}`}
										className={`aspect-square rounded-sm border transition duration-500 ${captured ? "border-[#3bb9e8]/35 bg-[#3bb9e8]/35 shadow-[0_0_8px_rgba(59,185,232,.2)]" : "border-white/8 bg-[#0b0e12]"}`}
									/>
								);
							})}
						</div>
						<p className="mt-3 font-mono text-[9px] uppercase tracking-[0.12em] text-white/30">4 positions × 8 speaker responses</p>
						<button
							type="button"
							onClick={() => setActivePoint(0)}
							className="mt-4 inline-flex items-center gap-2 rounded-lg border border-white/10 px-3 py-2 text-[11px] text-white/45 transition hover:border-white/20 hover:text-white/75"
						>
							<RotateCcw className="size-3.5" /> Restart sampling
						</button>
					</section>

					<section className="rounded-2xl border border-white/10 bg-[#111820] p-5">
						<p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.17em] text-[#168bd2]">
							<Camera className="size-3.5" /> Camera + acoustics
						</p>
						<div className="mt-4 space-y-3 text-xs leading-5 text-white/45">
							<p className="flex gap-2"><Check className="mt-0.5 size-3.5 shrink-0 text-[#3bb9e8]" /> Camera supplies geometry, scale, surfaces, and listener coordinates.</p>
							<p className="flex gap-2"><Check className="mt-0.5 size-3.5 shrink-0 text-[#3bb9e8]" /> Probe readings supply the phase and amplitude the camera cannot see.</p>
							<p className="flex gap-2"><Check className="mt-0.5 size-3.5 shrink-0 text-[#3bb9e8]" /> A scene model interpolates those sparse rows across the target bubble.</p>
						</div>
					</section>

					<section className="rounded-2xl border border-[#ff6a2a]/15 bg-[#ff6a2a]/[0.035] p-5">
						<p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.17em] text-[#ff6a2a]">
							<Radio className="size-3.5" /> What is learned
						</p>
						<p className="mt-3 text-xs leading-5 text-white/45">
							The model maps scene geometry plus sparse measured rows to the missing transfer values around the listener. It predicts H; the next pane still solves the speaker weights deterministically.
						</p>
					</section>
				</aside> : null}
			</section>
	);
}
