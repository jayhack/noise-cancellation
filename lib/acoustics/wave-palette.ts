type Rgb = readonly [number, number, number];

// These ramps come directly from jay.ai's FluidAnimation palette. Splitting
// the cool and warm halves preserves pressure polarity in the acoustic field.
const RESTING: Rgb = [42, 53, 80];
const FIELD_BACKGROUND: Rgb = [0, 0, 0];
const FIELD_BRIGHTNESS = 2 / 3;

const COOL_WAVE_COLORS: readonly Rgb[] = [
	RESTING,
	[52, 80, 122],
	[30, 106, 160],
	[32, 144, 200],
	[77, 171, 217],
	[125, 211, 252],
];

const WARM_WAVE_COLORS: readonly Rgb[] = [
	RESTING,
	[90, 32, 16],
	[124, 45, 18],
	[185, 28, 28],
	[220, 38, 38],
	[234, 88, 12],
	[249, 115, 22],
	[251, 146, 60],
	[251, 191, 36],
	[253, 230, 138],
	[254, 243, 199],
];

export const WAVE_FIELD_BACKGROUND = "#000000";

function lerp(left: number, right: number, amount: number) {
	return left + (right - left) * amount;
}

function samplePalette(palette: readonly Rgb[], amount: number): Rgb {
	const position = amount * (palette.length - 1);
	const lowerIndex = Math.floor(position);
	const upperIndex = Math.min(palette.length - 1, lowerIndex + 1);
	const mix = position - lowerIndex;
	const lower = palette[lowerIndex];
	const upper = palette[upperIndex];

	return [
		lerp(lower[0], upper[0], mix),
		lerp(lower[1], upper[1], mix),
		lerp(lower[2], upper[2], mix),
	];
}

export function pressureFieldColor(
	signedStrength: number,
	visualGain = 0.78,
) {
	const magnitude = Math.min(Math.abs(signedStrength) * visualGain, 1);
	const normalized = Math.pow(magnitude, 0.8);
	const palette = signedStrength >= 0 ? COOL_WAVE_COLORS : WARM_WAVE_COLORS;
	const target = samplePalette(palette, normalized);
	const visibility = (0.16 + normalized * 0.84) * FIELD_BRIGHTNESS;

	return `rgb(${Math.round(lerp(FIELD_BACKGROUND[0], target[0], visibility))} ${Math.round(lerp(FIELD_BACKGROUND[1], target[1], visibility))} ${Math.round(lerp(FIELD_BACKGROUND[2], target[2], visibility))})`;
}
