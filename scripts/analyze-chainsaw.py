#!/usr/bin/env python3
"""Extract a periodic chainsaw profile for the browser simulation.

The source recording is decoded to mono PCM, then a steady cutting interval is
phase-folded at its autocorrelation peak. The JSON contains the display
waveform, one averaged engine cycle, and a Fourier-series approximation used
by the multi-frequency controller.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "public" / "audio" / "chainsaw-10.ogg"
OUTPUT = ROOT / "lib" / "acoustics" / "chainsaw-analysis.json"
SAMPLE_RATE = 22_050
SEGMENT_START = 3.3
SEGMENT_END = 3.4


def rounded(values: np.ndarray, digits: int = 5) -> list[float]:
	return [round(float(value), digits) for value in values]


def main() -> None:
	with tempfile.TemporaryDirectory() as directory:
		wav_path = Path(directory) / "chainsaw.wav"
		subprocess.run(
			[
				"ffmpeg",
				"-hide_banner",
				"-loglevel",
				"error",
				"-y",
				"-i",
				str(SOURCE),
				"-ac",
				"1",
				"-ar",
				str(SAMPLE_RATE),
				"-c:a",
				"pcm_s16le",
				str(wav_path),
			],
			check=True,
		)
		sample_rate, raw = wavfile.read(wav_path)

	audio = raw.astype(np.float64) / 32_768.0
	segment = audio[
		int(SEGMENT_START * sample_rate) : int(SEGMENT_END * sample_rate)
	]
	segment = segment - np.mean(segment)

	# Find the strongest repeat period in the engine band (120–300 Hz).
	sos = signal.butter(4, [80, 600], btype="bandpass", fs=sample_rate, output="sos")
	filtered = signal.sosfiltfilt(sos, segment)
	autocorrelation = signal.correlate(filtered, filtered, mode="full", method="fft")
	autocorrelation = autocorrelation[len(filtered) - 1 :]
	autocorrelation /= autocorrelation[0]
	minimum_lag = int(sample_rate / 300)
	maximum_lag = int(sample_rate / 120)
	peaks, _ = signal.find_peaks(
		autocorrelation[minimum_lag:maximum_lag], prominence=0.04
	)
	if len(peaks) == 0:
		raise RuntimeError("No stable engine period found")
	period_samples = int(
		max(
			(peaks + minimum_lag),
			key=lambda lag: autocorrelation[lag],
		)
	)
	fundamental_hz = sample_rate / period_samples

	cycle_count = len(segment) // period_samples
	cycles = segment[: cycle_count * period_samples].reshape(
		cycle_count, period_samples
	)
	mean_cycle = np.mean(cycles, axis=0)
	mean_cycle -= np.mean(mean_cycle)
	cycle_peak = np.max(np.abs(mean_cycle))
	normalized_cycle = mean_cycle / max(cycle_peak, 1e-12)
	periodic_energy_fraction = np.var(mean_cycle) / max(np.var(segment), 1e-12)

	coefficients = np.fft.rfft(mean_cycle)
	amplitudes = 2 * np.abs(coefficients) / period_samples
	phases = np.angle(coefficients)
	orders = np.arange(1, min(13, len(amplitudes)))
	harmonic_power = amplitudes[orders] ** 2 / 2
	total_harmonic_power = np.sum(harmonic_power)
	max_amplitude = np.max(amplitudes[orders])
	harmonics = [
		{
			"order": int(order),
			"frequency": round(float(order * fundamental_hz), 2),
			"amplitude": round(float(amplitudes[order] / max_amplitude), 5),
			"phase": round(float(phases[order]), 5),
			"powerShare": round(
				float(harmonic_power[index] / total_harmonic_power), 5
			),
		}
		for index, order in enumerate(orders)
	]

	# A compact min/max envelope preserves the shape of the steady excerpt.
	bucket_count = 240
	bucket_size = len(segment) // bucket_count
	trimmed = segment[: bucket_count * bucket_size].reshape(bucket_count, bucket_size)
	waveform_min = np.min(trimmed, axis=1)
	waveform_max = np.max(trimmed, axis=1)
	waveform_peak = max(np.max(np.abs(waveform_min)), np.max(np.abs(waveform_max)))

	# Log-frequency spectral bins show the periodic peaks and broadband floor.
	welch_window = min(4096, len(segment))
	frequencies, power = signal.welch(
		segment,
		sample_rate,
		window="hann",
		nperseg=welch_window,
		noverlap=welch_window // 2,
	)
	edges = np.geomspace(40, 5000, 65)
	spectrum = []
	for low, high in zip(edges[:-1], edges[1:]):
		mask = (frequencies >= low) & (frequencies < high)
		value = np.max(power[mask]) if np.any(mask) else 1e-20
		spectrum.append(
			{
				"frequency": round(float(np.sqrt(low * high)), 2),
				"db": round(float(10 * np.log10(value + 1e-20)), 3),
			}
		)
	maximum_db = max(item["db"] for item in spectrum)
	for item in spectrum:
		item["relativeDb"] = round(item.pop("db") - maximum_db, 3)

	result = {
		"source": {
			"title": "Chainsaw 10",
			"author": "ezwa",
			"license": "Public domain",
			"url": "https://commons.wikimedia.org/wiki/File:Chainsaw_10.ogg",
			"durationSeconds": round(len(audio) / sample_rate, 3),
			"sampleRate": sample_rate,
		},
		"segment": {
			"startSeconds": SEGMENT_START,
			"endSeconds": SEGMENT_END,
			"rmsDbfs": round(
				float(20 * np.log10(np.sqrt(np.mean(segment**2)) + 1e-20)), 2
			),
			"fundamentalHz": round(float(fundamental_hz), 2),
			"periodMilliseconds": round(float(1000 / fundamental_hz), 3),
			"autocorrelation": round(float(autocorrelation[period_samples]), 3),
			"periodicEnergyFraction": round(float(periodic_energy_fraction), 3),
		},
		"waveformMin": rounded(waveform_min / waveform_peak),
		"waveformMax": rounded(waveform_max / waveform_peak),
		"cycle": rounded(normalized_cycle),
		"spectrum": spectrum,
		"harmonics": harmonics,
	}

	OUTPUT.parent.mkdir(parents=True, exist_ok=True)
	OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
	print(
		f"Wrote {OUTPUT} · {fundamental_hz:.2f} Hz · "
		f"{periodic_energy_fraction * 100:.1f}% periodic energy"
	)


if __name__ == "__main__":
	main()
