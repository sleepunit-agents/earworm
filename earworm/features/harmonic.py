"""Harmonic feature extraction — key, tonality, harmonic content."""

from __future__ import annotations

import librosa
import numpy as np

from earworm.models import HarmonicFeatures

# Krumhansl-Kessler key profiles
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(chroma: np.ndarray) -> tuple[str, float, list[float]]:
    """Estimate key using Krumhansl-Kessler profile correlation."""
    chroma_avg = chroma.mean(axis=1)
    if chroma_avg.sum() == 0:
        return "unknown", 0.0, [0.0] * 12

    # Correlate with all 24 major/minor keys (12 roots x 2 modes)
    correlations = []
    for shift in range(12):
        major_corr = float(np.corrcoef(np.roll(chroma_avg, -shift), _MAJOR_PROFILE)[0, 1])
        minor_corr = float(np.corrcoef(np.roll(chroma_avg, -shift), _MINOR_PROFILE)[0, 1])
        correlations.append((major_corr, f"{_NOTE_NAMES[shift]} major"))
        correlations.append((minor_corr, f"{_NOTE_NAMES[shift]} minor"))

    correlations.sort(key=lambda x: x[0], reverse=True)
    best_corr, best_key = correlations[0]

    # Confidence = gap between best and second-best correlation
    confidence = best_corr - correlations[1][0] if len(correlations) > 1 else 0.0
    confidence = max(0.0, min(confidence * 5.0, 1.0))  # Scale to 0-1

    # Key profile = correlation for each pitch class with the winning mode
    profile = []
    for shift in range(12):
        rolled = np.roll(chroma_avg, -shift)
        ref = _MAJOR_PROFILE if "major" in best_key else _MINOR_PROFILE
        profile.append(float(np.corrcoef(rolled, ref)[0, 1]))

    return best_key, confidence, profile


def extract_harmonic(y: np.ndarray, sr: int) -> HarmonicFeatures:
    """Extract harmonic/tonal features from a mono audio signal."""

    # Harmonic-percussive separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy = float(np.sum(y_harmonic**2))
    total_energy = float(np.sum(y**2))
    harmonic_ratio = harmonic_energy / total_energy if total_energy > 0 else 0.0

    # Constant-Q chromagram (better frequency resolution for harmony)
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

    # Key estimation
    key, key_confidence, key_profile = _estimate_key(chroma_cqt)

    # Tonnetz — tonal centroid features (6-dimensional)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)

    return HarmonicFeatures(
        key=key,
        key_confidence=key_confidence,
        key_profile=key_profile,
        chroma_cqt_mean=chroma_cqt.mean(axis=1).tolist(),
        tonnetz_mean=tonnetz.mean(axis=1).tolist(),
        tonnetz_std=tonnetz.std(axis=1).tolist(),
        harmonic_ratio=harmonic_ratio,
    )
