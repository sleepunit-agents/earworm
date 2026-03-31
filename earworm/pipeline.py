"""Pipeline orchestrator — loads audio, runs all feature extractors."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from earworm.features.harmonic import extract_harmonic
from earworm.features.loudness import extract_loudness
from earworm.features.spectral import extract_spectral
from earworm.features.stereo import extract_stereo
from earworm.features.temporal import extract_temporal
from earworm.models import Layer1Features, Layer2Features
from earworm.structure.energy import extract_energy_arc
from earworm.structure.phrase import extract_phrase
from earworm.structure.recurrence import extract_recurrence
from earworm.structure.segmentation import extract_segmentation


def load_audio(path: str | Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load audio file, returning (mono, original, sample_rate).

    Returns:
        y_mono: Mono mixdown for analysis.
        y_original: Original channel layout (mono or stereo) for stereo/loudness analysis.
        sr: Sample rate.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Load original channel layout
    y_original, sr = librosa.load(str(path), sr=None, mono=False)

    # Mono mixdown for most analysis
    if y_original.ndim > 1:
        y_mono = librosa.to_mono(y_original)
    else:
        y_mono = y_original

    return y_mono, y_original, sr


def analyze_layer1(path: str | Path) -> Layer1Features:
    """Run the full Layer 1 feature extraction pipeline on an audio file."""
    path = Path(path)
    y_mono, y_original, sr = load_audio(path)

    channels = y_original.shape[0] if y_original.ndim > 1 else 1
    duration = len(y_mono) / sr

    return Layer1Features(
        file_path=str(path),
        duration_seconds=duration,
        sample_rate=sr,
        channels=channels,
        spectral=extract_spectral(y_mono, sr),
        temporal=extract_temporal(y_mono, sr),
        harmonic=extract_harmonic(y_mono, sr),
        loudness=extract_loudness(y_mono, sr, y_stereo=y_original),
        stereo=extract_stereo(y_original, sr),
    )


def analyze_layer2(path: str | Path, layer1: Layer1Features | None = None) -> Layer2Features:
    """Run Layer 2 structural comprehension on an audio file.

    Optionally accepts Layer 1 results to avoid recomputing beat/tempo data.
    If not provided, runs Layer 1 internally for the needed features.
    """
    path = Path(path)
    y_mono, _, sr = load_audio(path)
    duration = len(y_mono) / sr

    # Get beat/tempo data from Layer 1 (or compute it)
    if layer1 is not None:
        beat_times = layer1.temporal.beat_times
        bpm = layer1.temporal.bpm
    else:
        from earworm.features.temporal import extract_temporal

        temporal = extract_temporal(y_mono, sr)
        beat_times = temporal.beat_times
        bpm = temporal.bpm

    # Segmentation first — recurrence depends on its output
    segmentation = extract_segmentation(y_mono, sr)

    # Recurrence uses segmentation boundaries and labels
    recurrence = extract_recurrence(
        y_mono, sr, segmentation.boundaries, segmentation.labels
    )

    # Energy arc
    energy_arc = extract_energy_arc(y_mono, sr)

    # Phrase structure from beat grid
    phrase = extract_phrase(y_mono, sr, beat_times, bpm)

    return Layer2Features(
        file_path=str(path),
        duration_seconds=duration,
        segmentation=segmentation,
        recurrence=recurrence,
        energy_arc=energy_arc,
        phrase=phrase,
    )
