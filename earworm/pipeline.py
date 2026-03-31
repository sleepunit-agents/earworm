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
from earworm.models import Layer1Features


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
