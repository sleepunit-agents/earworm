"""Loudness feature extraction — LUFS, dynamics, crest factor."""

from __future__ import annotations

import numpy as np
import pyloudnorm as pyln

from earworm.models import LoudnessFeatures


def _rms_envelope(y: np.ndarray, sr: int, hop_seconds: float = 1.0) -> list[float]:
    """Compute RMS envelope sampled at ~1 Hz."""
    hop = int(sr * hop_seconds)
    frames = [y[i : i + hop] for i in range(0, len(y), hop) if len(y[i : i + hop]) > 0]
    return [float(np.sqrt(np.mean(f**2))) for f in frames]


def extract_loudness(y: np.ndarray, sr: int, y_stereo: np.ndarray | None = None) -> LoudnessFeatures:
    """Extract loudness features.

    Args:
        y: Mono audio signal for analysis.
        sr: Sample rate.
        y_stereo: Original stereo signal for LUFS measurement (or mono if source is mono).
    """
    if y_stereo is None:
        y_stereo = y

    # LUFS measurement — needs the original channel layout
    meter = pyln.Meter(sr)
    # pyloudnorm expects shape (samples, channels) for stereo, or (samples,) for mono
    if y_stereo.ndim == 1:
        lufs_input = y_stereo
    else:
        lufs_input = y_stereo.T  # librosa is (channels, samples), pyloudnorm wants (samples, channels)

    try:
        lufs_integrated = float(meter.integrated_loudness(lufs_input))
    except ValueError:
        lufs_integrated = -70.0  # Silence fallback

    # Short-term loudness (3-second windows)
    hop_st = int(sr * 0.1)  # 100ms hop
    window_st = int(sr * 3.0)  # 3-second window
    st_loudness = []
    for i in range(0, len(y) - window_st, hop_st):
        chunk = y[i : i + window_st]
        try:
            st_loudness.append(float(meter.integrated_loudness(chunk)))
        except ValueError:
            st_loudness.append(-70.0)

    lufs_short_term_max = max(st_loudness) if st_loudness else lufs_integrated
    lufs_range = (max(st_loudness) - min(st_loudness)) if st_loudness else 0.0

    # Peak and RMS
    peak_linear = float(np.max(np.abs(y)))
    peak_db = float(20 * np.log10(peak_linear + 1e-10))
    rms_linear = float(np.sqrt(np.mean(y**2)))
    rms_db = float(20 * np.log10(rms_linear + 1e-10))

    # Crest factor — how punchy it is
    crest_factor_db = peak_db - rms_db

    # Dynamic range (simplified — difference between loud and quiet sections)
    dynamic_range_db = crest_factor_db

    # Loudness curve at ~1 Hz
    loudness_curve_linear = _rms_envelope(y, sr, hop_seconds=1.0)
    loudness_curve = [float(20 * np.log10(v + 1e-10)) for v in loudness_curve_linear]

    return LoudnessFeatures(
        lufs_integrated=lufs_integrated,
        lufs_short_term_max=lufs_short_term_max,
        lufs_range=lufs_range,
        dynamic_range_db=dynamic_range_db,
        peak_db=peak_db,
        rms_db=rms_db,
        crest_factor_db=crest_factor_db,
        loudness_curve=loudness_curve,
    )
