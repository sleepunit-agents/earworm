"""Mix quality assessment — frequency distribution, stereo usage, clarity."""

from __future__ import annotations

import librosa
import numpy as np

from earworm.models import MixQuality


def extract_mix(y: np.ndarray, sr: int, y_stereo: np.ndarray | None = None) -> MixQuality:
    """Assess mix quality: frequency distribution, stereo usage, clarity.

    Evaluates how well the frequency spectrum is utilized and how the
    stereo field is managed.
    """
    # Frequency band energy ratios
    S = np.abs(librosa.stft(y, n_fft=2048)) ** 2  # Power spectrum
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    total_energy = S.sum()
    if total_energy == 0:
        return MixQuality(
            low_ratio=0.0,
            mid_ratio=0.0,
            high_ratio=0.0,
            spectral_balance_score=0.0,
            stereo_width_score=0.0,
            low_end_clarity=0.0,
            high_frequency_clarity=0.0,
        )

    # Band boundaries: low < 200Hz, mid 200Hz-4kHz, high > 4kHz
    low_mask = freqs < 200
    mid_mask = (freqs >= 200) & (freqs < 4000)
    high_mask = freqs >= 4000

    low_energy = float(S[low_mask].sum() / total_energy)
    mid_energy = float(S[mid_mask].sum() / total_energy)
    high_energy = float(S[high_mask].sum() / total_energy)

    # Spectral balance score — how close to a reasonable distribution
    # Well-mixed music typically has: low ~20-30%, mid ~40-55%, high ~15-30%
    # Score penalizes extreme deviations from this
    ideal_low, ideal_mid, ideal_high = 0.25, 0.50, 0.25
    balance_error = (
        abs(low_energy - ideal_low)
        + abs(mid_energy - ideal_mid)
        + abs(high_energy - ideal_high)
    )
    spectral_balance_score = max(0.0, 1.0 - balance_error * 2.0)

    # Stereo width score — from stereo signal analysis
    stereo_width_score = _assess_stereo_width(y_stereo) if y_stereo is not None else 0.5

    # Low-end clarity — spectral contrast in the low band
    # High contrast = clear separation between bass elements
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    # First band covers lowest frequencies
    low_contrast = float(contrast[0].mean())
    # Normalize to 0-1 (contrast typically 0-50 dB)
    low_end_clarity = min(1.0, max(0.0, low_contrast / 40.0))

    # High-frequency clarity — spectral contrast in the high band
    high_contrast = float(contrast[-1].mean())
    high_frequency_clarity = min(1.0, max(0.0, high_contrast / 40.0))

    return MixQuality(
        low_ratio=round(low_energy, 4),
        mid_ratio=round(mid_energy, 4),
        high_ratio=round(high_energy, 4),
        spectral_balance_score=round(spectral_balance_score, 3),
        stereo_width_score=round(stereo_width_score, 3),
        low_end_clarity=round(low_end_clarity, 3),
        high_frequency_clarity=round(high_frequency_clarity, 3),
    )


def _assess_stereo_width(y_stereo: np.ndarray) -> float:
    """Score stereo width usage: 0 = mono/collapsed, 1 = well-utilized field."""
    if y_stereo.ndim == 1:
        return 0.0  # Mono source

    left = y_stereo[0]
    right = y_stereo[1]

    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    mid_energy = float(np.sum(mid ** 2))
    side_energy = float(np.sum(side ** 2))
    total = mid_energy + side_energy

    if total == 0:
        return 0.0

    # Side ratio: 0 = pure mono, approaching 1 = heavily stereo
    side_ratio = side_energy / total

    # Good stereo width is typically 0.1-0.4 side ratio
    # Too little = effectively mono; too much = phase issues likely
    if side_ratio < 0.05:
        return side_ratio / 0.05 * 0.3  # Barely stereo
    elif side_ratio < 0.4:
        return 0.5 + (side_ratio - 0.05) / 0.35 * 0.5  # Good range
    else:
        return max(0.3, 1.0 - (side_ratio - 0.4) * 2.0)  # Excessive width
