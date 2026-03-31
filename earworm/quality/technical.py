"""Technical audio quality — clipping, noise floor, frequency balance."""

from __future__ import annotations

import librosa
import numpy as np

from earworm.models import TechnicalQuality


def extract_technical(y: np.ndarray, sr: int) -> TechnicalQuality:
    """Assess technical audio quality from the raw signal.

    Checks for clipping, DC offset, noise floor, and spectral tilt.
    """
    # Clipping detection: samples at or very near ±1.0
    clip_threshold = 0.99
    clipped = np.abs(y) >= clip_threshold
    clipping_ratio = float(clipped.mean())

    # Count continuous clipping regions
    clipping_regions = 0
    in_clip = False
    for is_clipped in clipped:
        if is_clipped and not in_clip:
            clipping_regions += 1
            in_clip = True
        elif not is_clipped:
            in_clip = False

    # DC offset
    dc_offset = float(np.mean(y))
    has_dc_offset = abs(dc_offset) > 0.005

    # Noise floor estimation — find the quietest 10% of the signal (windowed RMS)
    hop = int(sr * 0.05)  # 50ms windows
    rms_frames = []
    for i in range(0, len(y) - hop, hop):
        frame_rms = float(np.sqrt(np.mean(y[i : i + hop] ** 2)))
        rms_frames.append(frame_rms)

    if rms_frames:
        rms_frames.sort()
        # Take the 10th percentile as noise floor estimate
        noise_idx = max(0, len(rms_frames) // 10 - 1)
        noise_rms = rms_frames[noise_idx]
        noise_floor_db = float(20 * np.log10(noise_rms + 1e-10))
    else:
        noise_floor_db = -70.0

    # Frequency balance: compare spectral energy distribution to flat reference
    # A well-balanced mix should have energy spread across the spectrum
    # Score = 1 - normalized spectral tilt coefficient
    S = np.abs(librosa.stft(y, n_fft=2048))
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mean_spectrum = S.mean(axis=1)

    if mean_spectrum.sum() > 0:
        # Spectral tilt: fit log-frequency vs log-magnitude slope
        # A very tilted spectrum (lots of bass, no highs) gets low score
        valid = (freq_bins > 20) & (freq_bins < sr / 2) & (mean_spectrum > 0)
        if valid.sum() > 10:
            log_freq = np.log10(freq_bins[valid])
            log_mag = np.log10(mean_spectrum[valid] + 1e-10)
            # Normalize
            log_freq_norm = (log_freq - log_freq.mean()) / (log_freq.std() + 1e-8)
            slope = float(np.polyfit(log_freq_norm, log_mag, 1)[0])
            # Map slope to 0-1 score: 0 slope = perfect balance
            # Typical music has negative slope (-2 to -6 dB/octave)
            # Score penalizes extreme tilt in either direction
            frequency_balance_score = max(0.0, 1.0 - abs(slope) / 3.0)
        else:
            frequency_balance_score = 0.5
    else:
        frequency_balance_score = 0.0

    return TechnicalQuality(
        clipping_ratio=round(clipping_ratio, 6),
        clipping_regions=clipping_regions,
        dc_offset=round(dc_offset, 6),
        noise_floor_db=round(noise_floor_db, 1),
        frequency_balance_score=round(frequency_balance_score, 3),
        has_dc_offset=has_dc_offset,
    )
