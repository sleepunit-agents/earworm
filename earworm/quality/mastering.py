"""Mastering quality assessment — loudness compliance, dynamics, limiter artifacts."""

from __future__ import annotations

import numpy as np

from earworm.models import MasteringQuality, LoudnessFeatures


# Standard streaming LUFS target
STREAMING_LUFS_TARGET = -14.0


def extract_mastering(
    y: np.ndarray,
    sr: int,
    loudness: LoudnessFeatures,
) -> MasteringQuality:
    """Assess mastering quality using Layer 1 loudness data and raw signal.

    Evaluates loudness compliance, dynamic range, consistency, and
    limiter artifacts.
    """
    # LUFS deviation from streaming target
    lufs_deviation = abs(loudness.lufs_integrated - STREAMING_LUFS_TARGET)

    # Dynamic range score — based on crest factor and loudness range
    # Good dynamic range: crest factor 8-14 dB, LRA > 6 LU
    # Over-compressed: crest factor < 6 dB, LRA < 4 LU
    # Under-mastered: crest factor > 20 dB (too dynamic for most contexts)
    cf = loudness.crest_factor_db
    lra = loudness.lufs_range

    if cf < 3:
        dr_score = 0.1  # Severely compressed
    elif cf < 6:
        dr_score = 0.3 + (cf - 3) / 3 * 0.2  # Compressed
    elif cf < 14:
        dr_score = 0.7 + (cf - 6) / 8 * 0.3  # Good range
    elif cf < 20:
        dr_score = 0.8 - (cf - 14) / 6 * 0.2  # Getting loose
    else:
        dr_score = 0.5  # Extremely dynamic — may be intentional

    # Factor in LRA
    if lra > 2:
        lra_bonus = min(0.1, (lra - 2) / 20.0 * 0.1)
        dr_score = min(1.0, dr_score + lra_bonus)

    # Loudness consistency — variance of loudness curve
    if len(loudness.loudness_curve) > 1:
        curve = np.array(loudness.loudness_curve)
        # Filter out silence (-70 dB entries)
        active = curve[curve > -60]
        if len(active) > 1:
            loudness_std = float(np.std(active))
            # Low std = consistent, high std = inconsistent
            # Typical music: 3-8 dB std
            loudness_consistency = max(0.0, 1.0 - loudness_std / 15.0)
        else:
            loudness_consistency = 0.5
    else:
        loudness_consistency = 0.5

    # Limiter artifact detection
    limiter_score = _detect_limiter_artifacts(y, sr)

    return MasteringQuality(
        lufs_integrated=round(loudness.lufs_integrated, 1),
        lufs_deviation_from_target=round(lufs_deviation, 1),
        dynamic_range_score=round(dr_score, 3),
        loudness_consistency=round(loudness_consistency, 3),
        limiter_artifact_score=round(limiter_score, 3),
        crest_factor_db=round(cf, 1),
    )


def _detect_limiter_artifacts(y: np.ndarray, sr: int = 44100) -> float:
    """Detect limiter/clipper artifacts in the signal.

    Distinguishes between natural peaks hitting near-max (healthy loud master)
    and brickwall limiting artifacts (flat-topped waveforms with near-zero
    derivative). A track mastered to -8 LUFS with preserved transient character
    should score low; a brickwalled master at the same loudness should score high.

    Scoring is based on three signals:
    1. Flat-topped runs — consecutive near-max samples with near-zero derivative
       (the primary signature of brickwall limiting)
    2. Run density — what fraction of the signal is spent in sustained near-max
       plateaus, weighted by run length
    3. Plateau flatness — how uniform the near-max regions are (true limiting
       produces nearly identical sample values)

    Returns 0.0 (no artifacts) to 1.0 (severe artifacts).
    """
    abs_y = np.abs(y)
    threshold = 0.95

    near_max = abs_y >= threshold
    if not near_max.any():
        return 0.0

    # Minimum run length scales with sample rate — at 48kHz, natural transient
    # peaks can sustain 3-4 samples above threshold. Require longer plateaus.
    min_run = max(4, int(sr * 0.0002))  # ~0.2ms minimum, at least 4 samples

    # Vectorized run-length detection using diff on the boolean mask
    padded = np.concatenate([[False], near_max, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    run_lengths = ends - starts

    long_runs = run_lengths[run_lengths >= min_run]
    if len(long_runs) == 0:
        return 0.0

    # Compute derivative flatness within each long run — the key discriminator.
    # True limiting produces near-zero derivative (flat-topped waveform).
    # Natural peaks have significant derivative even when above threshold.
    flat_run_samples = 0
    total_run_samples = 0
    for i in range(len(run_lengths)):
        if run_lengths[i] < min_run:
            continue
        s, e = starts[i], ends[i]
        run_signal = abs_y[s:e]
        total_run_samples += len(run_signal)

        if len(run_signal) < 3:
            continue
        deriv = np.abs(np.diff(run_signal))
        mean_deriv = float(np.mean(deriv))
        # Flat-topped: derivative < 0.002 per sample (essentially clamped)
        # Natural peak: derivative > 0.005 (signal is still moving)
        if mean_deriv < 0.003:
            flat_run_samples += len(run_signal)

    if total_run_samples == 0:
        return 0.0

    # Flatness ratio: what fraction of long near-max runs are truly flat-topped
    flatness_ratio = flat_run_samples / total_run_samples

    # Density: fraction of total signal in long near-max runs
    density = total_run_samples / len(y)

    # Length severity: longer runs are more characteristic of limiting.
    # Weight by log of mean run length relative to minimum.
    mean_long_run = float(np.mean(long_runs))
    length_factor = min(1.0, np.log2(mean_long_run / min_run + 1) / 4.0)

    # Combined score: density scaled by flatness and length severity
    # Pure density alone doesn't score high — it needs flat-topped character
    score = min(1.0, density * 50.0 * flatness_ratio * (0.3 + 0.7 * length_factor))

    return score
