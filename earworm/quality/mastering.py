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
    limiter_score = _detect_limiter_artifacts(y)

    return MasteringQuality(
        lufs_integrated=round(loudness.lufs_integrated, 1),
        lufs_deviation_from_target=round(lufs_deviation, 1),
        dynamic_range_score=round(dr_score, 3),
        loudness_consistency=round(loudness_consistency, 3),
        limiter_artifact_score=round(limiter_score, 3),
        crest_factor_db=round(cf, 1),
    )


def _detect_limiter_artifacts(y: np.ndarray) -> float:
    """Detect limiter/clipper artifacts in the signal.

    Looks for:
    1. Consecutive samples at near-maximum level (hard limiting)
    2. Flat-topped waveforms (soft clipping)

    Returns 0.0 (no artifacts) to 1.0 (severe artifacts).
    """
    abs_y = np.abs(y)
    threshold = 0.95

    # Count consecutive samples at near-maximum
    near_max = abs_y >= threshold
    if not near_max.any():
        return 0.0

    # Count runs of consecutive near-max samples
    # Runs of 3+ samples suggest limiting (natural peaks are usually 1-2 samples)
    run_lengths = []
    current_run = 0
    for is_max in near_max:
        if is_max:
            current_run += 1
        else:
            if current_run >= 3:
                run_lengths.append(current_run)
            current_run = 0
    if current_run >= 3:
        run_lengths.append(current_run)

    if not run_lengths:
        return 0.0

    # Score based on frequency and length of limiting runs
    total_limited_samples = sum(run_lengths)
    limited_ratio = total_limited_samples / len(y)

    # Also check for flat-topped waveforms: low variance in near-max regions
    near_max_values = abs_y[near_max]
    if len(near_max_values) > 10:
        flatness = 1.0 - float(np.std(near_max_values)) / 0.05  # Near 1.0 if flat
        flatness = max(0.0, flatness)
    else:
        flatness = 0.0

    # Combined score
    score = min(1.0, limited_ratio * 100.0 + flatness * 0.3)
    return score
