"""Phrase structure analysis — groupings, regularity, and surprises."""

from __future__ import annotations

import numpy as np
from collections import Counter

from earworm.models import PhraseFeatures


def extract_phrase(
    y: np.ndarray,
    sr: int,
    beat_times: list[float],
    bpm: float,
) -> PhraseFeatures:
    """Analyze phrase structure from beat grid.

    Most music organizes beats into phrases of 4, 8, or 16 bars (in 4/4 time).
    This extractor finds phrase boundaries by looking for energy changes
    aligned to the beat grid, then measures regularity.
    """
    duration = len(y) / sr
    beats = np.array(beat_times)

    # Need enough beats for phrase analysis
    if len(beats) < 8:
        return PhraseFeatures(
            phrase_boundaries=[0.0, duration],
            phrase_lengths_beats=[float(len(beats))],
            n_phrases=1,
            typical_phrase_beats=float(len(beats)),
            regularity=1.0,
            irregular_phrases=[],
        )

    # Compute onset strength at each beat position
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_frames = librosa.time_to_frames(beats, sr=sr)
    beat_frames = np.clip(beat_frames, 0, len(onset_env) - 1)
    beat_strengths = onset_env[beat_frames]

    # Try common phrase lengths: 4, 8, 16 beats
    # Score each by how well beat 1 of each phrase aligns with onset strength peaks
    best_phrase_len = _find_best_phrase_length(beat_strengths, candidates=[4, 8, 16])

    # Build phrase boundaries aligned to beat grid
    phrase_boundaries = [0.0]
    phrase_lengths_beats = []
    current_beat = 0

    while current_beat + best_phrase_len <= len(beats):
        end_beat = current_beat + best_phrase_len
        if end_beat < len(beats):
            phrase_boundaries.append(float(beats[end_beat]))
        else:
            phrase_boundaries.append(duration)
        phrase_lengths_beats.append(float(best_phrase_len))
        current_beat = end_beat

    # Handle remaining beats as a final (possibly short) phrase
    if current_beat < len(beats):
        remaining = len(beats) - current_beat
        phrase_boundaries.append(duration)
        phrase_lengths_beats.append(float(remaining))

    # Deduplicate end boundary
    if len(phrase_boundaries) > 1 and abs(phrase_boundaries[-1] - phrase_boundaries[-2]) < 0.05:
        phrase_boundaries.pop()

    n_phrases = len(phrase_lengths_beats)

    # Regularity: what fraction of phrases match the typical length?
    if n_phrases > 0:
        counts = Counter(int(round(pl)) for pl in phrase_lengths_beats)
        most_common_len, most_common_count = counts.most_common(1)[0]
        regularity = most_common_count / n_phrases
        typical_phrase_beats = float(most_common_len)
    else:
        regularity = 1.0
        typical_phrase_beats = float(best_phrase_len)

    # Find irregular phrases (those that differ from typical length)
    irregular = [
        i
        for i, pl in enumerate(phrase_lengths_beats)
        if abs(pl - typical_phrase_beats) > 0.5
    ]

    return PhraseFeatures(
        phrase_boundaries=[round(t, 3) for t in phrase_boundaries],
        phrase_lengths_beats=[round(pl, 1) for pl in phrase_lengths_beats],
        n_phrases=n_phrases,
        typical_phrase_beats=typical_phrase_beats,
        regularity=round(regularity, 3),
        irregular_phrases=irregular,
    )


def _find_best_phrase_length(
    beat_strengths: np.ndarray, candidates: list[int]
) -> int:
    """Score candidate phrase lengths by downbeat emphasis.

    A good phrase length means beat 1 of each phrase tends to be stronger
    than other beats (the downbeat effect).
    """
    best_score = -1.0
    best_len = candidates[0]

    for phrase_len in candidates:
        if phrase_len > len(beat_strengths) // 2:
            continue

        # Average strength at each position within the phrase
        n_full_phrases = len(beat_strengths) // phrase_len
        if n_full_phrases < 2:
            continue

        trimmed = beat_strengths[: n_full_phrases * phrase_len]
        reshaped = trimmed.reshape(n_full_phrases, phrase_len)
        position_means = reshaped.mean(axis=0)

        # Score: how much stronger is beat 1 vs the average?
        if position_means.mean() > 0:
            score = position_means[0] / (position_means.mean() + 1e-8)
        else:
            score = 0.0

        if score > best_score:
            best_score = score
            best_len = phrase_len

    return best_len
