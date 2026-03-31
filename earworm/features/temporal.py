"""Temporal feature extraction — rhythm, tempo, beat grid."""

from __future__ import annotations

import librosa
import numpy as np

from earworm.models import TemporalFeatures


def extract_temporal(y: np.ndarray, sr: int) -> TemporalFeatures:
    """Extract temporal/rhythmic features from a mono audio signal."""

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Tempo confidence from tempogram autocorrelation
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    # Confidence = strength of dominant tempo peak relative to mean
    tempo_profile = tempogram.mean(axis=1)
    if tempo_profile.max() > 0:
        bpm_confidence = float(tempo_profile.max() / (tempo_profile.mean() + 1e-8))
        bpm_confidence = min(bpm_confidence / 10.0, 1.0)  # Normalize to 0-1
    else:
        bpm_confidence = 0.0

    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    duration = len(y) / sr
    onset_rate = len(onset_times) / duration if duration > 0 else 0.0

    # Tempo stability — std of inter-beat intervals relative to mean
    if len(beat_times) > 1:
        ibis = np.diff(beat_times)
        ibi_mean = ibis.mean()
        if ibi_mean > 0:
            tempo_stability = 1.0 - min(float(ibis.std() / ibi_mean), 1.0)
        else:
            tempo_stability = 0.0
    else:
        tempo_stability = 0.0

    bpm = float(np.atleast_1d(tempo)[0])

    return TemporalFeatures(
        bpm=bpm,
        bpm_confidence=bpm_confidence,
        beat_times=beat_times.tolist(),
        onset_times=onset_times.tolist(),
        onset_rate=onset_rate,
        tempo_stability=tempo_stability,
    )
