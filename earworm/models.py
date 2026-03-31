"""Data models for earworm analysis results.

Each Layer 1 feature extractor returns one of these models.
The full Layer1Features model aggregates them all.
"""

from __future__ import annotations

from pydantic import BaseModel


# --- Layer 1: Signal Features ---


class SpectralFeatures(BaseModel):
    """Spectral characteristics extracted from the audio signal."""

    # MFCCs — shape description, not raw arrays
    mfcc_mean: list[float]  # Mean of each MFCC coefficient across time
    mfcc_std: list[float]  # Std dev of each coefficient

    # Spectral shape descriptors (time-averaged)
    spectral_centroid_mean: float  # Brightness — higher = brighter
    spectral_centroid_std: float
    spectral_bandwidth_mean: float  # Frequency spread
    spectral_rolloff_mean: float  # Frequency below which 85% of energy sits
    spectral_contrast_mean: list[float]  # Per-band peak-valley contrast
    spectral_flatness_mean: float  # 0=tonal, 1=noisy

    # Chromagram — pitch class distribution
    chroma_mean: list[float]  # 12-element pitch class energy profile
    chroma_std: list[float]


class TemporalFeatures(BaseModel):
    """Rhythmic and timing characteristics."""

    bpm: float  # Estimated tempo
    bpm_confidence: float  # How strong the tempo estimate is (0-1)
    beat_times: list[float]  # Timestamps of detected beats (seconds)
    onset_times: list[float]  # Timestamps of note/event onsets
    onset_rate: float  # Onsets per second — rhythmic density
    tempo_stability: float  # How consistent the tempo is (0-1, 1=metronomic)


class HarmonicFeatures(BaseModel):
    """Tonal and harmonic characteristics."""

    key: str  # Estimated key (e.g. "C minor", "F# major")
    key_confidence: float  # Confidence in key estimate (0-1)
    key_profile: list[float]  # 12-element Krumhansl-Kessler profile correlation
    chroma_cqt_mean: list[float]  # Constant-Q chromagram (better for harmony)
    tonnetz_mean: list[float]  # 6-dim tonal centroid features
    tonnetz_std: list[float]
    harmonic_ratio: float  # Ratio of harmonic to percussive energy


class LoudnessFeatures(BaseModel):
    """Loudness and dynamic range characteristics."""

    lufs_integrated: float  # Integrated loudness (LUFS)
    lufs_short_term_max: float  # Max short-term loudness
    lufs_range: float  # Loudness range (LRA)
    dynamic_range_db: float  # Peak-to-RMS difference
    peak_db: float  # True peak level
    rms_db: float  # RMS level
    crest_factor_db: float  # Peak-to-RMS ratio — how punchy it is
    loudness_curve: list[float]  # RMS envelope sampled at ~1 Hz


class StereoFeatures(BaseModel):
    """Stereo field characteristics. Mono files get neutral defaults."""

    is_stereo: bool
    width_mean: float  # Average stereo width (0=mono, 1=full)
    width_std: float  # Width variation over time
    correlation_mean: float  # L/R correlation (1=mono, 0=uncorrelated, -1=out of phase)
    correlation_min: float  # Worst-case phase correlation
    mid_side_ratio: float  # Mid vs side energy ratio
    balance: float  # L/R balance (-1=left, 0=center, 1=right)


class Layer1Features(BaseModel):
    """Complete Layer 1 signal feature extraction results."""

    # Metadata
    file_path: str
    duration_seconds: float
    sample_rate: int
    channels: int

    # Feature groups
    spectral: SpectralFeatures
    temporal: TemporalFeatures
    harmonic: HarmonicFeatures
    loudness: LoudnessFeatures
    stereo: StereoFeatures
