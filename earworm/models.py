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


# --- Layer 2: Structural Comprehension ---


class SegmentationFeatures(BaseModel):
    """Section boundaries and labels from structural segmentation."""

    boundaries: list[float]  # Boundary timestamps in seconds
    labels: list[int]  # Label per segment (same label = similar sections)
    n_sections: int  # Total number of detected sections
    section_durations: list[float]  # Duration of each section in seconds


class RecurrenceFeatures(BaseModel):
    """Self-similarity structure — what repeats, how much, and where."""

    n_distinct_labels: int  # Number of unique section types
    repetition_ratio: float  # Fraction of track that is repeated material (0-1)
    label_sequence: list[int]  # Section label sequence (e.g. [0,1,0,1,2,0] = ABABCA)
    label_durations: dict[int, float]  # Total duration per label in seconds
    novelty_curve: list[float]  # Novelty score over time (peaks = section changes)
    novelty_timestamps: list[float]  # Timestamps for novelty curve


class EnergyArcFeatures(BaseModel):
    """How intensity changes over time — builds, releases, climaxes."""

    energy_curve: list[float]  # Normalized energy envelope (0-1) sampled ~2Hz
    energy_timestamps: list[float]  # Timestamps for energy curve
    climax_time: float  # Timestamp of peak energy
    climax_position: float  # 0-1 position in track (0=start, 1=end)
    n_builds: int  # Number of sustained energy increases
    n_drops: int  # Number of sharp energy decreases
    build_times: list[float]  # Start timestamps of builds
    drop_times: list[float]  # Timestamps of drops
    dynamic_spread: float  # Range of energy curve (max - min, 0-1)


class PhraseFeatures(BaseModel):
    """Phrase groupings — regularity, typical length, surprises."""

    phrase_boundaries: list[float]  # Phrase boundary timestamps in seconds
    phrase_lengths_beats: list[float]  # Length of each phrase in beats
    n_phrases: int
    typical_phrase_beats: float  # Most common phrase length in beats
    regularity: float  # How regular the phrasing is (0-1, 1=perfectly regular)
    irregular_phrases: list[int]  # Indices of phrases that deviate from typical length


class Layer2Features(BaseModel):
    """Complete Layer 2 structural comprehension results."""

    file_path: str
    duration_seconds: float

    segmentation: SegmentationFeatures
    recurrence: RecurrenceFeatures
    energy_arc: EnergyArcFeatures
    phrase: PhraseFeatures


# --- Layer 3: Quality Assessment ---


class TechnicalQuality(BaseModel):
    """Technical audio quality — clipping, noise floor, frequency balance."""

    clipping_ratio: float  # Fraction of samples at or near ±1.0 (0-1)
    clipping_regions: int  # Number of continuous clipping regions
    dc_offset: float  # Mean of signal (should be near 0)
    noise_floor_db: float  # Estimated noise floor in dB
    frequency_balance_score: float  # Spectral tilt vs flat reference (0-1, 1=balanced)
    has_dc_offset: bool  # True if DC offset exceeds threshold


class MixQuality(BaseModel):
    """Mix quality — frequency distribution, stereo usage, clarity."""

    low_ratio: float  # Energy fraction below 200Hz
    mid_ratio: float  # Energy fraction 200Hz-4kHz
    high_ratio: float  # Energy fraction above 4kHz
    spectral_balance_score: float  # How evenly energy is distributed (0-1)
    stereo_width_score: float  # Quality of stereo field usage (0-1)
    low_end_clarity: float  # Low-frequency definition (0-1)
    high_frequency_clarity: float  # High-frequency definition (0-1)


class MasteringQuality(BaseModel):
    """Mastering quality — loudness compliance, dynamics, limiter artifacts."""

    lufs_integrated: float  # From Layer 1 (repeated for self-contained report)
    lufs_deviation_from_target: float  # Distance from -14 LUFS streaming target
    dynamic_range_score: float  # Quality of dynamic range (0-1)
    loudness_consistency: float  # How consistent loudness is across sections (0-1)
    limiter_artifact_score: float  # Detected limiter/clipping artifacts (0=none, 1=severe)
    crest_factor_db: float  # From Layer 1 (repeated)


class CompositionQuality(BaseModel):
    """Compositional quality — harmonic vocabulary, rhythmic variety, structure."""

    harmonic_vocabulary: int  # Number of distinct chord classes used
    chord_change_rate: float  # Chord changes per second
    rhythmic_variation: float  # Entropy of onset pattern (0-1, 1=highly varied)
    melodic_range_semitones: float  # Range of dominant pitch contour
    structural_variety: float  # How different sections are from each other (0-1)


class Layer3Features(BaseModel):
    """Complete Layer 3 quality assessment results."""

    file_path: str
    duration_seconds: float

    technical: TechnicalQuality
    mix: MixQuality
    mastering: MasteringQuality
    composition: CompositionQuality


# --- Phase 2: Voice (Interpretation) ---


class SampleReference(BaseModel):
    """A sample from samplebank that matched this track's perception."""

    sample_id: int
    filename: str
    path: str
    score: float  # Cosine similarity (0-1)
    match_source: str  # "text", "audio", or "combined"


class VoiceResult(BaseModel):
    """Natural language interpretation of a track — Art's subjective response."""

    file_path: str
    mode: str  # "quick" or "deep"
    description: str  # What the track sounds like
    opinion: str  # Subjective assessment
    tags: list[str]  # Genre, mood, texture, energy tags
    comparisons: list[str]  # Artist/track/genre comparisons
    highlights: list[str]  # What stands out positively
    concerns: list[str]  # Weaknesses or issues (can be empty)
    section_notes: str | None = None  # Deep mode only: section-by-section walkthrough
    related_samples: list[SampleReference] | None = None  # Samples matching this track's perception
