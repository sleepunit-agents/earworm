"""Tests for Layer 3 quality assessment extractors."""

from __future__ import annotations

import numpy as np
import pytest

from earworm.quality.technical import extract_technical
from earworm.quality.mix import extract_mix
from earworm.quality.mastering import extract_mastering
from earworm.quality.composition import extract_composition
from earworm.models import LoudnessFeatures, Layer3Features


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture
def clean_sine() -> tuple[np.ndarray, int]:
    """5 seconds of clean 440Hz sine at moderate level."""
    sr = 22050
    t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    return y.astype(np.float32), sr


@pytest.fixture
def clipped_sine() -> tuple[np.ndarray, int]:
    """5 seconds of hard-clipped sine — severe clipping."""
    sr = 22050
    t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    y = 1.5 * np.sin(2 * np.pi * 440 * t)
    y = np.clip(y, -1.0, 1.0)
    return y.astype(np.float32), sr


@pytest.fixture
def dc_offset_sine() -> tuple[np.ndarray, int]:
    """5 seconds of sine with significant DC offset."""
    sr = 22050
    t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1  # DC offset of 0.1
    return y.astype(np.float32), sr


@pytest.fixture
def multiband_signal() -> tuple[np.ndarray, int]:
    """5 seconds with energy across low, mid, and high bands."""
    sr = 22050
    t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    low = 0.3 * np.sin(2 * np.pi * 100 * t)
    mid = 0.3 * np.sin(2 * np.pi * 1000 * t)
    high = 0.3 * np.sin(2 * np.pi * 6000 * t)
    y = low + mid + high
    return y.astype(np.float32), sr


@pytest.fixture
def low_heavy_signal() -> tuple[np.ndarray, int]:
    """5 seconds dominated by low frequency."""
    sr = 22050
    t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    y = 0.8 * np.sin(2 * np.pi * 80 * t) + 0.05 * np.sin(2 * np.pi * 2000 * t)
    return y.astype(np.float32), sr


@pytest.fixture
def stereo_signal() -> tuple[np.ndarray, int]:
    """5 seconds stereo: different content L/R."""
    sr = 22050
    t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    y = np.stack([left, right]).astype(np.float32)
    return y, sr


@pytest.fixture
def rhythmic_complex() -> tuple[np.ndarray, int]:
    """10 seconds of complex rhythmic signal with harmonic changes."""
    sr = 22050
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Base tone that changes pitch halfway through
    freq = np.where(t < 5.0, 440.0, 660.0)
    phase = np.cumsum(freq / sr) * 2 * np.pi
    tone = 0.3 * np.sin(phase)
    # Rhythmic modulation
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))
    y = tone * envelope
    return y.astype(np.float32), sr


@pytest.fixture
def silence() -> tuple[np.ndarray, int]:
    """5 seconds of silence."""
    sr = 22050
    return np.zeros(int(sr * 5.0), dtype=np.float32), sr


@pytest.fixture
def fake_loudness() -> LoudnessFeatures:
    """Fake Layer 1 loudness features for mastering tests."""
    return LoudnessFeatures(
        lufs_integrated=-14.0,
        lufs_short_term_max=-10.0,
        lufs_range=6.0,
        dynamic_range_db=10.0,
        peak_db=-1.0,
        rms_db=-11.0,
        crest_factor_db=10.0,
        loudness_curve=[-15.0, -13.0, -14.0, -12.0, -14.0],
    )


@pytest.fixture
def compressed_loudness() -> LoudnessFeatures:
    """Fake loudness features for over-compressed audio."""
    return LoudnessFeatures(
        lufs_integrated=-8.0,
        lufs_short_term_max=-7.0,
        lufs_range=2.0,
        dynamic_range_db=4.0,
        peak_db=-0.1,
        rms_db=-4.1,
        crest_factor_db=4.0,
        loudness_curve=[-8.0, -8.5, -7.5, -8.0, -8.2],
    )


# --- Technical Quality Tests ------------------------------------------------


class TestTechnical:
    def test_clean_no_clipping(self, clean_sine):
        y, sr = clean_sine
        result = extract_technical(y, sr)
        assert result.clipping_ratio == 0.0
        assert result.clipping_regions == 0

    def test_clipped_detects_clipping(self, clipped_sine):
        y, sr = clipped_sine
        result = extract_technical(y, sr)
        assert result.clipping_ratio > 0.01
        assert result.clipping_regions > 0

    def test_dc_offset_detected(self, dc_offset_sine):
        y, sr = dc_offset_sine
        result = extract_technical(y, sr)
        assert result.has_dc_offset
        assert abs(result.dc_offset) > 0.005

    def test_clean_no_dc_offset(self, clean_sine):
        y, sr = clean_sine
        result = extract_technical(y, sr)
        assert not result.has_dc_offset

    def test_noise_floor_below_signal(self, clean_sine):
        y, sr = clean_sine
        result = extract_technical(y, sr)
        assert result.noise_floor_db < 0  # Below 0 dBFS

    def test_silence_very_low_noise_floor(self, silence):
        y, sr = silence
        result = extract_technical(y, sr)
        assert result.noise_floor_db < -50

    def test_frequency_balance_score_range(self, clean_sine):
        y, sr = clean_sine
        result = extract_technical(y, sr)
        assert 0.0 <= result.frequency_balance_score <= 1.0


# --- Mix Quality Tests -------------------------------------------------------


class TestMix:
    def test_band_ratios_sum_to_one(self, multiband_signal):
        y, sr = multiband_signal
        result = extract_mix(y, sr)
        total = result.low_ratio + result.mid_ratio + result.high_ratio
        assert abs(total - 1.0) < 0.01

    def test_low_heavy_high_low_ratio(self, low_heavy_signal):
        y, sr = low_heavy_signal
        result = extract_mix(y, sr)
        assert result.low_ratio > result.high_ratio

    def test_multiband_balanced(self, multiband_signal):
        y, sr = multiband_signal
        result = extract_mix(y, sr)
        # All three bands should have meaningful energy
        assert result.low_ratio > 0.05
        assert result.mid_ratio > 0.05
        assert result.high_ratio > 0.05

    def test_stereo_width_score(self, stereo_signal):
        y_stereo, sr = stereo_signal
        y_mono = (y_stereo[0] + y_stereo[1]) / 2.0
        result = extract_mix(y_mono, sr, y_stereo=y_stereo)
        assert result.stereo_width_score > 0.0

    def test_mono_zero_stereo_width(self, clean_sine):
        y, sr = clean_sine
        result = extract_mix(y, sr, y_stereo=y)
        assert result.stereo_width_score == 0.0

    def test_spectral_balance_range(self, multiband_signal):
        y, sr = multiband_signal
        result = extract_mix(y, sr)
        assert 0.0 <= result.spectral_balance_score <= 1.0

    def test_clarity_scores_range(self, clean_sine):
        y, sr = clean_sine
        result = extract_mix(y, sr)
        assert 0.0 <= result.low_end_clarity <= 1.0
        assert 0.0 <= result.high_frequency_clarity <= 1.0


# --- Mastering Quality Tests -------------------------------------------------


class TestMastering:
    def test_on_target_lufs(self, clean_sine, fake_loudness):
        y, sr = clean_sine
        result = extract_mastering(y, sr, fake_loudness)
        assert result.lufs_deviation_from_target < 1.0

    def test_compressed_low_dynamic_range(self, clean_sine, compressed_loudness):
        y, sr = clean_sine
        result = extract_mastering(y, sr, compressed_loudness)
        assert result.dynamic_range_score < 0.5

    def test_good_dynamic_range(self, clean_sine, fake_loudness):
        y, sr = clean_sine
        result = extract_mastering(y, sr, fake_loudness)
        assert result.dynamic_range_score > 0.5

    def test_loudness_consistency_range(self, clean_sine, fake_loudness):
        y, sr = clean_sine
        result = extract_mastering(y, sr, fake_loudness)
        assert 0.0 <= result.loudness_consistency <= 1.0

    def test_limiter_clean_signal(self, clean_sine, fake_loudness):
        y, sr = clean_sine
        result = extract_mastering(y, sr, fake_loudness)
        assert result.limiter_artifact_score < 0.1

    def test_limiter_clipped_signal(self, clipped_sine, fake_loudness):
        y, sr = clipped_sine
        result = extract_mastering(y, sr, fake_loudness)
        assert result.limiter_artifact_score > 0.1


# --- Composition Quality Tests -----------------------------------------------


class TestComposition:
    def test_pure_tone_low_vocabulary(self, clean_sine):
        y, sr = clean_sine
        result = extract_composition(y, sr)
        # A pure sine has very limited harmonic content
        assert result.harmonic_vocabulary >= 1

    def test_complex_signal_higher_vocabulary(self, rhythmic_complex):
        y, sr = rhythmic_complex
        result = extract_composition(y, sr)
        # Signal with pitch changes should have more harmonic variety
        assert result.harmonic_vocabulary >= 1

    def test_rhythmic_variation_range(self, rhythmic_complex):
        y, sr = rhythmic_complex
        result = extract_composition(y, sr)
        assert 0.0 <= result.rhythmic_variation <= 1.0

    def test_melodic_range_positive(self, rhythmic_complex):
        y, sr = rhythmic_complex
        result = extract_composition(y, sr)
        # Signal with two different pitches should have measurable range
        assert result.melodic_range_semitones >= 0.0

    def test_structural_variety_range(self, rhythmic_complex):
        y, sr = rhythmic_complex
        result = extract_composition(y, sr)
        assert 0.0 <= result.structural_variety <= 1.0

    def test_with_segment_labels(self, rhythmic_complex):
        y, sr = rhythmic_complex
        # Provide segment labels for structural variety
        result = extract_composition(y, sr, segment_labels=[0, 1, 0, 1])
        assert 0.0 <= result.structural_variety <= 1.0

    def test_silence_handles_gracefully(self, silence):
        y, sr = silence
        result = extract_composition(y, sr)
        assert result.harmonic_vocabulary >= 0
        assert result.rhythmic_variation >= 0.0


# --- Integration: Layer3Features assembly ------------------------------------


class TestLayer3Integration:
    def test_all_fields_populated(self, clean_sine, fake_loudness):
        y, sr = clean_sine
        tech = extract_technical(y, sr)
        mix = extract_mix(y, sr)
        mast = extract_mastering(y, sr, fake_loudness)
        comp = extract_composition(y, sr)

        result = Layer3Features(
            file_path="test.wav",
            duration_seconds=5.0,
            technical=tech,
            mix=mix,
            mastering=mast,
            composition=comp,
        )

        assert result.technical.clipping_ratio >= 0
        assert result.mix.spectral_balance_score >= 0
        assert result.mastering.dynamic_range_score >= 0
        assert result.composition.harmonic_vocabulary >= 0

    def test_json_round_trip(self, clean_sine, fake_loudness):
        y, sr = clean_sine
        tech = extract_technical(y, sr)
        mix = extract_mix(y, sr)
        mast = extract_mastering(y, sr, fake_loudness)
        comp = extract_composition(y, sr)

        result = Layer3Features(
            file_path="test.wav",
            duration_seconds=5.0,
            technical=tech,
            mix=mix,
            mastering=mast,
            composition=comp,
        )

        json_str = result.model_dump_json()
        restored = Layer3Features.model_validate_json(json_str)
        assert restored.technical.clipping_ratio == result.technical.clipping_ratio
        assert restored.mastering.lufs_integrated == result.mastering.lufs_integrated
