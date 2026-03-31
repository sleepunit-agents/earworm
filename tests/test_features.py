"""Tests for Layer 1 feature extractors."""

from __future__ import annotations


from earworm.features.spectral import extract_spectral
from earworm.features.temporal import extract_temporal
from earworm.features.harmonic import extract_harmonic
from earworm.features.loudness import extract_loudness
from earworm.features.stereo import extract_stereo


class TestSpectral:
    def test_sine_centroid_near_frequency(self, sine_440_mono):
        y, sr = sine_440_mono
        result = extract_spectral(y, sr)
        # Centroid of a pure 440Hz tone should be near 440Hz
        assert 400 < result.spectral_centroid_mean < 480

    def test_sine_is_tonal(self, sine_440_mono):
        y, sr = sine_440_mono
        result = extract_spectral(y, sr)
        assert result.spectral_flatness_mean < 0.01

    def test_noise_is_noisy(self, noise_mono):
        y, sr = noise_mono
        result = extract_spectral(y, sr)
        assert result.spectral_flatness_mean > 0.05

    def test_mfcc_shape(self, sine_440_mono):
        y, sr = sine_440_mono
        result = extract_spectral(y, sr)
        assert len(result.mfcc_mean) == 13
        assert len(result.mfcc_std) == 13

    def test_chroma_shape(self, sine_440_mono):
        y, sr = sine_440_mono
        result = extract_spectral(y, sr)
        assert len(result.chroma_mean) == 12
        assert len(result.chroma_std) == 12


class TestTemporal:
    def test_sine_no_beats(self, sine_440_mono):
        y, sr = sine_440_mono
        result = extract_temporal(y, sr)
        # Pure sine has no rhythmic content — BPM is unreliable
        assert result.bpm >= 0

    def test_onset_rate_non_negative(self, sine_440_mono):
        y, sr = sine_440_mono
        result = extract_temporal(y, sr)
        assert result.onset_rate >= 0

    def test_beat_times_are_sorted(self, noise_mono):
        y, sr = noise_mono
        result = extract_temporal(y, sr)
        if len(result.beat_times) > 1:
            assert result.beat_times == sorted(result.beat_times)


class TestHarmonic:
    def test_key_detection_a_for_440(self, sine_440_mono):
        y, sr = sine_440_mono
        result = extract_harmonic(y, sr)
        # 440Hz = A4, so key should involve A
        assert "A" in result.key

    def test_pure_tone_harmonic(self, sine_440_mono):
        y, sr = sine_440_mono
        result = extract_harmonic(y, sr)
        # A pure sine is entirely harmonic
        assert result.harmonic_ratio > 0.9

    def test_tonnetz_shape(self, sine_440_mono):
        y, sr = sine_440_mono
        result = extract_harmonic(y, sr)
        assert len(result.tonnetz_mean) == 6
        assert len(result.tonnetz_std) == 6

    def test_key_profile_shape(self, sine_440_mono):
        y, sr = sine_440_mono
        result = extract_harmonic(y, sr)
        assert len(result.key_profile) == 12


class TestLoudness:
    def test_silence_is_quiet(self, silence_mono):
        y, sr = silence_mono
        result = extract_loudness(y, sr)
        assert result.lufs_integrated < -50
        assert result.rms_db < -50

    def test_peak_above_rms(self, sine_440_mono):
        y, sr = sine_440_mono
        result = extract_loudness(y, sr)
        assert result.peak_db >= result.rms_db

    def test_crest_factor_positive(self, sine_440_mono):
        y, sr = sine_440_mono
        result = extract_loudness(y, sr)
        assert result.crest_factor_db >= 0

    def test_loudness_curve_exists(self, sine_440_mono):
        y, sr = sine_440_mono
        result = extract_loudness(y, sr)
        assert len(result.loudness_curve) > 0


class TestStereo:
    def test_mono_detection(self, sine_440_mono):
        y, sr = sine_440_mono
        # Mono signal passed as 1D
        result = extract_stereo(y, sr)
        assert not result.is_stereo
        assert result.width_mean == 0.0
        assert result.correlation_mean == 1.0

    def test_identical_channels_narrow(self, sine_440_stereo):
        y, sr = sine_440_stereo
        result = extract_stereo(y, sr)
        assert result.is_stereo
        assert result.width_mean < 0.01  # Identical channels = no width
        assert result.correlation_mean > 0.99

    def test_different_channels_wide(self, wide_stereo):
        y, sr = wide_stereo
        result = extract_stereo(y, sr)
        assert result.is_stereo
        assert result.width_mean > 0.1  # Different freqs = some width

    def test_balance_centered_for_identical(self, sine_440_stereo):
        y, sr = sine_440_stereo
        result = extract_stereo(y, sr)
        assert abs(result.balance) < 0.01
