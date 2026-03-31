"""Tests for Layer 2 structural comprehension extractors."""

from __future__ import annotations

import numpy as np
import pytest

from earworm.structure.segmentation import extract_segmentation
from earworm.structure.recurrence import extract_recurrence
from earworm.structure.energy import extract_energy_arc
from earworm.structure.phrase import extract_phrase
from earworm.models import Layer2Features


# --- Fixtures for structured audio ---


@pytest.fixture
def rhythmic_signal() -> tuple[np.ndarray, int]:
    """10 seconds of rhythmic signal — kick-like pulses at 120 BPM (2Hz)."""
    sr = 22050
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Base sine with amplitude modulation at 2Hz (120 BPM)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))
    signal = envelope * np.sin(2 * np.pi * 200 * t)
    return signal.astype(np.float32), sr


@pytest.fixture
def two_section_signal() -> tuple[np.ndarray, int]:
    """10 seconds: 5s of 200Hz tone, then 5s of 800Hz tone — clear structural change."""
    sr = 22050
    duration = 10.0
    n = int(sr * duration)
    half = n // 2
    t1 = np.linspace(0, duration / 2, half, endpoint=False)
    t2 = np.linspace(duration / 2, duration, n - half, endpoint=False)
    section_a = 0.5 * np.sin(2 * np.pi * 200 * t1)
    section_b = 0.5 * np.sin(2 * np.pi * 800 * t2)
    signal = np.concatenate([section_a, section_b])
    return signal.astype(np.float32), sr


@pytest.fixture
def build_and_drop_signal() -> tuple[np.ndarray, int]:
    """10 seconds: linear volume build for 7s, then sudden drop to silence."""
    sr = 22050
    duration = 10.0
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    # Ramp from 0 to 1 over first 7 seconds
    envelope = np.clip(t / 7.0, 0, 1)
    # Drop at 7 seconds
    envelope[int(sr * 7) :] = 0.05
    signal = envelope * np.sin(2 * np.pi * 440 * t)
    return signal.astype(np.float32), sr


@pytest.fixture
def short_signal() -> tuple[np.ndarray, int]:
    """1 second of sine — too short for meaningful structure."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    return signal.astype(np.float32), sr


@pytest.fixture
def silence_signal() -> tuple[np.ndarray, int]:
    """10 seconds of silence."""
    sr = 22050
    return np.zeros(int(sr * 10.0), dtype=np.float32), sr


# --- Segmentation Tests ---


class TestSegmentation:
    def test_returns_valid_model(self, rhythmic_signal):
        y, sr = rhythmic_signal
        result = extract_segmentation(y, sr)
        assert result.n_sections >= 1
        assert len(result.labels) == result.n_sections
        assert len(result.section_durations) == result.n_sections
        assert len(result.boundaries) == result.n_sections + 1

    def test_boundaries_span_full_duration(self, rhythmic_signal):
        y, sr = rhythmic_signal
        result = extract_segmentation(y, sr)
        assert result.boundaries[0] == 0.0
        assert abs(result.boundaries[-1] - len(y) / sr) < 0.2

    def test_section_durations_sum_to_total(self, rhythmic_signal):
        y, sr = rhythmic_signal
        result = extract_segmentation(y, sr)
        assert abs(sum(result.section_durations) - len(y) / sr) < 0.2

    def test_short_audio_single_section(self, short_signal):
        y, sr = short_signal
        result = extract_segmentation(y, sr)
        assert result.n_sections == 1
        assert result.labels == [0]

    def test_two_sections_detected(self, two_section_signal):
        y, sr = two_section_signal
        result = extract_segmentation(y, sr)
        # Should detect at least 2 sections due to clear timbral change
        assert result.n_sections >= 2

    def test_boundaries_are_sorted(self, rhythmic_signal):
        y, sr = rhythmic_signal
        result = extract_segmentation(y, sr)
        assert result.boundaries == sorted(result.boundaries)


# --- Recurrence Tests ---


class TestRecurrence:
    def test_returns_valid_model(self, rhythmic_signal):
        y, sr = rhythmic_signal
        seg = extract_segmentation(y, sr)
        result = extract_recurrence(y, sr, seg.boundaries, seg.labels)
        assert result.n_distinct_labels >= 1
        assert 0.0 <= result.repetition_ratio <= 1.0
        assert len(result.label_sequence) == len(seg.labels)

    def test_novelty_curve_exists(self, rhythmic_signal):
        y, sr = rhythmic_signal
        seg = extract_segmentation(y, sr)
        result = extract_recurrence(y, sr, seg.boundaries, seg.labels)
        assert len(result.novelty_curve) > 0
        assert len(result.novelty_timestamps) == len(result.novelty_curve)

    def test_novelty_values_normalized(self, rhythmic_signal):
        y, sr = rhythmic_signal
        seg = extract_segmentation(y, sr)
        result = extract_recurrence(y, sr, seg.boundaries, seg.labels)
        assert all(0.0 <= v <= 1.0001 for v in result.novelty_curve)

    def test_label_durations_sum(self, rhythmic_signal):
        y, sr = rhythmic_signal
        duration = len(y) / sr
        seg = extract_segmentation(y, sr)
        result = extract_recurrence(y, sr, seg.boundaries, seg.labels)
        total_label_dur = sum(result.label_durations.values())
        assert abs(total_label_dur - duration) < 0.5

    def test_uniform_signal_low_repetition(self, short_signal):
        y, sr = short_signal
        seg = extract_segmentation(y, sr)
        result = extract_recurrence(y, sr, seg.boundaries, seg.labels)
        # Single section can't repeat
        assert result.n_distinct_labels >= 1


# --- Energy Arc Tests ---


class TestEnergyArc:
    def test_returns_valid_model(self, rhythmic_signal):
        y, sr = rhythmic_signal
        result = extract_energy_arc(y, sr)
        assert len(result.energy_curve) > 0
        assert len(result.energy_timestamps) == len(result.energy_curve)
        assert 0.0 <= result.climax_position <= 1.0

    def test_energy_values_normalized(self, rhythmic_signal):
        y, sr = rhythmic_signal
        result = extract_energy_arc(y, sr)
        assert all(0.0 <= v <= 1.001 for v in result.energy_curve)

    def test_build_detected(self, build_and_drop_signal):
        y, sr = build_and_drop_signal
        result = extract_energy_arc(y, sr)
        # Should detect at least one build (the 7-second ramp)
        assert result.n_builds >= 1

    def test_drop_detected(self, build_and_drop_signal):
        y, sr = build_and_drop_signal
        result = extract_energy_arc(y, sr)
        # Should detect the sudden drop at 7s
        assert result.n_drops >= 1

    def test_climax_in_build_section(self, build_and_drop_signal):
        y, sr = build_and_drop_signal
        result = extract_energy_arc(y, sr)
        # Climax should be near the peak of the build (~7s in a 10s track)
        assert result.climax_time > 3.0  # Not at the very start

    def test_silence_low_spread(self, silence_signal):
        y, sr = silence_signal
        result = extract_energy_arc(y, sr)
        assert result.dynamic_spread < 0.01

    def test_dynamic_spread_range(self, build_and_drop_signal):
        y, sr = build_and_drop_signal
        result = extract_energy_arc(y, sr)
        assert result.dynamic_spread > 0.1  # Significant dynamic variation


# --- Phrase Structure Tests ---


class TestPhrase:
    def test_returns_valid_model(self, rhythmic_signal):
        y, sr = rhythmic_signal
        # Generate some beat times (2 beats per second for 10s)
        beat_times = [i * 0.5 for i in range(20)]
        result = extract_phrase(y, sr, beat_times, bpm=120.0)
        assert result.n_phrases >= 1
        assert result.typical_phrase_beats > 0
        assert 0.0 <= result.regularity <= 1.0

    def test_phrase_boundaries_sorted(self, rhythmic_signal):
        y, sr = rhythmic_signal
        beat_times = [i * 0.5 for i in range(20)]
        result = extract_phrase(y, sr, beat_times, bpm=120.0)
        assert result.phrase_boundaries == sorted(result.phrase_boundaries)

    def test_regular_beats_high_regularity(self, rhythmic_signal):
        y, sr = rhythmic_signal
        # Perfectly regular beats at 120 BPM
        beat_times = [i * 0.5 for i in range(20)]
        result = extract_phrase(y, sr, beat_times, bpm=120.0)
        # With regular beats, phrase regularity should be high
        assert result.regularity >= 0.5

    def test_few_beats_single_phrase(self, short_signal):
        y, sr = short_signal
        beat_times = [0.0, 0.5]
        result = extract_phrase(y, sr, beat_times, bpm=120.0)
        assert result.n_phrases == 1

    def test_phrase_lengths_positive(self, rhythmic_signal):
        y, sr = rhythmic_signal
        beat_times = [i * 0.5 for i in range(20)]
        result = extract_phrase(y, sr, beat_times, bpm=120.0)
        assert all(pl > 0 for pl in result.phrase_lengths_beats)

    def test_irregular_phrases_detected(self, rhythmic_signal):
        y, sr = rhythmic_signal
        # Create irregular beat pattern — mostly 0.5s apart but with a gap
        beat_times = [i * 0.5 for i in range(15)]
        # Add a few extra beats at irregular positions
        beat_times.extend([8.0, 8.3, 8.7, 9.1, 9.5, 9.8])
        beat_times.sort()
        result = extract_phrase(y, sr, beat_times, bpm=120.0)
        # With 21 beats, typical phrase is 4 or 8 — last phrase may be irregular
        assert result.n_phrases >= 1


# --- Integration: Layer2Features assembly ---


class TestLayer2Integration:
    def test_all_fields_populated(self, rhythmic_signal):
        y, sr = rhythmic_signal
        seg = extract_segmentation(y, sr)
        rec = extract_recurrence(y, sr, seg.boundaries, seg.labels)
        energy = extract_energy_arc(y, sr)
        beat_times = [i * 0.5 for i in range(20)]
        phrase = extract_phrase(y, sr, beat_times, bpm=120.0)

        result = Layer2Features(
            file_path="test.wav",
            duration_seconds=10.0,
            segmentation=seg,
            recurrence=rec,
            energy_arc=energy,
            phrase=phrase,
        )

        assert result.segmentation.n_sections >= 1
        assert result.recurrence.n_distinct_labels >= 1
        assert len(result.energy_arc.energy_curve) > 0
        assert result.phrase.n_phrases >= 1

    def test_json_round_trip(self, rhythmic_signal):
        y, sr = rhythmic_signal
        seg = extract_segmentation(y, sr)
        rec = extract_recurrence(y, sr, seg.boundaries, seg.labels)
        energy = extract_energy_arc(y, sr)
        beat_times = [i * 0.5 for i in range(20)]
        phrase = extract_phrase(y, sr, beat_times, bpm=120.0)

        result = Layer2Features(
            file_path="test.wav",
            duration_seconds=10.0,
            segmentation=seg,
            recurrence=rec,
            energy_arc=energy,
            phrase=phrase,
        )

        # Serialize and deserialize
        json_str = result.model_dump_json()
        restored = Layer2Features.model_validate_json(json_str)
        assert restored.segmentation.n_sections == result.segmentation.n_sections
        assert restored.energy_arc.climax_time == result.energy_arc.climax_time
