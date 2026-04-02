"""Tests for earworm compose — structural response pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile

from earworm.compose import compose, compose_generative
from earworm.compose.composer import (
    _build_energy_curve,
    _build_synthetic_layer2,
    _default_section_pattern,
    _parse_section_pattern,
)
from earworm.models import Layer2Features


@pytest.fixture
def layer2_fixture() -> Layer2Features:
    """Minimal Layer2Features fixture with 4 sections and 6 phrases."""
    return Layer2Features.model_validate(
        {
            "file_path": "/test/fixture.wav",
            "duration_seconds": 32.0,
            "segmentation": {
                "boundaries": [0.0, 8.0, 16.0, 24.0],
                "labels": [0, 1, 0, 2],
                "n_sections": 4,
                "section_durations": [8.0, 8.0, 8.0, 8.0],
            },
            "recurrence": {
                "n_distinct_labels": 3,
                "repetition_ratio": 0.5,
                "label_sequence": [0, 1, 0, 2],
                "label_durations": {"0": 16.0, "1": 8.0, "2": 8.0},
                "novelty_curve": [0.2, 0.8, 0.3, 0.7],
                "novelty_timestamps": [0.0, 8.0, 16.0, 24.0],
            },
            "energy_arc": {
                "energy_curve": [0.4, 0.6, 0.9, 0.5],
                "energy_timestamps": [0.0, 8.0, 16.0, 24.0],
                "climax_time": 16.0,
                "climax_position": 0.5,
                "n_builds": 1,
                "n_drops": 1,
                "build_times": [8.0],
                "drop_times": [24.0],
                "dynamic_spread": 0.5,
            },
            "phrase": {
                "phrase_boundaries": [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0],
                "phrase_lengths_beats": [4.0] * 7,
                "n_phrases": 7,
                "typical_phrase_beats": 4.0,
                "regularity": 0.9,
                "irregular_phrases": [],
            },
        }
    )


def test_compose_produces_wav(layer2_fixture, tmp_path):
    """compose() writes a valid stereo WAV file."""
    output = tmp_path / "response.wav"
    manifest = compose(
        layer2_fixture,
        output,
        bpm_override=120.0,
        key_override="C minor",
        duration_override=10.0,
    )

    assert output.exists(), "WAV file was not created"
    rate, data = scipy.io.wavfile.read(str(output))
    assert rate == 44100
    assert data.ndim == 2  # stereo
    assert data.shape[1] == 2
    # Duration should be close to 10s + 1s tail
    duration_s = data.shape[0] / rate
    assert 10.0 <= duration_s <= 12.5


def test_compose_manifest_fields(layer2_fixture, tmp_path):
    """Manifest captures musical choices correctly."""
    output = tmp_path / "response.wav"
    manifest = compose(
        layer2_fixture,
        output,
        bpm_override=128.0,
        key_override="A minor",
        duration_override=10.0,
    )

    assert manifest.bpm == 128.0
    assert manifest.key == "A"
    assert manifest.mode == "minor"
    assert manifest.n_sections == 4
    assert manifest.label_sequence == [0, 1, 0, 2]
    assert set(manifest.chord_map.keys()) == {0, 1, 2}
    assert manifest.n_phrases > 0
    assert manifest.style == "edm"


def test_compose_default_key_and_bpm(layer2_fixture, tmp_path):
    """compose() uses sensible defaults when no L1 is provided."""
    output = tmp_path / "response.wav"
    manifest = compose(layer2_fixture, output, duration_override=5.0)

    # Should default to 120 BPM and A minor
    assert manifest.bpm == 120.0
    assert manifest.key == "A"
    assert manifest.mode == "minor"


def test_compose_major_key(layer2_fixture, tmp_path):
    """compose() works with major keys."""
    output = tmp_path / "response.wav"
    manifest = compose(
        layer2_fixture,
        output,
        key_override="D major",
        duration_override=5.0,
    )

    assert manifest.key == "D"
    assert manifest.mode == "major"
    assert output.exists()


# ─── compose_generative tests ──────────────────────────────────────────────


def test_compose_generative_produces_wav(tmp_path):
    """compose_generative() writes a valid stereo WAV file."""
    output = tmp_path / "gen.wav"
    manifest = compose_generative(
        output=output,
        bpm=120.0,
        key="C minor",
        n_sections=4,
        duration_seconds=8.0,
    )

    assert output.exists(), "WAV file was not created"
    rate, data = scipy.io.wavfile.read(str(output))
    assert rate == 44100
    assert data.ndim == 2  # stereo
    assert data.shape[0] > 0


def test_compose_generative_manifest_fields(tmp_path):
    """Manifest from generative mode has correct metadata."""
    output = tmp_path / "gen.wav"
    manifest = compose_generative(
        output=output,
        bpm=128.0,
        key="A minor",
        n_sections=8,
        duration_seconds=8.0,
    )

    assert manifest.bpm == 128.0
    assert manifest.key == "A"
    assert manifest.mode == "minor"
    assert manifest.n_sections == 8
    assert manifest.n_phrases > 0
    assert manifest.duration_seconds == 8.0


def test_compose_generative_section_pattern(tmp_path):
    """Section pattern is parsed and drives the label_sequence."""
    output = tmp_path / "gen.wav"
    manifest = compose_generative(
        output=output,
        section_pattern="AABABC",
        duration_seconds=6.0,
    )

    # Pattern AABABC → 6 sections
    assert manifest.n_sections == 6
    assert manifest.label_sequence == [0, 0, 1, 0, 1, 2]


def test_compose_generative_energy_arc(tmp_path):
    """All four energy presets produce valid output."""
    for preset in ("arc", "peak-drop", "flat", "pulse"):
        output = tmp_path / f"gen_{preset}.wav"
        manifest = compose_generative(
            output=output,
            energy_preset=preset,
            n_sections=4,
            duration_seconds=6.0,
        )
        assert output.exists(), f"No WAV for preset={preset}"
        assert manifest.n_sections == 4


def test_compose_generative_major_key(tmp_path):
    """compose_generative() works with major keys."""
    output = tmp_path / "gen.wav"
    manifest = compose_generative(
        output=output,
        key="G major",
        n_sections=4,
        duration_seconds=6.0,
    )

    assert manifest.key == "G"
    assert manifest.mode == "major"


# ─── helper unit tests ────────────────────────────────────────────────────


def test_parse_section_pattern_basic():
    """Letter pattern converts to integer labels correctly."""
    assert _parse_section_pattern("AABABC") == [0, 0, 1, 0, 1, 2]
    assert _parse_section_pattern("ABCD") == [0, 1, 2, 3]
    assert _parse_section_pattern("AAAA") == [0, 0, 0, 0]


def test_parse_section_pattern_lowercase():
    """Lowercase letters are treated as uppercase."""
    assert _parse_section_pattern("aAbB") == [0, 0, 1, 1]


def test_default_section_pattern_length():
    """Default pattern has exactly n_sections elements."""
    for n in (1, 4, 8, 13, 20):
        pattern = _default_section_pattern(n)
        assert len(pattern) == n, f"Expected {n}, got {len(pattern)}"


def test_build_energy_curve_arc_peaks_in_middle():
    """Arc preset reaches peak energy in the middle third."""
    curve, times = _build_energy_curve("arc", 100, 60.0, 8, 7.5)
    peak_pos = float(times[int(np.argmax(curve))]) / 60.0
    assert 0.30 <= peak_pos <= 0.80, f"Arc peak at {peak_pos:.2f}, expected [0.30, 0.80]"


def test_build_energy_curve_flat_is_constant():
    """Flat preset returns near-constant energy at 0.6."""
    curve, _ = _build_energy_curve("flat", 50, 30.0, 4, 7.5)
    assert np.allclose(curve, 0.6), "Flat preset should be constant 0.6"


def test_build_energy_curve_pulse_alternates():
    """Pulse preset has distinct high and low values."""
    curve, _ = _build_energy_curve("pulse", 80, 40.0, 4, 10.0)
    unique_vals = set(np.round(curve, 2))
    assert len(unique_vals) == 2, f"Pulse should have 2 distinct values, got {unique_vals}"


def test_build_energy_curve_peak_drop_drops_sharply():
    """Peak-drop preset has its maximum before 70% through the track."""
    curve, times = _build_energy_curve("peak-drop", 100, 60.0, 8, 7.5)
    peak_pos = float(times[int(np.argmax(curve))]) / 60.0
    assert peak_pos <= 0.70, f"Peak-drop peak at {peak_pos:.2f}, expected ≤ 0.70"


def test_build_synthetic_layer2_structure():
    """Synthetic Layer2 has correct section count and phrase boundaries."""
    layer2 = _build_synthetic_layer2(
        bpm=120.0,
        key="A minor",
        n_sections=4,
        section_pattern="ABAB",
        energy_preset="arc",
        duration_seconds=16.0,
    )

    assert layer2.segmentation.n_sections == 4
    assert layer2.recurrence.label_sequence == [0, 1, 0, 1]
    assert len(layer2.phrase.phrase_boundaries) > 0
    assert layer2.duration_seconds == 16.0
    # All phrase boundaries should be within [0, duration)
    for pb in layer2.phrase.phrase_boundaries:
        assert 0.0 <= pb < 16.0
