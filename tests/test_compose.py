"""Tests for earworm compose — structural response pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import scipy.io.wavfile

from earworm.compose import compose
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
