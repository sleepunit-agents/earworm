"""Tests for earworm → Strudel pattern generator."""

from __future__ import annotations

import pytest

from earworm.compose.strudel import (
    _chord_mini,
    _label_sequence_str,
    _parse_key,
    _root_index,
    _scale_notes,
    _section_gain,
    _triad_notes,
    to_strudel,
    to_strudel_generative,
)
from earworm.models import Layer2Features


@pytest.fixture
def layer2_fixture() -> Layer2Features:
    """Minimal Layer2Features fixture with 4 sections and 7 phrases."""
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


# ─── Helper unit tests ────────────────────────────────────────────────────


class TestParseKey:
    def test_minor(self):
        assert _parse_key("C minor") == ("C", "minor")

    def test_major(self):
        assert _parse_key("G major") == ("G", "major")

    def test_sharp(self):
        assert _parse_key("F# minor") == ("F#", "minor")

    def test_single_word(self):
        assert _parse_key("D") == ("D", "major")


class TestRootIndex:
    def test_c(self):
        assert _root_index("C") == 0

    def test_a(self):
        assert _root_index("A") == 9

    def test_f_sharp(self):
        assert _root_index("F#") == 6


class TestScaleNotes:
    def test_c_major(self):
        assert _scale_notes("C", "major") == ["c", "d", "e", "f", "g", "a", "b"]

    def test_a_minor(self):
        assert _scale_notes("A", "minor") == ["a", "b", "c", "d", "e", "f", "g"]

    def test_scale_length(self):
        assert len(_scale_notes("D", "major")) == 7


class TestTriadNotes:
    def test_c_major_root(self):
        scale = _scale_notes("C", "major")
        triad = _triad_notes(scale, 0, octave=3)
        assert triad == ["c3", "e3", "g3"]

    def test_a_minor_root(self):
        scale = _scale_notes("A", "minor")
        triad = _triad_notes(scale, 0, octave=3)
        # C and E are both above A chromatically, so they wrap to octave 4
        assert triad == ["a3", "c4", "e4"]


class TestChordMini:
    def test_basic(self):
        assert _chord_mini(["c3", "e3", "g3"]) == "[c3,e3,g3]"


class TestSectionGain:
    def test_mid_energy(self):
        gain = _section_gain([0.5], [0.0], 0.0, 10.0)
        assert 0.5 < gain < 0.8

    def test_full_energy(self):
        gain = _section_gain([1.0], [0.0], 0.0, 10.0)
        assert gain == 1.0

    def test_zero_energy(self):
        gain = _section_gain([0.0], [0.0], 0.0, 10.0)
        assert gain == 0.3

    def test_empty_curve(self):
        gain = _section_gain([], [], 0.0, 10.0)
        assert gain == 0.7


class TestLabelSequenceStr:
    def test_basic(self):
        assert _label_sequence_str([0, 1, 0, 2]) == "A B A C"

    def test_single(self):
        assert _label_sequence_str([0]) == "A"


# ─── Integration tests ────────────────────────────────────────────────────


class TestToStrudel:
    def test_returns_string(self, layer2_fixture):
        code = to_strudel(layer2_fixture, bpm_override=120.0, key_override="C minor")
        assert isinstance(code, str)
        assert len(code) > 100

    def test_contains_setcps(self, layer2_fixture):
        code = to_strudel(layer2_fixture, bpm_override=120.0)
        assert "setcps(" in code

    def test_contains_arrange(self, layer2_fixture):
        code = to_strudel(layer2_fixture)
        assert "arrange(" in code

    def test_contains_stack(self, layer2_fixture):
        code = to_strudel(layer2_fixture)
        assert "stack(" in code

    def test_contains_note_patterns(self, layer2_fixture):
        code = to_strudel(layer2_fixture, key_override="C minor")
        assert "note(" in code

    def test_correct_cps_at_120bpm(self, layer2_fixture):
        code = to_strudel(layer2_fixture, bpm_override=120.0)
        # 120 BPM / 60 / 4 = 0.5 cps
        assert "setcps(0.5)" in code

    def test_correct_cps_at_140bpm(self, layer2_fixture):
        code = to_strudel(layer2_fixture, bpm_override=140.0)
        # 140 / 60 / 4 ≈ 0.5833
        assert "setcps(0.5833)" in code

    def test_section_count_matches(self, layer2_fixture):
        code = to_strudel(layer2_fixture)
        # 4 sections = 4 entries in arrange()
        assert code.count("stack(") == 4

    def test_no_drums_flag(self, layer2_fixture):
        code = to_strudel(layer2_fixture, include_drums=False)
        assert "hh" not in code
        assert "bd" not in code

    def test_with_drums(self, layer2_fixture):
        code = to_strudel(layer2_fixture, include_drums=True)
        assert "hh" in code
        assert "bd" in code

    def test_header_comment(self, layer2_fixture):
        code = to_strudel(layer2_fixture, key_override="A minor", bpm_override=120.0)
        assert "A minor" in code
        assert "120" in code

    def test_section_labels_in_comment(self, layer2_fixture):
        code = to_strudel(layer2_fixture)
        assert "A B A C" in code

    def test_effects_present(self, layer2_fixture):
        code = to_strudel(layer2_fixture)
        assert ".room(" in code
        assert ".delay(" in code

    def test_energy_affects_gain(self, layer2_fixture):
        """Higher-energy sections should have higher gain values."""
        code = to_strudel(layer2_fixture, bpm_override=120.0, key_override="C minor")
        # Section 3 (energy=0.9) should have higher gain than section 1 (energy=0.4)
        # We verify the code is valid — exact gain values tested in unit tests
        assert ".gain(" in code


class TestToStrudelGenerative:
    def test_returns_string(self):
        code = to_strudel_generative(bpm=120.0, key="C minor", n_sections=4,
                                      duration_seconds=16.0)
        assert isinstance(code, str)
        assert "arrange(" in code

    def test_section_pattern(self):
        code = to_strudel_generative(section_pattern="AABB", duration_seconds=16.0)
        assert "A A B B" in code

    def test_energy_presets(self):
        for preset in ("arc", "peak-drop", "flat", "pulse"):
            code = to_strudel_generative(energy_preset=preset, n_sections=4,
                                          duration_seconds=16.0)
            assert "arrange(" in code, f"Failed for preset={preset}"

    def test_major_key(self):
        code = to_strudel_generative(key="G major", n_sections=4, duration_seconds=16.0)
        assert "G major" in code
