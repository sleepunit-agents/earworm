"""Tests for Voice — Phase 2 interpretation layer.

Tests prompt construction, response parsing, provider abstraction,
and the interpret orchestrator. LLM calls are mocked.
"""

from __future__ import annotations

import json

import pytest

from earworm.models import (
    CompositionQuality,
    EnergyArcFeatures,
    HarmonicFeatures,
    Layer1Features,
    Layer2Features,
    Layer3Features,
    LoudnessFeatures,
    MasteringQuality,
    MixQuality,
    PhraseFeatures,
    RecurrenceFeatures,
    SegmentationFeatures,
    SpectralFeatures,
    StereoFeatures,
    TechnicalQuality,
    TemporalFeatures,
    VoiceResult,
)
from earworm.voice.interpret import interpret, parse_voice_response
from earworm.voice.prompt import build_prompt, format_pipeline_data
from earworm.voice.provider import AnthropicProvider, OllamaProvider, get_provider


# --- Fixtures ---


@pytest.fixture
def sample_layer1() -> Layer1Features:
    """Realistic Layer 1 features for a hypothetical electronic track."""
    return Layer1Features(
        file_path="/music/test-track.wav",
        duration_seconds=240.0,
        sample_rate=44100,
        channels=2,
        spectral=SpectralFeatures(
            mfcc_mean=[0.0] * 13,
            mfcc_std=[1.0] * 13,
            spectral_centroid_mean=3200.0,
            spectral_centroid_std=800.0,
            spectral_bandwidth_mean=2800.0,
            spectral_rolloff_mean=7500.0,
            spectral_contrast_mean=[20.0] * 7,
            spectral_flatness_mean=0.05,
            chroma_mean=[0.5] * 12,
            chroma_std=[0.1] * 12,
        ),
        temporal=TemporalFeatures(
            bpm=128.0,
            bpm_confidence=0.92,
            beat_times=[i * 0.469 for i in range(512)],
            onset_times=[i * 0.234 for i in range(1024)],
            onset_rate=4.3,
            tempo_stability=0.88,
        ),
        harmonic=HarmonicFeatures(
            key="A minor",
            key_confidence=0.78,
            key_profile=[0.5] * 12,
            chroma_cqt_mean=[0.5] * 12,
            tonnetz_mean=[0.0] * 6,
            tonnetz_std=[0.1] * 6,
            harmonic_ratio=0.35,
        ),
        loudness=LoudnessFeatures(
            lufs_integrated=-8.5,
            lufs_short_term_max=-5.2,
            lufs_range=6.3,
            dynamic_range_db=12.0,
            peak_db=-0.3,
            rms_db=-12.3,
            crest_factor_db=12.0,
            loudness_curve=[0.5] * 240,
        ),
        stereo=StereoFeatures(
            is_stereo=True,
            width_mean=0.65,
            width_std=0.15,
            correlation_mean=0.72,
            correlation_min=0.3,
            mid_side_ratio=2.1,
            balance=0.01,
        ),
    )


@pytest.fixture
def sample_layer2() -> Layer2Features:
    return Layer2Features(
        file_path="/music/test-track.wav",
        duration_seconds=240.0,
        segmentation=SegmentationFeatures(
            boundaries=[0.0, 32.0, 64.0, 96.0, 128.0, 192.0, 224.0],
            labels=[0, 1, 0, 1, 2, 0],
            n_sections=6,
            section_durations=[32.0, 32.0, 32.0, 32.0, 64.0, 32.0],
        ),
        recurrence=RecurrenceFeatures(
            n_distinct_labels=3,
            repetition_ratio=0.67,
            label_sequence=[0, 1, 0, 1, 2, 0],
            label_durations={0: 96.0, 1: 64.0, 2: 64.0},
            novelty_curve=[0.5] * 100,
            novelty_timestamps=[i * 2.4 for i in range(100)],
        ),
        energy_arc=EnergyArcFeatures(
            energy_curve=[0.3, 0.4, 0.5, 0.7, 0.9, 0.6],
            energy_timestamps=[0.0, 48.0, 96.0, 144.0, 192.0, 224.0],
            climax_time=192.0,
            climax_position=0.8,
            n_builds=2,
            n_drops=1,
            build_times=[64.0, 128.0],
            drop_times=[192.0],
            dynamic_spread=0.6,
        ),
        phrase=PhraseFeatures(
            phrase_boundaries=[i * 16.0 for i in range(16)],
            phrase_lengths_beats=[64.0] * 15,
            n_phrases=15,
            typical_phrase_beats=64.0,
            regularity=0.92,
            irregular_phrases=[],
        ),
    )


@pytest.fixture
def sample_layer3() -> Layer3Features:
    return Layer3Features(
        file_path="/music/test-track.wav",
        duration_seconds=240.0,
        technical=TechnicalQuality(
            clipping_ratio=0.0001,
            clipping_regions=2,
            dc_offset=0.0003,
            noise_floor_db=-72.0,
            frequency_balance_score=0.85,
            has_dc_offset=False,
        ),
        mix=MixQuality(
            low_ratio=0.35,
            mid_ratio=0.45,
            high_ratio=0.20,
            spectral_balance_score=0.80,
            stereo_width_score=0.75,
            low_end_clarity=0.70,
            high_frequency_clarity=0.82,
        ),
        mastering=MasteringQuality(
            lufs_integrated=-8.5,
            lufs_deviation_from_target=5.5,
            dynamic_range_score=0.65,
            loudness_consistency=0.88,
            limiter_artifact_score=0.15,
            crest_factor_db=12.0,
        ),
        composition=CompositionQuality(
            harmonic_vocabulary=6,
            chord_change_rate=0.25,
            rhythmic_variation=0.55,
            melodic_range_semitones=7.0,
            structural_variety=0.45,
        ),
    )


# --- Mock Provider ---


class MockProvider:
    """LLM provider that returns canned responses for testing."""

    def __init__(self, response: str | None = None):
        self.calls: list[tuple[str, str]] = []
        self._response = response or json.dumps({
            "description": "A driving, hypnotic electronic track built on a four-on-the-floor "
            "kick pattern with layered synth textures and a relentless minor-key bassline.",
            "opinion": "Solid production with good energy management. The builds work well "
            "and the stereo field is used effectively. Could benefit from more harmonic "
            "variation in the B sections.",
            "tags": ["techno", "driving", "dark", "hypnotic", "minimal"],
            "comparisons": ["Ben Klock", "Surgeon", "early Shed"],
            "highlights": [
                "Energy arc builds effectively to a late climax",
                "Clean low end with good stereo separation",
            ],
            "concerns": [
                "Loudness is hot for streaming (-8.5 LUFS vs -14 target)",
                "Limited harmonic vocabulary (6 chord classes)",
            ],
        })

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self._response


# --- Prompt Construction Tests ---


class TestFormatPipelineData:
    def test_includes_signal_features(self, sample_layer1):
        output = format_pipeline_data(sample_layer1)
        assert "128.0 BPM" in output
        assert "A minor" in output
        assert "3200 Hz centroid" in output
        assert "-8.5 LUFS" in output

    def test_includes_structural_features(self, sample_layer1, sample_layer2):
        output = format_pipeline_data(sample_layer1, sample_layer2)
        assert "ABABCA" in output
        assert "6 sections" in output
        assert "climax at 192.0s" in output
        assert "92%" in output  # regularity

    def test_includes_quality_features(self, sample_layer1, sample_layer2, sample_layer3):
        output = format_pipeline_data(sample_layer1, sample_layer2, sample_layer3)
        assert "clipping" in output.lower()
        assert "spectral balance" in output.lower()
        assert "limiter" in output.lower()
        assert "6 chord classes" in output

    def test_layer1_only(self, sample_layer1):
        output = format_pipeline_data(sample_layer1)
        assert "Structural Comprehension" not in output
        assert "Quality Assessment" not in output

    def test_layer1_and_2(self, sample_layer1, sample_layer2):
        output = format_pipeline_data(sample_layer1, sample_layer2)
        assert "Structural Comprehension" in output
        assert "Quality Assessment" not in output

    def test_stereo_info(self, sample_layer1):
        output = format_pipeline_data(sample_layer1)
        assert "width" in output.lower()
        assert "0.65" in output

    def test_mono_handling(self, sample_layer1):
        sample_layer1.stereo.is_stereo = False
        output = format_pipeline_data(sample_layer1)
        assert "Mono source" in output


class TestBuildPrompt:
    def test_returns_tuple(self, sample_layer1):
        system, user = build_prompt(sample_layer1)
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_quick_mode_uses_standard_prompt(self, sample_layer1):
        system, _ = build_prompt(sample_layer1, mode="quick")
        assert "section_notes" not in system

    def test_deep_mode_uses_deep_prompt(self, sample_layer1):
        system, _ = build_prompt(sample_layer1, mode="deep")
        assert "section_notes" in system
        assert "section-by-section" in system.lower()

    def test_user_prompt_contains_data(self, sample_layer1):
        _, user = build_prompt(sample_layer1)
        assert "128.0 BPM" in user
        assert "A minor" in user


# --- Response Parsing Tests ---


class TestParseVoiceResponse:
    def test_parses_clean_json(self):
        raw = json.dumps({
            "description": "A dark techno track.",
            "opinion": "Solid but predictable.",
            "tags": ["techno", "dark"],
            "comparisons": ["Surgeon"],
            "highlights": ["Good energy"],
            "concerns": [],
        })
        result = parse_voice_response(raw, "/test.wav", "quick")
        assert isinstance(result, VoiceResult)
        assert result.description == "A dark techno track."
        assert result.mode == "quick"
        assert result.tags == ["techno", "dark"]
        assert result.concerns == []

    def test_strips_markdown_fences(self):
        raw = '```json\n{"description": "test", "opinion": "ok", "tags": [], "comparisons": [], "highlights": [], "concerns": []}\n```'
        result = parse_voice_response(raw, "/test.wav", "quick")
        assert result.description == "test"

    def test_strips_bare_fences(self):
        raw = '```\n{"description": "test", "opinion": "ok", "tags": [], "comparisons": [], "highlights": [], "concerns": []}\n```'
        result = parse_voice_response(raw, "/test.wav", "quick")
        assert result.description == "test"

    def test_deep_mode_with_section_notes(self):
        raw = json.dumps({
            "description": "A progressive track.",
            "opinion": "Masterful layering.",
            "tags": ["progressive", "trance"],
            "comparisons": ["Sasha"],
            "highlights": ["Layering"],
            "concerns": [],
            "section_notes": "Section A opens with a sparse kick...",
        })
        result = parse_voice_response(raw, "/test.wav", "deep")
        assert result.section_notes is not None
        assert "Section A" in result.section_notes

    def test_missing_optional_fields_default(self):
        raw = json.dumps({
            "description": "test",
            "opinion": "ok",
        })
        result = parse_voice_response(raw, "/test.wav", "quick")
        assert result.tags == []
        assert result.comparisons == []
        assert result.section_notes is None

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_voice_response("not json at all", "/test.wav", "quick")


# --- Interpret Orchestrator Tests ---


class TestInterpret:
    def test_calls_provider_with_prompt(self, sample_layer1):
        mock = MockProvider()
        result = interpret(sample_layer1, provider=mock)
        assert len(mock.calls) == 1
        system, prompt = mock.calls[0]
        assert "128.0 BPM" in prompt
        assert isinstance(result, VoiceResult)

    def test_quick_mode(self, sample_layer1, sample_layer2, sample_layer3):
        mock = MockProvider()
        result = interpret(sample_layer1, sample_layer2, sample_layer3, mode="quick", provider=mock)
        assert result.mode == "quick"
        assert result.file_path == "/music/test-track.wav"

    def test_deep_mode(self, sample_layer1):
        deep_response = json.dumps({
            "description": "A deep track.",
            "opinion": "Impressive.",
            "tags": ["deep"],
            "comparisons": [],
            "highlights": ["depth"],
            "concerns": [],
            "section_notes": "Section by section analysis here.",
        })
        mock = MockProvider(response=deep_response)
        result = interpret(sample_layer1, mode="deep", provider=mock)
        assert result.mode == "deep"
        assert result.section_notes is not None

    def test_result_fields(self, sample_layer1, sample_layer2, sample_layer3):
        mock = MockProvider()
        result = interpret(sample_layer1, sample_layer2, sample_layer3, provider=mock)
        assert "techno" in result.tags
        assert len(result.comparisons) > 0
        assert len(result.highlights) > 0
        assert result.description != ""
        assert result.opinion != ""

    def test_all_layers_in_prompt(self, sample_layer1, sample_layer2, sample_layer3):
        mock = MockProvider()
        interpret(sample_layer1, sample_layer2, sample_layer3, provider=mock)
        _, prompt = mock.calls[0]
        assert "ABABCA" in prompt  # Layer 2 structure
        assert "clipping" in prompt.lower()  # Layer 3 quality


# --- Provider Tests ---


class TestGetProvider:
    def test_ollama_default(self, monkeypatch):
        monkeypatch.delenv("EARWORM_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        provider = get_provider("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_anthropic_requires_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            get_provider("anthropic")

    def test_anthropic_with_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
        provider = get_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider("gpt4")

    def test_env_var_selection(self, monkeypatch):
        monkeypatch.setenv("EARWORM_LLM_PROVIDER", "ollama")
        provider = get_provider()
        assert isinstance(provider, OllamaProvider)

    def test_model_override(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        provider = get_provider("anthropic", model="claude-haiku-4-5-20251001")
        assert provider.model == "claude-haiku-4-5-20251001"


# --- VoiceResult Model Tests ---


class TestVoiceResult:
    def test_serialization(self):
        result = VoiceResult(
            file_path="/test.wav",
            mode="quick",
            description="A test track.",
            opinion="It's OK.",
            tags=["test"],
            comparisons=["nothing"],
            highlights=["exists"],
            concerns=[],
        )
        data = result.model_dump()
        assert data["file_path"] == "/test.wav"
        assert data["section_notes"] is None

    def test_json_roundtrip(self):
        result = VoiceResult(
            file_path="/test.wav",
            mode="deep",
            description="Deep.",
            opinion="Very.",
            tags=["deep"],
            comparisons=[],
            highlights=[],
            concerns=[],
            section_notes="Section A is good.",
        )
        json_str = result.model_dump_json()
        restored = VoiceResult.model_validate_json(json_str)
        assert restored.section_notes == "Section A is good."
