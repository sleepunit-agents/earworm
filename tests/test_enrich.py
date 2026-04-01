"""Tests for bridge voice enrichment."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from earworm.bridge.enrich import enrich_voice_with_samples
from earworm.bridge.models import SampleMatch
from earworm.models import SampleReference, VoiceResult


def _make_voice(**kwargs) -> VoiceResult:
    defaults = {
        "file_path": "/tmp/test.wav",
        "mode": "quick",
        "description": "A driving techno track with layered percussion",
        "opinion": "Solid production, effective energy arc",
        "tags": ["techno", "driving", "dark"],
        "comparisons": ["early Surgeon"],
        "highlights": ["Clean low end"],
        "concerns": [],
    }
    defaults.update(kwargs)
    return VoiceResult(**defaults)


def _make_matches(n: int = 3) -> list[SampleMatch]:
    return [
        SampleMatch(
            sample_id=i,
            filename=f"sample_{i}.wav",
            path=f"/samples/sample_{i}.wav",
            score=0.9 - i * 0.1,
            match_source="text",
        )
        for i in range(n)
    ]


class TestEnrichVoiceWithSamples:
    def test_text_mode_attaches_samples(self):
        voice = _make_voice()
        bridge = MagicMock()
        bridge.search_by_voice.return_value = _make_matches(3)

        result = enrich_voice_with_samples(voice, bridge=bridge, mode="text")

        assert result.related_samples is not None
        assert len(result.related_samples) == 3
        assert result.related_samples[0].filename == "sample_0.wav"
        assert result.related_samples[0].score == pytest.approx(0.9)
        bridge.search_by_voice.assert_called_once()

    def test_audio_mode_requires_path(self):
        voice = _make_voice()
        bridge = MagicMock()

        result = enrich_voice_with_samples(voice, bridge=bridge, mode="audio")

        assert result.related_samples is None
        bridge.search_by_audio.assert_not_called()

    def test_audio_mode_with_path(self):
        voice = _make_voice()
        bridge = MagicMock()
        bridge.search_by_audio.return_value = _make_matches(2)

        result = enrich_voice_with_samples(
            voice, bridge=bridge, audio_path="/tmp/test.wav", mode="audio"
        )

        assert result.related_samples is not None
        assert len(result.related_samples) == 2
        bridge.search_by_audio.assert_called_once_with("/tmp/test.wav", limit=10)

    def test_combined_mode_with_path(self):
        voice = _make_voice()
        bridge = MagicMock()
        bridge.search_combined.return_value = _make_matches(5)

        result = enrich_voice_with_samples(
            voice, bridge=bridge, audio_path="/tmp/test.wav", mode="combined"
        )

        assert result.related_samples is not None
        assert len(result.related_samples) == 5
        bridge.search_combined.assert_called_once()

    def test_combined_mode_without_path_falls_back_to_text(self):
        voice = _make_voice()
        bridge = MagicMock()
        bridge.search_by_voice.return_value = _make_matches(2)

        result = enrich_voice_with_samples(voice, bridge=bridge, mode="combined")

        assert result.related_samples is not None
        assert len(result.related_samples) == 2
        bridge.search_by_voice.assert_called_once()

    def test_creates_default_bridge_if_none(self):
        voice = _make_voice()

        with patch("earworm.bridge.enrich.SamplebankBridge") as MockBridge:
            instance = MockBridge.return_value
            instance.search_by_voice.return_value = _make_matches(1)

            result = enrich_voice_with_samples(voice, mode="text")

            MockBridge.assert_called_once()
            assert result.related_samples is not None

    def test_original_voice_fields_preserved(self):
        voice = _make_voice()
        bridge = MagicMock()
        bridge.search_by_voice.return_value = _make_matches(1)

        result = enrich_voice_with_samples(voice, bridge=bridge, mode="text")

        assert result.description == voice.description
        assert result.opinion == voice.opinion
        assert result.tags == voice.tags
        assert result.highlights == voice.highlights
        assert result.mode == voice.mode

    def test_exception_returns_original_voice(self):
        voice = _make_voice()
        bridge = MagicMock()
        bridge.search_by_voice.side_effect = ConnectionError("unreachable")

        result = enrich_voice_with_samples(voice, bridge=bridge, mode="text")

        assert result.related_samples is None

    def test_custom_limit(self):
        voice = _make_voice()
        bridge = MagicMock()
        bridge.search_by_voice.return_value = _make_matches(5)

        enrich_voice_with_samples(voice, bridge=bridge, limit=5, mode="text")

        bridge.search_by_voice.assert_called_once_with(voice, limit=5)

    def test_unknown_mode_returns_original(self):
        voice = _make_voice()
        bridge = MagicMock()

        result = enrich_voice_with_samples(voice, bridge=bridge, mode="unknown")

        assert result.related_samples is None


class TestSampleReferenceModel:
    def test_serialization(self):
        ref = SampleReference(
            sample_id=42,
            filename="kick_808.wav",
            path="/samples/drums/kick_808.wav",
            score=0.95,
            match_source="text",
        )
        data = ref.model_dump()
        assert data["sample_id"] == 42
        assert data["filename"] == "kick_808.wav"
        assert data["score"] == pytest.approx(0.95)

    def test_in_voice_result(self):
        voice = _make_voice(
            related_samples=[
                SampleReference(
                    sample_id=1,
                    filename="a.wav",
                    path="/a.wav",
                    score=0.8,
                    match_source="text",
                )
            ]
        )
        assert voice.related_samples is not None
        assert len(voice.related_samples) == 1
        assert voice.related_samples[0].filename == "a.wav"

    def test_voice_result_json_roundtrip_with_samples(self):
        voice = _make_voice(
            related_samples=[
                SampleReference(
                    sample_id=1,
                    filename="a.wav",
                    path="/a.wav",
                    score=0.8,
                    match_source="text",
                )
            ]
        )
        json_str = voice.model_dump_json()
        restored = VoiceResult.model_validate_json(json_str)
        assert restored.related_samples is not None
        assert len(restored.related_samples) == 1
        assert restored.related_samples[0].sample_id == 1

    def test_voice_result_without_samples_is_none(self):
        voice = _make_voice()
        assert voice.related_samples is None
