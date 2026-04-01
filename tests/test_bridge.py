"""Tests for the samplebank bridge — text-to-audio and audio-to-audio search."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from earworm.bridge.models import SampleMatch
from earworm.bridge.samplebank import SamplebankBridge
from earworm.models import VoiceResult


def _voice(
    description: str = "A driving techno track",
    tags: list[str] | None = None,
    highlights: list[str] | None = None,
    concerns: list[str] | None = None,
    comparisons: list[str] | None = None,
) -> VoiceResult:
    return VoiceResult(
        file_path="/test/track.wav",
        mode="quick",
        description=description,
        opinion="Solid work",
        tags=tags or [],
        comparisons=comparisons or [],
        highlights=highlights or [],
        concerns=concerns or [],
    )


def _samplebank_response(results: list[dict], total: int | None = None) -> dict:
    return {
        "results": results,
        "total": total if total is not None else len(results),
        "query": "test",
    }


def _sample_hit(sid: int, filename: str, score: float) -> dict:
    return {
        "id": sid,
        "filename": filename,
        "path": f"/samples/{filename}",
        "semantic_score": score,
    }


# ── Query construction ───────────────────────────────────────────────────


class TestQueryConstruction:
    def test_description_only(self):
        bridge = SamplebankBridge(samplebank_url="http://fake:8000")
        voice = _voice(description="Dark ambient pad textures")
        queries = bridge._build_queries(voice)
        assert len(queries) == 1
        assert queries[0] == ("Dark ambient pad textures", 1.0)

    def test_description_and_tags(self):
        bridge = SamplebankBridge(samplebank_url="http://fake:8000")
        voice = _voice(
            description="Driving techno",
            tags=["techno", "dark", "minimal"],
        )
        queries = bridge._build_queries(voice)
        assert queries[0] == ("Driving techno", 1.0)
        assert any(q[1] == 0.6 for q in queries)

    def test_description_tags_and_highlights(self):
        bridge = SamplebankBridge(samplebank_url="http://fake:8000")
        voice = _voice(
            description="Driving track",
            tags=["techno"],
            highlights=["Clean low end", "Wide stereo field"],
        )
        queries = bridge._build_queries(voice)
        weights = [w for _, w in queries]
        assert 1.0 in weights
        assert 0.6 in weights
        assert 0.4 in weights

    def test_empty_voice_returns_only_description(self):
        bridge = SamplebankBridge(samplebank_url="http://fake:8000")
        voice = _voice(description="Something")
        queries = bridge._build_queries(voice)
        assert len(queries) == 1

    def test_empty_description_yields_no_description_query(self):
        bridge = SamplebankBridge(samplebank_url="http://fake:8000")
        voice = _voice(description="", tags=["techno"])
        queries = bridge._build_queries(voice)
        assert all(w != 1.0 for _, w in queries)


class TestTagGrouping:
    def test_small_set_single_group(self):
        groups = SamplebankBridge._group_tags(["techno", "dark"])
        assert groups == ["techno dark"]

    def test_groups_of_three(self):
        groups = SamplebankBridge._group_tags(
            ["techno", "dark", "minimal", "driving", "hypnotic"]
        )
        assert groups == ["techno dark minimal", "driving hypnotic"]

    def test_empty_tags(self):
        groups = SamplebankBridge._group_tags([])
        assert groups == []

    def test_single_tag(self):
        groups = SamplebankBridge._group_tags(["ambient"])
        assert groups == ["ambient"]

    def test_custom_group_size(self):
        groups = SamplebankBridge._group_tags(["a", "b", "c", "d"], max_per_group=2)
        assert groups == ["a b", "c d"]


# ── Text-to-audio search ────────────────────────────────────────────────


class TestSearchByVoice:
    def test_basic_search(self):
        bridge = SamplebankBridge(samplebank_url="http://fake:8000")
        voice = _voice(description="Dark techno kick")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _samplebank_response([
            _sample_hit(1, "kick_dark_01.wav", 0.92),
            _sample_hit(2, "kick_techno_03.wav", 0.87),
        ])

        with patch("earworm.bridge.samplebank.httpx.get", return_value=mock_resp):
            results = bridge.search_by_voice(voice, limit=10)

        assert len(results) == 2
        assert results[0].sample_id == 1
        assert results[0].score == pytest.approx(0.92)
        assert results[0].match_source == "text"

    def test_multi_query_dedup(self):
        """Same sample from multiple queries keeps highest score."""
        bridge = SamplebankBridge(samplebank_url="http://fake:8000")
        voice = _voice(
            description="Techno kick",
            tags=["techno", "kick"],
            highlights=["Punchy low end"],
        )

        call_count = [0]

        def mock_get(url, **kwargs):
            call_count[0] += 1
            resp = MagicMock()
            resp.status_code = 200
            if call_count[0] == 1:
                resp.json.return_value = _samplebank_response([
                    _sample_hit(1, "kick.wav", 0.90),
                    _sample_hit(2, "snare.wav", 0.80),
                ])
            elif call_count[0] == 2:
                resp.json.return_value = _samplebank_response([
                    _sample_hit(1, "kick.wav", 0.85),
                    _sample_hit(3, "hat.wav", 0.75),
                ])
            else:
                resp.json.return_value = _samplebank_response([
                    _sample_hit(1, "kick.wav", 0.70),
                ])
            return resp

        with patch("earworm.bridge.samplebank.httpx.get", side_effect=mock_get):
            results = bridge.search_by_voice(voice, limit=10)

        ids = {r.sample_id for r in results}
        assert ids == {1, 2, 3}
        sample_1 = next(r for r in results if r.sample_id == 1)
        assert sample_1.score == pytest.approx(0.90)

    def test_samplebank_unreachable(self):
        bridge = SamplebankBridge(samplebank_url="http://fake:8000")
        voice = _voice(description="Test")

        with patch(
            "earworm.bridge.samplebank.httpx.get",
            side_effect=httpx.ConnectError("refused"),
        ):
            results = bridge.search_by_voice(voice)

        assert results == []

    def test_samplebank_error_status(self):
        bridge = SamplebankBridge(samplebank_url="http://fake:8000")
        voice = _voice(description="Test")

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("earworm.bridge.samplebank.httpx.get", return_value=mock_resp):
            results = bridge.search_by_voice(voice)

        assert results == []

    def test_limit_respected(self):
        bridge = SamplebankBridge(samplebank_url="http://fake:8000")
        voice = _voice(description="Test", tags=["a", "b", "c"])

        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = _samplebank_response([
                _sample_hit(i, f"sample_{i}.wav", 0.9 - i * 0.01)
                for i in range(10)
            ])
            return resp

        with patch("earworm.bridge.samplebank.httpx.get", side_effect=mock_get):
            results = bridge.search_by_voice(voice, limit=3)

        assert len(results) <= 3


# ── Audio-to-audio search ───────────────────────────────────────────────


class TestSearchByAudio:
    def test_basic_audio_search(self, tmp_path):
        bridge = SamplebankBridge(
            clap_url="http://fake-clap:8100",
            qdrant_url="http://fake-qdrant:6333",
        )
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_clap = MagicMock()
        mock_clap.status_code = 200
        mock_clap.json.return_value = {"embedding": [0.1] * 512, "dim": 512}

        mock_qdrant = MagicMock()
        mock_qdrant.status_code = 200
        mock_qdrant.json.return_value = {
            "result": [
                {
                    "id": 42,
                    "score": 0.95,
                    "payload": {"sample_id": 42, "filename": "match.wav", "path": "/samples/match.wav"},
                }
            ]
        }

        def route_request(url, **kwargs):
            if "embed/audio" in url:
                return mock_clap
            if "points/search" in url:
                return mock_qdrant
            raise ValueError(f"Unexpected URL: {url}")

        with patch("earworm.bridge.samplebank.httpx.post", side_effect=route_request):
            with patch("earworm.bridge.samplebank.open", create=True):
                results = bridge.search_by_audio(str(audio_file), limit=5)

        assert len(results) == 1
        assert results[0].sample_id == 42
        assert results[0].match_source == "audio"

    def test_clap_unavailable(self, tmp_path):
        bridge = SamplebankBridge(clap_url="http://fake:8100")
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        with patch(
            "earworm.bridge.samplebank.httpx.post",
            side_effect=httpx.ConnectError("refused"),
        ):
            results = bridge.search_by_audio(str(audio_file))

        assert results == []

    def test_qdrant_unavailable(self, tmp_path):
        bridge = SamplebankBridge(
            clap_url="http://fake:8100",
            qdrant_url="http://fake:6333",
        )
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_clap = MagicMock()
        mock_clap.status_code = 200
        mock_clap.json.return_value = {"embedding": [0.1] * 512, "dim": 512}

        call_count = [0]

        def route(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_clap
            raise httpx.ConnectError("refused")

        with patch("earworm.bridge.samplebank.httpx.post", side_effect=route):
            with patch("earworm.bridge.samplebank.open", create=True):
                results = bridge.search_by_audio(str(audio_file))

        assert results == []


# ── Combined search ──────────────────────────────────────────────────────


class TestSearchCombined:
    def test_combined_merges_sources(self, tmp_path):
        bridge = SamplebankBridge(
            samplebank_url="http://fake:8000",
            clap_url="http://fake:8100",
            qdrant_url="http://fake:6333",
        )
        voice = _voice(description="Dark techno")
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        text_results = [
            SampleMatch(sample_id=1, filename="a.wav", path="/a.wav", score=0.9, match_source="text"),
            SampleMatch(sample_id=2, filename="b.wav", path="/b.wav", score=0.8, match_source="text"),
        ]
        audio_results = [
            SampleMatch(sample_id=2, filename="b.wav", path="/b.wav", score=0.85, match_source="audio"),
            SampleMatch(sample_id=3, filename="c.wav", path="/c.wav", score=0.7, match_source="audio"),
        ]

        with (
            patch.object(bridge, "search_by_voice", return_value=text_results),
            patch.object(bridge, "search_by_audio", return_value=audio_results),
        ):
            results = bridge.search_combined(
                voice, str(audio_file), text_weight=0.6, audio_weight=0.4
            )

        ids = {r.sample_id for r in results}
        assert ids == {1, 2, 3}

        sample_2 = next(r for r in results if r.sample_id == 2)
        assert sample_2.match_source == "combined"
        assert sample_2.score == pytest.approx(0.8 * 0.6 + 0.85 * 0.4)

    def test_combined_text_only_fallback(self, tmp_path):
        """When audio search fails, text results still come through."""
        bridge = SamplebankBridge(samplebank_url="http://fake:8000")
        voice = _voice(description="Test")
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        text_results = [
            SampleMatch(sample_id=1, filename="a.wav", path="/a.wav", score=0.9, match_source="text"),
        ]

        with (
            patch.object(bridge, "search_by_voice", return_value=text_results),
            patch.object(bridge, "search_by_audio", return_value=[]),
        ):
            results = bridge.search_combined(voice, str(audio_file))

        assert len(results) == 1
        assert results[0].sample_id == 1


# ── Status check ─────────────────────────────────────────────────────────


class TestCheckStatus:
    def test_healthy_status(self):
        bridge = SamplebankBridge(samplebank_url="http://fake:8000")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "total_samples": 50000,
            "indexed_samples": 45000,
        }

        with patch("earworm.bridge.samplebank.httpx.get", return_value=mock_resp):
            status = bridge.check_status()

        assert status.samplebank_reachable is True
        assert status.indexed_samples == 45000
        assert status.total_samples == 50000
        assert status.coverage_pct == pytest.approx(90.0)

    def test_unreachable_status(self):
        bridge = SamplebankBridge(samplebank_url="http://fake:8000")

        with patch(
            "earworm.bridge.samplebank.httpx.get",
            side_effect=httpx.ConnectError("refused"),
        ):
            status = bridge.check_status()

        assert status.samplebank_reachable is False
        assert status.indexed_samples == 0

    def test_zero_samples(self):
        bridge = SamplebankBridge(samplebank_url="http://fake:8000")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"total_samples": 0, "indexed_samples": 0}

        with patch("earworm.bridge.samplebank.httpx.get", return_value=mock_resp):
            status = bridge.check_status()

        assert status.coverage_pct == 0.0


# ── Config from env ──────────────────────────────────────────────────────


class TestConfig:
    def test_defaults(self):
        bridge = SamplebankBridge()
        assert "8000" in bridge._samplebank_url
        assert "8100" in bridge._clap_url
        assert "6333" in bridge._qdrant_url

    def test_explicit_urls(self):
        bridge = SamplebankBridge(
            samplebank_url="http://custom:9000",
            clap_url="http://custom:9100",
            qdrant_url="http://custom:7333",
        )
        assert bridge._samplebank_url == "http://custom:9000"
        assert bridge._clap_url == "http://custom:9100"
        assert bridge._qdrant_url == "http://custom:7333"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("EARWORM_SAMPLEBANK_URL", "http://env:8000")
        monkeypatch.setenv("EARWORM_CLAP_URL", "http://env:8100")
        monkeypatch.setenv("EARWORM_QDRANT_URL", "http://env:6333")
        bridge = SamplebankBridge()
        assert bridge._samplebank_url == "http://env:8000"
        assert bridge._clap_url == "http://env:8100"
        assert bridge._qdrant_url == "http://env:6333"
