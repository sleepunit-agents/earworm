"""Samplebank bridge — connect earworm perception to samplebank's sample library.

Uses samplebank's semantic search API (text-to-audio via CLAP) and optionally
queries Qdrant directly for audio-to-audio search.
"""

from __future__ import annotations

import logging
import os
from itertools import islice

import httpx

from earworm.bridge.models import BridgeStatus, SampleMatch
from earworm.models import VoiceResult

logger = logging.getLogger(__name__)

DEFAULT_SAMPLEBANK_URL = "http://localhost:8000"
DEFAULT_CLAP_URL = "http://localhost:8100"
DEFAULT_QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "samplebank_laion_clap"


class SamplebankBridge:
    """Connect earworm perception to samplebank's sample library."""

    def __init__(
        self,
        samplebank_url: str | None = None,
        clap_url: str | None = None,
        qdrant_url: str | None = None,
        timeout: float = 30.0,
    ):
        self._samplebank_url = (
            samplebank_url
            or os.environ.get("EARWORM_SAMPLEBANK_URL", DEFAULT_SAMPLEBANK_URL)
        )
        self._clap_url = (
            clap_url or os.environ.get("EARWORM_CLAP_URL", DEFAULT_CLAP_URL)
        )
        self._qdrant_url = (
            qdrant_url or os.environ.get("EARWORM_QDRANT_URL", DEFAULT_QDRANT_URL)
        )
        self._timeout = timeout

    def search_by_voice(
        self,
        voice: VoiceResult,
        limit: int = 20,
    ) -> list[SampleMatch]:
        """Text-to-audio: find samples matching Voice interpretation.

        Runs multiple queries (description, tags, highlights) weighted by
        relevance, then merges results by highest score per sample.
        """
        queries = self._build_queries(voice)
        merged: dict[int, SampleMatch] = {}

        for query_text, weight in queries:
            results = self._search_semantic(query_text, limit=limit)
            for r in results:
                weighted_score = r.score * weight
                existing = merged.get(r.sample_id)
                if existing is None or weighted_score > existing.score:
                    merged[r.sample_id] = SampleMatch(
                        sample_id=r.sample_id,
                        filename=r.filename,
                        path=r.path,
                        score=weighted_score,
                        match_source="text",
                    )

        ranked = sorted(merged.values(), key=lambda m: m.score, reverse=True)
        return list(islice(ranked, limit))

    def search_by_audio(
        self,
        audio_path: str,
        limit: int = 20,
    ) -> list[SampleMatch]:
        """Audio-to-audio: find samples that sound like this track.

        Embeds audio via the CLAP service, then searches samplebank's
        Qdrant collection directly.
        """
        embedding = self._embed_audio(audio_path)
        if embedding is None:
            return []
        return self._search_qdrant(embedding, limit=limit)

    def search_combined(
        self,
        voice: VoiceResult,
        audio_path: str,
        limit: int = 20,
        text_weight: float = 0.6,
        audio_weight: float = 0.4,
    ) -> list[SampleMatch]:
        """Both search paths combined with weighted merge."""
        text_results = self.search_by_voice(voice, limit=limit)
        audio_results = self.search_by_audio(audio_path, limit=limit)

        merged: dict[int, SampleMatch] = {}

        for r in text_results:
            merged[r.sample_id] = SampleMatch(
                sample_id=r.sample_id,
                filename=r.filename,
                path=r.path,
                score=r.score * text_weight,
                match_source="text",
            )

        for r in audio_results:
            existing = merged.get(r.sample_id)
            audio_score = r.score * audio_weight
            if existing is None:
                merged[r.sample_id] = SampleMatch(
                    sample_id=r.sample_id,
                    filename=r.filename,
                    path=r.path,
                    score=audio_score,
                    match_source="audio",
                )
            else:
                combined_score = existing.score + audio_score
                merged[r.sample_id] = SampleMatch(
                    sample_id=r.sample_id,
                    filename=r.filename,
                    path=r.path,
                    score=combined_score,
                    match_source="combined",
                )

        ranked = sorted(merged.values(), key=lambda m: m.score, reverse=True)
        return list(islice(ranked, limit))

    def check_status(self) -> BridgeStatus:
        """Check samplebank health and embedding coverage."""
        try:
            resp = httpx.get(
                f"{self._samplebank_url}/api/samples/semantic/status",
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                return BridgeStatus(
                    samplebank_reachable=False,
                    indexed_samples=0,
                    total_samples=0,
                    coverage_pct=0.0,
                )
            data = resp.json()
            total = data.get("total_samples", 0)
            indexed = data.get("indexed_samples", data.get("embedded", 0))
            return BridgeStatus(
                samplebank_reachable=True,
                indexed_samples=indexed,
                total_samples=total,
                coverage_pct=(indexed / total * 100) if total > 0 else 0.0,
            )
        except (httpx.ConnectError, httpx.TimeoutException):
            return BridgeStatus(
                samplebank_reachable=False,
                indexed_samples=0,
                total_samples=0,
                coverage_pct=0.0,
            )

    # ── Internal: query construction ─────────────────────────────────────

    def _build_queries(self, voice: VoiceResult) -> list[tuple[str, float]]:
        """Build weighted search queries from VoiceResult fields.

        Returns list of (query_text, weight) tuples.
        """
        queries: list[tuple[str, float]] = []

        if voice.description:
            queries.append((voice.description, 1.0))

        if voice.tags:
            for group in self._group_tags(voice.tags):
                queries.append((group, 0.6))

        if voice.highlights:
            for highlight in voice.highlights:
                queries.append((highlight, 0.4))

        return queries

    @staticmethod
    def _group_tags(tags: list[str], max_per_group: int = 3) -> list[str]:
        """Group tags into semantic clusters for better CLAP matching.

        Simple strategy: chunk tags into groups of up to max_per_group,
        joined with spaces. Keeps order from the Voice output, which
        tends to put genre/energy tags first.
        """
        groups = []
        for i in range(0, len(tags), max_per_group):
            chunk = tags[i : i + max_per_group]
            groups.append(" ".join(chunk))
        return groups

    # ── Internal: samplebank API ─────────────────────────────────────────

    def _search_semantic(self, query: str, limit: int = 20) -> list[SampleMatch]:
        """Hit samplebank's semantic search endpoint."""
        try:
            resp = httpx.get(
                f"{self._samplebank_url}/api/samples/semantic",
                params={"q": query, "limit": limit},
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                logger.warning("Samplebank semantic search returned %d", resp.status_code)
                return []
            data = resp.json()
            results = data.get("results", [])
            return [
                SampleMatch(
                    sample_id=r["id"],
                    filename=r.get("filename", ""),
                    path=r.get("path", ""),
                    score=r.get("semantic_score", r.get("score", 0.0)),
                    match_source="text",
                )
                for r in results
            ]
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.warning("Samplebank unreachable: %s", e)
            return []
        except Exception as e:
            logger.error("Semantic search error: %s", e)
            return []

    # ── Internal: CLAP + Qdrant (audio-to-audio) ────────────────────────

    def _embed_audio(self, audio_path: str) -> list[float] | None:
        """Send audio file to the CLAP service for embedding."""
        try:
            with open(audio_path, "rb") as f:
                resp = httpx.post(
                    f"{self._clap_url}/embed/audio",
                    files={"file": (audio_path.rsplit("/", 1)[-1], f)},
                    timeout=self._timeout,
                )
            if resp.status_code != 200:
                logger.warning("CLAP embed returned %d", resp.status_code)
                return None
            return resp.json().get("embedding")
        except (httpx.ConnectError, httpx.TimeoutException):
            logger.debug("CLAP service not reachable")
            return None
        except Exception as e:
            logger.warning("CLAP embedding error: %s", e)
            return None

    def _search_qdrant(
        self, embedding: list[float], limit: int = 20
    ) -> list[SampleMatch]:
        """Search samplebank's Qdrant collection directly with a vector."""
        try:
            resp = httpx.post(
                f"{self._qdrant_url}/collections/{QDRANT_COLLECTION}/points/search",
                json={"vector": embedding, "limit": limit, "with_payload": True},
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                logger.warning("Qdrant search returned %d", resp.status_code)
                return []
            hits = resp.json().get("result", [])
            return [
                SampleMatch(
                    sample_id=h["payload"].get("sample_id", h.get("id", 0)),
                    filename=h["payload"].get("filename", ""),
                    path=h["payload"].get("path", ""),
                    score=h.get("score", 0.0),
                    match_source="audio",
                )
                for h in hits
                if h.get("payload")
            ]
        except (httpx.ConnectError, httpx.TimeoutException):
            logger.debug("Qdrant not reachable")
            return []
        except Exception as e:
            logger.warning("Qdrant search error: %s", e)
            return []
