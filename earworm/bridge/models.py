"""Data models for samplebank bridge results."""

from __future__ import annotations

from pydantic import BaseModel


class SampleMatch(BaseModel):
    """A sample from samplebank that matches an earworm query."""

    sample_id: int
    filename: str
    path: str
    score: float
    match_source: str  # "text", "audio", or "combined"


class BridgeStatus(BaseModel):
    """Health/readiness status of the samplebank bridge."""

    samplebank_reachable: bool
    indexed_samples: int
    total_samples: int
    coverage_pct: float
