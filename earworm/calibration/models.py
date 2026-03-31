"""Data models for the calibration system."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from earworm.models import Layer1Features, Layer2Features, Layer3Features, VoiceResult


class AlignmentDimension(str, Enum):
    """Dimensions along which pipeline perception is compared to human descriptions."""

    TEMPO = "tempo"
    KEY = "key"
    ENERGY = "energy"
    STRUCTURE = "structure"
    MOOD = "mood"
    PRODUCTION = "production"
    GENRE = "genre"
    DYNAMICS = "dynamics"
    TEXTURE = "texture"
    RHYTHM = "rhythm"


class HumanDescription(BaseModel):
    """A human description of a track, used as calibration reference."""

    source: str  # Where this came from (e.g. "pitchfork", "rym", "allmusic", "art")
    text: str  # The actual description
    tags: list[str] = Field(default_factory=list)
    key_observations: list[str] = Field(default_factory=list)


class AlignmentResult(BaseModel):
    """Result of comparing pipeline perception against a human description."""

    dimension: AlignmentDimension
    pipeline_says: str
    human_says: str
    aligned: bool
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class CalibrationEntry(BaseModel):
    """A single track in the calibration corpus with all associated data."""

    track_id: str  # Unique identifier (artist-title slug)
    artist: str
    title: str
    album: str = ""
    year: int = 0

    audio_path: str | None = None
    human_descriptions: list[HumanDescription] = Field(default_factory=list)

    # Pipeline results (populated after running)
    layer1: Layer1Features | None = None
    layer2: Layer2Features | None = None
    layer3: Layer3Features | None = None
    voice_result: VoiceResult | None = None

    # Alignment and divergence (populated after checking)
    alignments: list[AlignmentResult] = Field(default_factory=list)
    divergences: list[str] = Field(default_factory=list)

    analyzed_at: datetime | None = None
    checked_at: datetime | None = None


class CorpusManifest(BaseModel):
    """Top-level manifest of the calibration corpus."""

    version: int = 1
    entries: list[CalibrationEntry] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class CalibrationReport(BaseModel):
    """Summary report across the calibration corpus."""

    total_tracks: int
    analyzed_tracks: int
    checked_tracks: int
    alignment_rate: float = Field(ge=0.0, le=1.0)
    strongest_dimensions: list[str] = Field(default_factory=list)
    weakest_dimensions: list[str] = Field(default_factory=list)
    notable_divergences: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)
