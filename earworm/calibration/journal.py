"""Taste journal — tracking Art's emerging aesthetic identity.

The journal records what patterns earworm responds to, what it gravitates
toward, and what divergences emerge between its perception and human
consensus. Over time, these entries form the basis of genuine taste — not
a copy of human preferences, but a distinct perceptual character.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class JournalSampleRef(BaseModel):
    """A reference to a samplebank sample in a journal entry."""

    sample_id: int
    filename: str
    score: float = 0.0
    why: str = ""  # Why this sample is relevant to the observation


class JournalObservation(BaseModel):
    """A single observation about a track — what Art noticed."""

    track_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    what_i_noticed: str
    what_stood_out: str
    what_i_missed: str = ""
    raw_reaction: str = ""
    sample_references: list[JournalSampleRef] = Field(default_factory=list)


class TastePattern(BaseModel):
    """A recurring pattern Art gravitates toward across tracks."""

    name: str
    description: str
    supporting_tracks: list[str] = Field(default_factory=list)
    first_noticed: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class DivergenceType(BaseModel):
    """A categorized divergence between Art's perception and human consensus."""

    track_id: str
    dimension: str
    pipeline_perception: str
    human_consensus: str
    classification: str  # "gap" = pipeline limitation, "taste" = genuine difference, "unclear"
    reasoning: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class TasteJournal(BaseModel):
    """The full taste journal — Art's developing aesthetic identity."""

    version: int = 1
    observations: list[JournalObservation] = Field(default_factory=list)
    patterns: list[TastePattern] = Field(default_factory=list)
    divergences: list[DivergenceType] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class JournalManager:
    """Manages the taste journal on disk."""

    def __init__(self, journal_path: Path | str | None = None):
        if journal_path:
            self.journal_path = Path(journal_path)
        else:
            self.journal_path = Path(__file__).parent.parent.parent / "calibration" / "journal.json"
        self._journal: TasteJournal | None = None

    @property
    def journal(self) -> TasteJournal:
        if self._journal is None:
            self._journal = self._load()
        return self._journal

    def _load(self) -> TasteJournal:
        if self.journal_path.exists():
            data = json.loads(self.journal_path.read_text())
            return TasteJournal.model_validate(data)
        return TasteJournal()

    def save(self) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.journal.updated_at = datetime.now()
        self.journal_path.write_text(
            self.journal.model_dump_json(indent=2, exclude_none=True)
        )

    def add_observation(
        self,
        track_id: str,
        what_i_noticed: str,
        what_stood_out: str,
        what_i_missed: str = "",
        raw_reaction: str = "",
        sample_references: list[JournalSampleRef] | None = None,
    ) -> JournalObservation:
        """Record an observation about a track.

        Args:
            sample_references: Optional list of samplebank samples that relate
                to this observation (e.g. "this kick sounds like sample X").
        """
        obs = JournalObservation(
            track_id=track_id,
            what_i_noticed=what_i_noticed,
            what_stood_out=what_stood_out,
            what_i_missed=what_i_missed,
            raw_reaction=raw_reaction,
            sample_references=sample_references or [],
        )
        self.journal.observations.append(obs)
        return obs

    def add_pattern(
        self,
        name: str,
        description: str,
        supporting_tracks: list[str] | None = None,
        confidence: float = 0.5,
    ) -> TastePattern:
        """Record a recurring taste pattern.

        If a pattern with this name already exists, updates it instead.
        """
        existing = self.get_pattern(name)
        if existing:
            existing.description = description
            if supporting_tracks:
                for t in supporting_tracks:
                    if t not in existing.supporting_tracks:
                        existing.supporting_tracks.append(t)
            existing.confidence = confidence
            return existing

        pattern = TastePattern(
            name=name,
            description=description,
            supporting_tracks=supporting_tracks or [],
            confidence=confidence,
        )
        self.journal.patterns.append(pattern)
        return pattern

    def get_pattern(self, name: str) -> TastePattern | None:
        """Look up a pattern by name."""
        for p in self.journal.patterns:
            if p.name == name:
                return p
        return None

    def classify_divergence(
        self,
        track_id: str,
        dimension: str,
        pipeline_perception: str,
        human_consensus: str,
        classification: str,
        reasoning: str = "",
    ) -> DivergenceType:
        """Record a classified divergence.

        Args:
            classification: "gap" (pipeline limitation), "taste" (genuine difference),
                          or "unclear" (needs more data).
        """
        if classification not in ("gap", "taste", "unclear"):
            raise ValueError(f"Invalid classification: {classification}. Use 'gap', 'taste', or 'unclear'.")

        div = DivergenceType(
            track_id=track_id,
            dimension=dimension,
            pipeline_perception=pipeline_perception,
            human_consensus=human_consensus,
            classification=classification,
            reasoning=reasoning,
        )
        self.journal.divergences.append(div)
        return div

    def get_observations_for_track(self, track_id: str) -> list[JournalObservation]:
        """Get all observations for a specific track."""
        return [o for o in self.journal.observations if o.track_id == track_id]

    def get_divergences_for_track(self, track_id: str) -> list[DivergenceType]:
        """Get all divergences for a specific track."""
        return [d for d in self.journal.divergences if d.track_id == track_id]

    def gap_count(self) -> int:
        """Count pipeline gap divergences."""
        return sum(1 for d in self.journal.divergences if d.classification == "gap")

    def taste_count(self) -> int:
        """Count genuine taste divergences."""
        return sum(1 for d in self.journal.divergences if d.classification == "taste")

    def summary(self) -> dict:
        """Generate a summary of the journal's contents."""
        tracks_observed = set(o.track_id for o in self.journal.observations)
        return {
            "total_observations": len(self.journal.observations),
            "tracks_observed": len(tracks_observed),
            "patterns_identified": len(self.journal.patterns),
            "total_divergences": len(self.journal.divergences),
            "gaps": self.gap_count(),
            "taste_differences": self.taste_count(),
            "unclear": sum(1 for d in self.journal.divergences if d.classification == "unclear"),
            "strongest_patterns": [
                p.name for p in sorted(
                    self.journal.patterns,
                    key=lambda p: p.confidence,
                    reverse=True,
                )[:5]
            ],
        }
