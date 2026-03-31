"""Tests for the taste journal and divergence tracking."""

from __future__ import annotations

import json

import pytest

from earworm.calibration.journal import (
    DivergenceType,
    JournalManager,
    JournalObservation,
    TasteJournal,
    TastePattern,
)


class TestJournalManager:
    def test_new_journal_is_empty(self, tmp_path):
        mgr = JournalManager(journal_path=tmp_path / "journal.json")
        assert len(mgr.journal.observations) == 0
        assert len(mgr.journal.patterns) == 0
        assert len(mgr.journal.divergences) == 0

    def test_add_observation(self, tmp_path):
        mgr = JournalManager(journal_path=tmp_path / "journal.json")
        obs = mgr.add_observation(
            track_id="test-track",
            what_i_noticed="Dense polyrhythmic layers",
            what_stood_out="The bass line anchors everything",
            what_i_missed="Vocal processing details",
            raw_reaction="Immediately engaging — groove pulls you in",
        )
        assert obs.track_id == "test-track"
        assert "polyrhythmic" in obs.what_i_noticed
        assert len(mgr.journal.observations) == 1

    def test_add_pattern(self, tmp_path):
        mgr = JournalManager(journal_path=tmp_path / "journal.json")
        pattern = mgr.add_pattern(
            name="polyrhythmic-density",
            description="Drawn to tracks with interlocking rhythmic layers",
            supporting_tracks=["talking-heads-born-under-punches"],
            confidence=0.7,
        )
        assert pattern.name == "polyrhythmic-density"
        assert pattern.confidence == 0.7
        assert len(mgr.journal.patterns) == 1

    def test_update_existing_pattern(self, tmp_path):
        mgr = JournalManager(journal_path=tmp_path / "journal.json")
        mgr.add_pattern(
            name="polyrhythmic-density",
            description="Initial description",
            supporting_tracks=["track-a"],
            confidence=0.5,
        )
        updated = mgr.add_pattern(
            name="polyrhythmic-density",
            description="Refined description",
            supporting_tracks=["track-b"],
            confidence=0.8,
        )
        assert len(mgr.journal.patterns) == 1
        assert updated.description == "Refined description"
        assert updated.confidence == 0.8
        assert "track-a" in updated.supporting_tracks
        assert "track-b" in updated.supporting_tracks

    def test_get_pattern(self, tmp_path):
        mgr = JournalManager(journal_path=tmp_path / "journal.json")
        mgr.add_pattern(
            name="test-pattern",
            description="Test",
        )
        assert mgr.get_pattern("test-pattern") is not None
        assert mgr.get_pattern("nonexistent") is None

    def test_classify_divergence(self, tmp_path):
        mgr = JournalManager(journal_path=tmp_path / "journal.json")
        div = mgr.classify_divergence(
            track_id="test-track",
            dimension="texture",
            pipeline_perception="bright, forward",
            human_consensus="warm texture",
            classification="taste",
            reasoning="Pipeline captures spectral brightness, humans respond to timbral warmth",
        )
        assert div.classification == "taste"
        assert len(mgr.journal.divergences) == 1

    def test_classify_divergence_invalid(self, tmp_path):
        mgr = JournalManager(journal_path=tmp_path / "journal.json")
        with pytest.raises(ValueError, match="Invalid classification"):
            mgr.classify_divergence(
                track_id="test",
                dimension="texture",
                pipeline_perception="bright",
                human_consensus="warm",
                classification="invalid",
            )

    def test_gap_and_taste_counts(self, tmp_path):
        mgr = JournalManager(journal_path=tmp_path / "journal.json")
        mgr.classify_divergence("t1", "mood", "dark", "anxious", "gap")
        mgr.classify_divergence("t1", "texture", "bright", "warm", "taste")
        mgr.classify_divergence("t2", "energy", "steady", "building", "gap")
        mgr.classify_divergence("t2", "dynamics", "compressed", "dynamic", "unclear")
        assert mgr.gap_count() == 2
        assert mgr.taste_count() == 1

    def test_get_observations_for_track(self, tmp_path):
        mgr = JournalManager(journal_path=tmp_path / "journal.json")
        mgr.add_observation("track-a", "noticed-a", "stood-out-a")
        mgr.add_observation("track-b", "noticed-b", "stood-out-b")
        mgr.add_observation("track-a", "noticed-a2", "stood-out-a2")
        assert len(mgr.get_observations_for_track("track-a")) == 2
        assert len(mgr.get_observations_for_track("track-b")) == 1

    def test_get_divergences_for_track(self, tmp_path):
        mgr = JournalManager(journal_path=tmp_path / "journal.json")
        mgr.classify_divergence("track-a", "mood", "x", "y", "gap")
        mgr.classify_divergence("track-b", "mood", "x", "y", "taste")
        assert len(mgr.get_divergences_for_track("track-a")) == 1
        assert len(mgr.get_divergences_for_track("track-b")) == 1

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "journal.json"
        mgr = JournalManager(journal_path=path)
        mgr.add_observation("track-a", "noticed", "stood-out")
        mgr.add_pattern("pattern-a", "description")
        mgr.classify_divergence("track-a", "mood", "x", "y", "gap")
        mgr.save()

        mgr2 = JournalManager(journal_path=path)
        assert len(mgr2.journal.observations) == 1
        assert len(mgr2.journal.patterns) == 1
        assert len(mgr2.journal.divergences) == 1

    def test_summary(self, tmp_path):
        mgr = JournalManager(journal_path=tmp_path / "journal.json")
        mgr.add_observation("track-a", "noticed-a", "stood-out-a")
        mgr.add_observation("track-b", "noticed-b", "stood-out-b")
        mgr.add_pattern("pattern-a", "desc-a", confidence=0.8)
        mgr.add_pattern("pattern-b", "desc-b", confidence=0.3)
        mgr.classify_divergence("track-a", "mood", "x", "y", "gap")
        mgr.classify_divergence("track-b", "texture", "x", "y", "taste")

        summary = mgr.summary()
        assert summary["total_observations"] == 2
        assert summary["tracks_observed"] == 2
        assert summary["patterns_identified"] == 2
        assert summary["total_divergences"] == 2
        assert summary["gaps"] == 1
        assert summary["taste_differences"] == 1
        assert summary["strongest_patterns"][0] == "pattern-a"

    def test_summary_empty(self, tmp_path):
        mgr = JournalManager(journal_path=tmp_path / "journal.json")
        summary = mgr.summary()
        assert summary["total_observations"] == 0
        assert summary["strongest_patterns"] == []


class TestJournalModels:
    def test_observation_defaults(self):
        obs = JournalObservation(
            track_id="test",
            what_i_noticed="Something",
            what_stood_out="Something else",
        )
        assert obs.what_i_missed == ""
        assert obs.raw_reaction == ""
        assert obs.timestamp is not None

    def test_pattern_defaults(self):
        p = TastePattern(name="test", description="test desc")
        assert p.supporting_tracks == []
        assert p.confidence == 0.5

    def test_pattern_confidence_bounds(self):
        with pytest.raises(Exception):
            TastePattern(name="test", description="test", confidence=1.5)
        with pytest.raises(Exception):
            TastePattern(name="test", description="test", confidence=-0.1)

    def test_divergence_fields(self):
        d = DivergenceType(
            track_id="test",
            dimension="mood",
            pipeline_perception="dark",
            human_consensus="anxious",
            classification="gap",
        )
        assert d.reasoning == ""
        assert d.timestamp is not None

    def test_journal_round_trip(self):
        j = TasteJournal()
        j.observations.append(JournalObservation(
            track_id="test", what_i_noticed="x", what_stood_out="y"
        ))
        data = json.loads(j.model_dump_json())
        j2 = TasteJournal.model_validate(data)
        assert len(j2.observations) == 1
        assert j2.observations[0].track_id == "test"


class TestJournalWithCalibration:
    """Integration tests — journal working with calibration corpus data."""

    def test_observations_from_alignment(self, tmp_path):
        """Simulate the workflow: run alignment, then record observations."""
        mgr = JournalManager(journal_path=tmp_path / "journal.json")

        mgr.add_observation(
            track_id="talking-heads-born-under-punches",
            what_i_noticed=(
                "Pipeline detected 130 BPM, complex structure with 3 section types, "
                "and building energy with multiple builds. Harmonic ratio suggests "
                "percussion-dominant, which aligns with the polyrhythmic description."
            ),
            what_stood_out=(
                "The building energy detection matches the human description of "
                "'tension builds throughout without conventional release.' The pipeline "
                "sees the same arc."
            ),
            what_i_missed=(
                "Human descriptions emphasize the 'organized chaos' quality — the feeling "
                "of interlocking parts. The pipeline sees builds and sections but not "
                "the interlock between parts."
            ),
            raw_reaction="The numbers paint a picture consistent with what I'd expect from the descriptions.",
        )

        mgr.classify_divergence(
            track_id="talking-heads-born-under-punches",
            dimension="rhythm",
            pipeline_perception="percussion-dominant",
            human_consensus="polyrhythmic, syncopated",
            classification="gap",
            reasoning=(
                "Pipeline captures harmonic/percussive ratio but not polyrhythmic complexity. "
                "The distinction between 'percussion-dominant' and 'polyrhythmic' is real — "
                "one is energy balance, the other is structural interlocking."
            ),
        )

        mgr.add_pattern(
            name="structural-interlocking",
            description=(
                "Drawn to music where multiple independent rhythmic/melodic lines "
                "interlock — not just layered, but interdependent. The pipeline sees "
                "layers (energy arc, section types) but not interlocking."
            ),
            supporting_tracks=["talking-heads-born-under-punches"],
            confidence=0.4,
        )

        mgr.save()

        mgr2 = JournalManager(journal_path=tmp_path / "journal.json")
        assert len(mgr2.journal.observations) == 1
        assert len(mgr2.journal.divergences) == 1
        assert len(mgr2.journal.patterns) == 1
        assert mgr2.gap_count() == 1

    def test_multi_track_pattern_building(self, tmp_path):
        """Simulate building a pattern across multiple tracks."""
        mgr = JournalManager(journal_path=tmp_path / "journal.json")

        mgr.add_pattern(
            name="glacial-build",
            description="Tracks that build slowly over long runtimes reward patience",
            supporting_tracks=["deadmau5-xpander"],
            confidence=0.4,
        )

        mgr.add_observation(
            track_id="another-track",
            what_i_noticed="9-minute runtime with single build arc",
            what_stood_out="Similar patience-rewarding quality to Xpander",
        )

        mgr.add_pattern(
            name="glacial-build",
            description="Extended tracks with slow, deliberate builds over 6+ minutes",
            supporting_tracks=["another-track"],
            confidence=0.7,
        )

        pattern = mgr.get_pattern("glacial-build")
        assert pattern is not None
        assert len(pattern.supporting_tracks) == 2
        assert pattern.confidence == 0.7
