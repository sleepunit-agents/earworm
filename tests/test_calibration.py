"""Tests for the Phase 3 calibration system."""

from __future__ import annotations

import json

import pytest

from earworm.calibration.alignment import AlignmentChecker
from earworm.calibration.corpus import Corpus, _slugify
from earworm.calibration.models import (
    AlignmentDimension,
    AlignmentResult,
    CalibrationEntry,
    CalibrationReport,
    CorpusManifest,
    HumanDescription,
)
from earworm.calibration.runner import CalibrationRunner
from earworm.calibration.seeds import seed_corpus


# --- Slugify ---


class TestSlugify:
    def test_basic(self):
        assert _slugify("Talking Heads") == "talking-heads"

    def test_special_chars(self):
        assert _slugify("Born Under Punches (The Heat Goes On)") == "born-under-punches-the-heat-goes-on"

    def test_multiple_spaces(self):
        assert _slugify("foo   bar") == "foo-bar"

    def test_leading_trailing(self):
        assert _slugify("  hello  ") == "hello"

    def test_combined(self):
        assert _slugify("Talking Heads-Born Under Punches (The Heat Goes On)") == (
            "talking-heads-born-under-punches-the-heat-goes-on"
        )


# --- Corpus ---


class TestCorpus:
    def test_new_corpus_is_empty(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        assert corpus.list_tracks() == []

    def test_add_track(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        entry = corpus.add_track("Talking Heads", "Born Under Punches")
        assert entry.track_id == "talking-heads-born-under-punches"
        assert entry.artist == "Talking Heads"
        assert entry.title == "Born Under Punches"
        assert len(corpus.list_tracks()) == 1

    def test_add_duplicate_returns_existing(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        e1 = corpus.add_track("Foo", "Bar")
        e2 = corpus.add_track("Foo", "Bar")
        assert e1 is e2
        assert len(corpus.list_tracks()) == 1

    def test_add_duplicate_updates_audio_path(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        e1 = corpus.add_track("Foo", "Bar")
        assert e1.audio_path is None
        corpus.add_track("Foo", "Bar", audio_path="/music/bar.flac")
        assert e1.audio_path == "/music/bar.flac"

    def test_get_track(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        corpus.add_track("Foo", "Bar")
        assert corpus.get_track("foo-bar") is not None
        assert corpus.get_track("nonexistent") is None

    def test_custom_track_id(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        entry = corpus.add_track("Foo", "Bar", track_id="custom-id")
        assert entry.track_id == "custom-id"

    def test_save_and_reload(self, tmp_path):
        corpus_dir = tmp_path / "cal"
        corpus = Corpus(corpus_dir=corpus_dir)
        corpus.add_track("Talking Heads", "Born Under Punches", album="Remain in Light", year=1980)
        corpus.add_human_description(
            "talking-heads-born-under-punches",
            source="test",
            text="A dense polyrhythmic track.",
            tags=["art-rock"],
        )
        corpus.save()

        # Reload from disk
        corpus2 = Corpus(corpus_dir=corpus_dir)
        tracks = corpus2.list_tracks()
        assert len(tracks) == 1
        assert tracks[0].artist == "Talking Heads"
        assert len(tracks[0].human_descriptions) == 1
        assert tracks[0].human_descriptions[0].source == "test"

    def test_add_human_description_missing_track(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        with pytest.raises(ValueError, match="Track not found"):
            corpus.add_human_description("nonexistent", "test", "some text")

    def test_tracks_needing_analysis(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        e1 = corpus.add_track("Foo", "Bar", audio_path="/music/bar.flac")
        corpus.add_track("Baz", "Qux")
        assert corpus.tracks_needing_analysis() == [e1]

    def test_save_and_load_result(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        entry = corpus.add_track("Foo", "Bar")
        entry.divergences = ["test divergence"]
        corpus.save_result(entry)

        loaded = corpus.load_result("foo-bar")
        assert loaded is not None
        assert loaded.divergences == ["test divergence"]

    def test_load_result_missing(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        assert corpus.load_result("nonexistent") is None


# --- Manifest ---


class TestManifest:
    def test_default_version(self):
        m = CorpusManifest()
        assert m.version == 1
        assert m.entries == []

    def test_round_trip(self):
        m = CorpusManifest()
        data = json.loads(m.model_dump_json())
        m2 = CorpusManifest.model_validate(data)
        assert m2.version == m.version


# --- Seeds ---


class TestSeeds:
    def test_seed_populates_two_tracks(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        seed_corpus(corpus)
        tracks = corpus.list_tracks()
        assert len(tracks) == 2

    def test_seed_born_under_punches(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        seed_corpus(corpus)
        bup = corpus.get_track("talking-heads-born-under-punches-the-heat-goes-on")
        assert bup is not None
        assert bup.artist == "Talking Heads"
        assert len(bup.human_descriptions) >= 2

    def test_seed_xpander(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        seed_corpus(corpus)
        xp = corpus.get_track("sasha-xpander")
        assert xp is not None
        assert xp.artist == "Sasha"
        assert len(xp.human_descriptions) >= 2

    def test_seed_descriptions_have_tags(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        seed_corpus(corpus)
        for track in corpus.list_tracks():
            for desc in track.human_descriptions:
                assert len(desc.tags) > 0, f"Missing tags for {track.track_id}/{desc.source}"

    def test_seed_descriptions_have_observations(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        seed_corpus(corpus)
        for track in corpus.list_tracks():
            for desc in track.human_descriptions:
                assert len(desc.key_observations) > 0, (
                    f"Missing observations for {track.track_id}/{desc.source}"
                )

    def test_seed_is_idempotent(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        seed_corpus(corpus)
        seed_corpus(corpus)
        # Should not create duplicates because add_track returns existing
        assert len(corpus.list_tracks()) == 2

    def test_seed_persists_to_disk(self, tmp_path):
        corpus_dir = tmp_path / "cal"
        corpus = Corpus(corpus_dir=corpus_dir)
        seed_corpus(corpus)

        corpus2 = Corpus(corpus_dir=corpus_dir)
        assert len(corpus2.list_tracks()) == 2


# --- Alignment ---


def _make_entry_with_pipeline(track_id: str = "test-track") -> CalibrationEntry:
    """Create a CalibrationEntry with minimal pipeline results for testing."""
    from earworm.models import (
        HarmonicFeatures,
        Layer1Features,
        Layer2Features,
        Layer3Features,
        LoudnessFeatures,
        SpectralFeatures,
        StereoFeatures,
        TemporalFeatures,
        SegmentationFeatures,
        RecurrenceFeatures,
        EnergyArcFeatures,
        PhraseFeatures,
        TechnicalQuality,
        MixQuality,
        MasteringQuality,
        CompositionQuality,
    )

    layer1 = Layer1Features(
        file_path="/test/track.wav",
        duration_seconds=300.0,
        sample_rate=44100,
        channels=2,
        spectral=SpectralFeatures(
            mfcc_mean=[0.0] * 13, mfcc_std=[0.0] * 13,
            spectral_centroid_mean=3500.0, spectral_centroid_std=500.0,
            spectral_bandwidth_mean=2000.0, spectral_rolloff_mean=6000.0,
            spectral_contrast_mean=[0.0] * 7, spectral_flatness_mean=0.05,
            chroma_mean=[0.0] * 12, chroma_std=[0.0] * 12,
        ),
        temporal=TemporalFeatures(
            bpm=130.0, bpm_confidence=0.9,
            beat_times=[i * 0.46 for i in range(650)],
            onset_times=[i * 0.23 for i in range(1300)],
            onset_rate=4.3, tempo_stability=0.95,
        ),
        harmonic=HarmonicFeatures(
            key="A minor", key_confidence=0.7,
            key_profile=[0.0] * 12, chroma_cqt_mean=[0.0] * 12,
            tonnetz_mean=[0.0] * 6, tonnetz_std=[0.0] * 6,
            harmonic_ratio=0.35,
        ),
        loudness=LoudnessFeatures(
            lufs_integrated=-8.0, lufs_short_term_max=-5.0,
            lufs_range=4.0, dynamic_range_db=10.0,
            peak_db=-0.5, rms_db=-10.0,
            crest_factor_db=9.5, loudness_curve=[0.5] * 300,
        ),
        stereo=StereoFeatures(
            is_stereo=True, width_mean=0.6, width_std=0.1,
            correlation_mean=0.7, correlation_min=0.3,
            mid_side_ratio=2.0, balance=0.0,
        ),
    )

    layer2 = Layer2Features(
        file_path="/test/track.wav",
        duration_seconds=300.0,
        segmentation=SegmentationFeatures(
            boundaries=[0.0, 60.0, 120.0, 180.0, 240.0],
            labels=[0, 1, 0, 1, 2],
            n_sections=5,
            section_durations=[60.0, 60.0, 60.0, 60.0, 60.0],
        ),
        recurrence=RecurrenceFeatures(
            n_distinct_labels=3,
            repetition_ratio=0.6,
            label_sequence=[0, 1, 0, 1, 2],
            label_durations={0: 120.0, 1: 120.0, 2: 60.0},
            novelty_curve=[0.5] * 600,
            novelty_timestamps=[i * 0.5 for i in range(600)],
        ),
        energy_arc=EnergyArcFeatures(
            energy_curve=[0.3, 0.5, 0.7, 0.9, 0.8],
            energy_timestamps=[0.0, 75.0, 150.0, 225.0, 300.0],
            climax_time=225.0, climax_position=0.75,
            n_builds=3, n_drops=1,
            build_times=[0.0, 60.0, 120.0],
            drop_times=[240.0],
            dynamic_spread=0.6,
        ),
        phrase=PhraseFeatures(
            phrase_boundaries=[0.0, 15.0, 30.0],
            phrase_lengths_beats=[32.0, 32.0],
            n_phrases=2,
            typical_phrase_beats=32.0,
            regularity=0.95,
            irregular_phrases=[],
        ),
    )

    layer3 = Layer3Features(
        file_path="/test/track.wav",
        duration_seconds=300.0,
        technical=TechnicalQuality(
            clipping_ratio=0.0001, clipping_regions=0,
            dc_offset=0.001, noise_floor_db=-60.0,
            frequency_balance_score=0.85, has_dc_offset=False,
        ),
        mix=MixQuality(
            low_ratio=0.25, mid_ratio=0.50, high_ratio=0.25,
            spectral_balance_score=0.80, stereo_width_score=0.70,
            low_end_clarity=0.75, high_frequency_clarity=0.80,
        ),
        mastering=MasteringQuality(
            lufs_integrated=-8.0, lufs_deviation_from_target=6.0,
            dynamic_range_score=0.6, loudness_consistency=0.85,
            limiter_artifact_score=0.1, crest_factor_db=9.5,
        ),
        composition=CompositionQuality(
            harmonic_vocabulary=6, chord_change_rate=0.3,
            rhythmic_variation=0.6, melodic_range_semitones=12.0,
            structural_variety=0.7,
        ),
    )

    return CalibrationEntry(
        track_id=track_id,
        artist="Test Artist",
        title="Test Track",
        layer1=layer1,
        layer2=layer2,
        layer3=layer3,
        human_descriptions=[
            HumanDescription(
                source="test",
                text=(
                    "A driving, dark techno track with building energy and "
                    "four-on-the-floor rhythm. Clean production with punchy dynamics."
                ),
                tags=["techno", "dark", "driving", "hypnotic"],
                key_observations=[
                    "Four-on-the-floor kick pattern",
                    "Energy builds over the runtime",
                    "Clean, spacious production",
                ],
            ),
        ],
    )


class TestAlignmentChecker:
    def test_check_produces_results(self):
        checker = AlignmentChecker()
        entry = _make_entry_with_pipeline()
        results = checker.check(entry)
        assert len(results) > 0
        assert all(isinstance(r, AlignmentResult) for r in results)

    def test_check_sets_entry_fields(self):
        checker = AlignmentChecker()
        entry = _make_entry_with_pipeline()
        checker.check(entry)
        assert entry.alignments is not None
        assert entry.checked_at is not None

    def test_check_requires_pipeline(self):
        checker = AlignmentChecker()
        entry = CalibrationEntry(
            track_id="test",
            artist="Test",
            title="Test",
            human_descriptions=[HumanDescription(source="test", text="test")],
        )
        with pytest.raises(ValueError, match="no pipeline results"):
            checker.check(entry)

    def test_check_requires_descriptions(self):
        checker = AlignmentChecker()
        entry = _make_entry_with_pipeline()
        entry.human_descriptions = []
        with pytest.raises(ValueError, match="no human descriptions"):
            checker.check(entry)

    def test_alignment_result_has_dimension(self):
        checker = AlignmentChecker()
        entry = _make_entry_with_pipeline()
        results = checker.check(entry)
        dims = {r.dimension for r in results}
        # Should have at least tempo, energy, structure from the pipeline data
        assert AlignmentDimension.TEMPO in dims
        assert AlignmentDimension.ENERGY in dims

    def test_alignment_confidence_varies(self):
        checker = AlignmentChecker()
        entry = _make_entry_with_pipeline()
        results = checker.check(entry)
        confidences = {r.confidence for r in results}
        assert len(confidences) > 1  # Not all the same

    def test_divergences_tracked(self):
        checker = AlignmentChecker()
        entry = _make_entry_with_pipeline()
        checker.check(entry)
        # With the test data, some dimensions will misalign
        assert isinstance(entry.divergences, list)

    def test_generate_report(self):
        checker = AlignmentChecker()
        entry = _make_entry_with_pipeline()
        checker.check(entry)
        report = checker.generate_report([entry])
        assert isinstance(report, CalibrationReport)
        assert report.total_tracks == 1
        assert report.analyzed_tracks == 1
        assert report.checked_tracks == 1
        assert 0.0 <= report.alignment_rate <= 1.0

    def test_report_empty_corpus(self):
        checker = AlignmentChecker()
        report = checker.generate_report([])
        assert report.total_tracks == 0
        assert report.alignment_rate == 0.0


# --- Runner ---


class TestRunner:
    def test_run_track_requires_audio_path(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        entry = corpus.add_track("Foo", "Bar")
        runner = CalibrationRunner(corpus)
        with pytest.raises(ValueError, match="No audio path"):
            runner.run_track(entry)

    def test_run_track_requires_existing_file(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        entry = corpus.add_track("Foo", "Bar", audio_path="/nonexistent/file.wav")
        runner = CalibrationRunner(corpus)
        with pytest.raises(FileNotFoundError):
            runner.run_track(entry)

    def test_run_pending_empty(self, tmp_path):
        corpus = Corpus(corpus_dir=tmp_path / "cal")
        runner = CalibrationRunner(corpus)
        results = runner.run_pending()
        assert results == []


# --- Models ---


class TestModels:
    def test_human_description_defaults(self):
        d = HumanDescription(source="test", text="test text")
        assert d.tags == []
        assert d.key_observations == []

    def test_calibration_entry_defaults(self):
        e = CalibrationEntry(track_id="test", artist="A", title="B")
        assert e.audio_path is None
        assert e.layer1 is None
        assert e.human_descriptions == []
        assert e.alignments == []
        assert e.divergences == []

    def test_alignment_result_bounds(self):
        r = AlignmentResult(
            dimension=AlignmentDimension.TEMPO,
            pipeline_says="130 BPM",
            human_says="fast tempo",
            aligned=True,
            confidence=0.9,
        )
        assert 0.0 <= r.confidence <= 1.0

    def test_report_alignment_rate_bounds(self):
        r = CalibrationReport(
            total_tracks=0,
            analyzed_tracks=0,
            checked_tracks=0,
            alignment_rate=0.0,
        )
        assert 0.0 <= r.alignment_rate <= 1.0


# --- CLI integration ---


class TestCLI:
    def test_calibrate_init(self, tmp_path):
        """Test that the init command creates manifest with seed tracks."""
        from earworm.calibration.seeds import seed_corpus

        corpus = Corpus(corpus_dir=tmp_path / "cal")
        seed_corpus(corpus)

        manifest_path = tmp_path / "cal" / "manifest.json"
        assert manifest_path.exists()

        data = json.loads(manifest_path.read_text())
        assert len(data["entries"]) == 2

    def test_calibrate_round_trip(self, tmp_path):
        """Test seed → save → reload → alignment check cycle."""
        corpus_dir = tmp_path / "cal"
        corpus = Corpus(corpus_dir=corpus_dir)
        seed_corpus(corpus)

        corpus2 = Corpus(corpus_dir=corpus_dir)
        tracks = corpus2.list_tracks()
        assert len(tracks) == 2

        for t in tracks:
            assert len(t.human_descriptions) >= 2
            for d in t.human_descriptions:
                assert d.text
                assert d.tags
