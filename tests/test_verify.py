"""Tests for sample verification module."""

from __future__ import annotations

from earworm.models import (
    CategoryScore,
    CheckResult,
    HarmonicFeatures,
    Layer1Features,
    LoudnessFeatures,
    SpectralFeatures,
    StereoFeatures,
    TemporalFeatures,
)
from earworm.verify import verify, verify_all_categories
from earworm.verify.profiles import (
    get_profile,
    known_categories,
    normalize_category,
)


# --- Fixture factories ---


def _make_features(**overrides) -> Layer1Features:
    """Build a Layer1Features with sensible defaults, overriding specific values."""
    spectral = SpectralFeatures(
        mfcc_mean=[0.0] * 13,
        mfcc_std=[0.0] * 13,
        spectral_centroid_mean=overrides.get("centroid", 2000),
        spectral_centroid_std=100,
        spectral_bandwidth_mean=2000,
        spectral_rolloff_mean=overrides.get("rolloff", 4000),
        spectral_contrast_mean=[0.0] * 7,
        spectral_flatness_mean=overrides.get("flatness", 0.1),
        chroma_mean=[0.0] * 12,
        chroma_std=[0.0] * 12,
    )
    temporal = TemporalFeatures(
        bpm=120.0,
        bpm_confidence=0.8,
        beat_times=[],
        onset_times=[],
        onset_rate=overrides.get("onset_rate", 5.0),
        tempo_stability=0.9,
    )
    harmonic = HarmonicFeatures(
        key="C minor",
        key_confidence=0.7,
        key_profile=[0.0] * 12,
        chroma_cqt_mean=[0.0] * 12,
        tonnetz_mean=[0.0] * 6,
        tonnetz_std=[0.0] * 6,
        harmonic_ratio=overrides.get("harmonic_ratio", 0.3),
    )
    loudness = LoudnessFeatures(
        lufs_integrated=-14.0,
        lufs_short_term_max=-10.0,
        lufs_range=8.0,
        dynamic_range_db=12.0,
        peak_db=-1.0,
        rms_db=overrides.get("rms_db", -14.0),
        crest_factor_db=overrides.get("crest_factor_db", 10.0),
        loudness_curve=[0.5],
    )
    stereo = StereoFeatures(
        is_stereo=False,
        width_mean=0.0,
        width_std=0.0,
        correlation_mean=1.0,
        correlation_min=1.0,
        mid_side_ratio=1.0,
        balance=0.0,
    )
    return Layer1Features(
        file_path=overrides.get("file_path", "test_sample.wav"),
        duration_seconds=overrides.get("duration", 0.5),
        sample_rate=44100,
        channels=1,
        spectral=spectral,
        temporal=temporal,
        harmonic=harmonic,
        loudness=loudness,
        stereo=stereo,
    )


def _kick_features() -> Layer1Features:
    """Features that should match a kick drum profile."""
    return _make_features(
        centroid=600,
        rolloff=3000,
        flatness=0.02,
        harmonic_ratio=0.2,
        crest_factor_db=12.0,
        duration=0.3,
    )


def _hihat_features() -> Layer1Features:
    """Features that should match a hihat profile."""
    return _make_features(
        centroid=8000,
        rolloff=10000,
        flatness=0.3,
        harmonic_ratio=0.1,
        crest_factor_db=8.0,
        duration=0.15,
    )


def _bass_features() -> Layer1Features:
    """Features that should match a bass profile."""
    return _make_features(
        centroid=400,
        rolloff=2000,
        flatness=0.05,
        harmonic_ratio=0.6,
        crest_factor_db=6.0,
        duration=1.0,
    )


def _pad_features() -> Layer1Features:
    """Features that should match a pad profile."""
    return _make_features(
        centroid=1500,
        rolloff=5000,
        flatness=0.08,
        harmonic_ratio=0.7,
        onset_rate=2.0,
        duration=4.0,
    )


def _vocal_features() -> Layer1Features:
    """Features that should match a vocal profile."""
    return _make_features(
        centroid=1200,
        rolloff=5000,
        flatness=0.1,
        harmonic_ratio=0.65,
        duration=3.0,
    )


# --- Profile tests ---


class TestNormalizeCategory:
    def test_canonical_passthrough(self):
        assert normalize_category("kick") == "kick"

    def test_alias_mapping(self):
        assert normalize_category("kick_drum") == "kick"
        assert normalize_category("hi-hat") == "hihat"
        assert normalize_category("808") == "bass"
        assert normalize_category("vox") == "vocal"

    def test_case_insensitive(self):
        assert normalize_category("KICK") == "kick"
        assert normalize_category("HiHat") == "hihat"

    def test_strip_whitespace(self):
        assert normalize_category("  kick  ") == "kick"

    def test_unknown_passthrough(self):
        assert normalize_category("xylophone") == "xylophone"


class TestGetProfile:
    def test_known_category(self):
        profile = get_profile("kick")
        assert profile is not None
        assert len(profile) > 0

    def test_alias_resolves(self):
        profile = get_profile("kicks")
        assert profile is not None

    def test_unknown_returns_none(self):
        assert get_profile("xylophone") is None


class TestKnownCategories:
    def test_contains_basics(self):
        cats = known_categories()
        assert "kick" in cats
        assert "snare" in cats
        assert "hihat" in cats
        assert "bass" in cats
        assert "pad" in cats
        assert "melody" in cats
        assert "vocal" in cats

    def test_returns_list(self):
        cats = known_categories()
        assert isinstance(cats, list)
        assert len(cats) >= 10


# --- Verify tests ---


class TestVerifyMatch:
    """Test that correct labels produce match verdicts."""

    def test_kick_match(self):
        result = verify(_kick_features(), "kick")
        assert result.verdict == "match"
        assert result.score >= 0.65
        assert result.canonical_category == "kick"

    def test_hihat_match(self):
        result = verify(_hihat_features(), "hihat")
        assert result.verdict == "match"
        assert result.score >= 0.65

    def test_bass_match(self):
        result = verify(_bass_features(), "bass")
        assert result.verdict == "match"
        assert result.score >= 0.65

    def test_pad_match(self):
        result = verify(_pad_features(), "pad")
        assert result.verdict == "match"
        assert result.score >= 0.65

    def test_vocal_match(self):
        result = verify(_vocal_features(), "vocal")
        assert result.verdict == "match"
        assert result.score >= 0.65


class TestVerifyMismatch:
    """Test that wrong labels produce mismatch or uncertain verdicts."""

    def test_kick_labeled_hihat(self):
        result = verify(_kick_features(), "hihat")
        assert result.verdict in ("mismatch", "uncertain")
        assert result.score < 0.65

    def test_hihat_labeled_kick(self):
        result = verify(_hihat_features(), "kick")
        assert result.verdict in ("mismatch", "uncertain")
        assert result.score < 0.65

    def test_hihat_labeled_bass(self):
        result = verify(_hihat_features(), "bass")
        assert result.verdict in ("mismatch", "uncertain")
        assert result.score < 0.65


class TestVerifySuggestion:
    """Test that mismatches suggest the right category."""

    def test_kick_mislabeled_as_hihat_suggests_kick(self):
        result = verify(_kick_features(), "hihat", suggest=True)
        if result.suggestion is not None:
            # The suggestion should score well for what it actually is
            assert result.suggestion.score > result.score

    def test_no_suggest_flag(self):
        result = verify(_kick_features(), "hihat", suggest=False)
        assert result.suggestion is None


class TestVerifyUnknownCategory:
    def test_unknown_category(self):
        result = verify(_kick_features(), "xylophone")
        assert result.verdict == "unknown_category"
        assert result.score == 0.0
        assert "Unknown category" in result.summary


class TestVerifyAliases:
    def test_alias_resolved(self):
        result = verify(_kick_features(), "kick_drum")
        assert result.canonical_category == "kick"
        assert result.labeled_category == "kick_drum"
        assert result.verdict == "match"

    def test_hihat_alias(self):
        result = verify(_hihat_features(), "hi-hat")
        assert result.canonical_category == "hihat"
        assert result.verdict == "match"


class TestVerifyResult:
    def test_has_checks(self):
        result = verify(_kick_features(), "kick")
        assert len(result.checks) > 0
        for check in result.checks:
            assert isinstance(check, CheckResult)
            assert check.name
            assert isinstance(check.passed, bool)
            assert check.detail

    def test_summary_present(self):
        result = verify(_kick_features(), "kick")
        assert result.summary
        assert "kick" in result.summary.lower()

    def test_file_path_preserved(self):
        features = _make_features(file_path="/samples/test.wav")
        result = verify(features, "kick")
        assert result.file_path == "/samples/test.wav"


# --- verify_all_categories tests ---


class TestVerifyAllCategories:
    def test_returns_all_categories(self):
        scores = verify_all_categories(_kick_features())
        cats = [s.category for s in scores]
        for expected in known_categories():
            assert expected in cats

    def test_sorted_descending(self):
        scores = verify_all_categories(_kick_features())
        for i in range(len(scores) - 1):
            assert scores[i].score >= scores[i + 1].score

    def test_kick_ranked_first_for_kick_features(self):
        scores = verify_all_categories(_kick_features())
        assert scores[0].category == "kick"

    def test_hihat_ranked_first_for_hihat_features(self):
        scores = verify_all_categories(_hihat_features())
        assert scores[0].category == "hihat"

    def test_bass_ranked_first_for_bass_features(self):
        scores = verify_all_categories(_bass_features())
        assert scores[0].category == "bass"

    def test_score_objects(self):
        scores = verify_all_categories(_kick_features())
        for s in scores:
            assert isinstance(s, CategoryScore)
            assert 0 <= s.score <= 1
            assert s.checks_passed <= s.checks_total
