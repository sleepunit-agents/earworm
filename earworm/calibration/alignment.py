"""Alignment checker — compares pipeline perception against human descriptions.

The core of calibration: did the pipeline capture the same structural reality
that humans describe? Where does it agree, and more importantly, where does
it diverge?
"""

from __future__ import annotations

from datetime import datetime

from earworm.calibration.models import (
    AlignmentDimension,
    AlignmentResult,
    CalibrationEntry,
    CalibrationReport,
)


class AlignmentChecker:
    """Compares pipeline output against human descriptions for alignment."""

    def check(self, entry: CalibrationEntry) -> list[AlignmentResult]:
        """Run alignment checks between pipeline results and human descriptions.

        Extracts checkable claims from both pipeline output and human descriptions,
        then compares along each dimension where both have something to say.
        """
        if entry.layer1 is None:
            raise ValueError(f"Track {entry.track_id} has no pipeline results")

        if not entry.human_descriptions:
            raise ValueError(f"Track {entry.track_id} has no human descriptions")

        results: list[AlignmentResult] = []

        pipeline_claims = self._extract_pipeline_claims(entry)
        human_claims = self._extract_human_claims(entry)

        for dim in AlignmentDimension:
            p_claim = pipeline_claims.get(dim)
            h_claim = human_claims.get(dim)

            if p_claim and h_claim:
                results.append(
                    AlignmentResult(
                        dimension=dim,
                        pipeline_says=p_claim,
                        human_says=h_claim,
                        aligned=self._claims_align(dim, p_claim, h_claim),
                        confidence=self._alignment_confidence(dim, p_claim, h_claim),
                    )
                )

        entry.alignments = results
        entry.checked_at = datetime.now()

        entry.divergences = [
            f"{r.dimension.value}: pipeline='{r.pipeline_says}' vs human='{r.human_says}'"
            for r in results
            if not r.aligned
        ]

        return results

    def generate_report(self, entries: list[CalibrationEntry]) -> CalibrationReport:
        """Generate a summary report across the entire corpus."""
        total = len(entries)
        analyzed = sum(1 for e in entries if e.layer1 is not None)
        checked = sum(1 for e in entries if e.alignments)

        all_alignments = [a for e in entries for a in e.alignments]

        if all_alignments:
            alignment_rate = sum(1 for a in all_alignments if a.aligned) / len(all_alignments)
        else:
            alignment_rate = 0.0

        dim_scores: dict[str, list[bool]] = {}
        for a in all_alignments:
            dim_scores.setdefault(a.dimension.value, []).append(a.aligned)

        sorted_dims = sorted(
            dim_scores.items(),
            key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0.0,
            reverse=True,
        )

        strongest = [d for d, scores in sorted_dims[:3] if scores and sum(scores) / len(scores) > 0.5]
        weakest = [d for d, scores in sorted_dims[-3:] if scores and sum(scores) / len(scores) < 0.5]

        notable_divergences = [
            d for e in entries for d in e.divergences
        ][:10]

        return CalibrationReport(
            total_tracks=total,
            analyzed_tracks=analyzed,
            checked_tracks=checked,
            alignment_rate=alignment_rate,
            strongest_dimensions=strongest,
            weakest_dimensions=weakest,
            notable_divergences=notable_divergences,
        )

    def _extract_pipeline_claims(self, entry: CalibrationEntry) -> dict[AlignmentDimension, str]:
        """Extract checkable claims from pipeline output."""
        claims: dict[AlignmentDimension, str] = {}

        if entry.layer1:
            l1 = entry.layer1
            claims[AlignmentDimension.TEMPO] = f"{l1.temporal.bpm:.0f} BPM"
            claims[AlignmentDimension.KEY] = l1.harmonic.key

            brightness = l1.spectral.spectral_centroid_mean
            if brightness > 4000:
                claims[AlignmentDimension.TEXTURE] = "bright, forward"
            elif brightness > 2000:
                claims[AlignmentDimension.TEXTURE] = "balanced"
            else:
                claims[AlignmentDimension.TEXTURE] = "dark, warm"

            if l1.loudness.lufs_range > 10:
                claims[AlignmentDimension.DYNAMICS] = "very dynamic"
            elif l1.loudness.lufs_range > 6:
                claims[AlignmentDimension.DYNAMICS] = "moderately dynamic"
            else:
                claims[AlignmentDimension.DYNAMICS] = "compressed"

            if l1.harmonic.harmonic_ratio > 0.6:
                claims[AlignmentDimension.RHYTHM] = "harmonic-dominant"
            elif l1.harmonic.harmonic_ratio < 0.4:
                claims[AlignmentDimension.RHYTHM] = "percussion-dominant"
            else:
                claims[AlignmentDimension.RHYTHM] = "balanced harmonic/percussive"

        if entry.layer2:
            l2 = entry.layer2
            ea = l2.energy_arc
            if ea.n_builds > 2:
                claims[AlignmentDimension.ENERGY] = f"building energy ({ea.n_builds} builds)"
            elif ea.n_drops > 2:
                claims[AlignmentDimension.ENERGY] = f"dramatic drops ({ea.n_drops} drops)"
            else:
                claims[AlignmentDimension.ENERGY] = "steady energy"

            n_types = l2.recurrence.n_distinct_labels
            if n_types >= 4:
                claims[AlignmentDimension.STRUCTURE] = f"complex ({n_types} section types)"
            elif n_types >= 2:
                claims[AlignmentDimension.STRUCTURE] = f"standard ({n_types} section types)"
            else:
                claims[AlignmentDimension.STRUCTURE] = "minimal sectional variety"

        if entry.layer3:
            l3 = entry.layer3
            claims[AlignmentDimension.PRODUCTION] = (
                f"balance {l3.mix.spectral_balance_score:.0%}, "
                f"width {l3.mix.stereo_width_score:.0%}, "
                f"LUFS {l3.mastering.lufs_integrated:.1f}"
            )

        if entry.voice_result:
            claims[AlignmentDimension.MOOD] = ", ".join(entry.voice_result.tags[:5])
            claims[AlignmentDimension.GENRE] = ", ".join(entry.voice_result.tags[:3])

        return claims

    def _extract_human_claims(self, entry: CalibrationEntry) -> dict[AlignmentDimension, str]:
        """Extract checkable claims from human descriptions.

        Merges key observations from all human descriptions.
        """
        claims: dict[AlignmentDimension, str] = {}
        all_text = " ".join(d.text for d in entry.human_descriptions).lower()
        all_tags = [t.lower() for d in entry.human_descriptions for t in d.tags]
        all_observations = [o for d in entry.human_descriptions for o in d.key_observations]

        _tempo_keywords = {
            "fast": "fast tempo", "slow": "slow tempo", "mid-tempo": "mid-tempo",
            "driving": "driving tempo", "hypnotic": "hypnotic tempo",
            "uptempo": "fast tempo", "downtempo": "slow tempo",
        }
        for kw, desc in _tempo_keywords.items():
            if kw in all_text:
                claims[AlignmentDimension.TEMPO] = desc
                break

        _key_keywords = {
            "minor": "minor key", "major": "major key",
            "dark": "dark tonality", "bright": "bright tonality",
            "atonal": "atonal", "dissonant": "dissonant",
        }
        for kw, desc in _key_keywords.items():
            if kw in all_text:
                claims[AlignmentDimension.KEY] = desc
                break

        _energy_keywords = {
            "build": "building energy", "crescendo": "building energy",
            "intense": "high energy", "explosive": "high energy",
            "calm": "low energy", "ambient": "low energy",
            "dynamic": "dynamic energy", "tension": "building tension",
        }
        for kw, desc in _energy_keywords.items():
            if kw in all_text:
                claims[AlignmentDimension.ENERGY] = desc
                break

        _structure_keywords = {
            "repetitive": "repetitive structure", "complex": "complex structure",
            "evolving": "evolving structure", "layered": "layered structure",
            "minimal": "minimal structure", "through-composed": "through-composed",
            "verse-chorus": "verse-chorus", "suite": "suite-like",
        }
        for kw, desc in _structure_keywords.items():
            if kw in all_text:
                claims[AlignmentDimension.STRUCTURE] = desc
                break

        mood_tags = [t for t in all_tags if t in {
            "anxious", "joyful", "dark", "euphoric", "melancholic",
            "aggressive", "peaceful", "tense", "playful", "haunting",
            "frantic", "hypnotic", "paranoid", "ecstatic",
        }]
        if mood_tags:
            claims[AlignmentDimension.MOOD] = ", ".join(mood_tags)

        genre_tags = [t for t in all_tags if t in {
            "post-punk", "new wave", "funk", "techno", "electronic",
            "rock", "ambient", "minimal", "industrial", "art-rock",
            "afrobeat", "disco", "house", "acid", "experimental",
        }]
        if genre_tags:
            claims[AlignmentDimension.GENRE] = ", ".join(genre_tags)

        _production_keywords = {
            "lo-fi": "lo-fi production", "polished": "polished production",
            "raw": "raw production", "crisp": "crisp production",
            "muddy": "muddy production", "clean": "clean production",
            "lush": "lush production", "spacious": "spacious production",
        }
        for kw, desc in _production_keywords.items():
            if kw in all_text:
                claims[AlignmentDimension.PRODUCTION] = desc
                break

        _dynamics_keywords = {
            "dynamic": "dynamic", "compressed": "compressed",
            "loud": "loud", "quiet": "quiet", "whisper": "quiet",
            "punchy": "punchy dynamics", "flat": "flat dynamics",
        }
        for kw, desc in _dynamics_keywords.items():
            if kw in all_text:
                claims[AlignmentDimension.DYNAMICS] = desc
                break

        _texture_keywords = {
            "warm": "warm texture", "cold": "cold texture",
            "bright": "bright texture", "dark": "dark texture",
            "gritty": "gritty texture", "smooth": "smooth texture",
            "abrasive": "abrasive texture", "shimmering": "shimmering texture",
        }
        for kw, desc in _texture_keywords.items():
            if kw in all_text:
                claims[AlignmentDimension.TEXTURE] = desc
                break

        _rhythm_keywords = {
            "polyrhythm": "polyrhythmic", "syncopated": "syncopated",
            "groove": "groovy", "funky": "funky rhythm",
            "four-on-the-floor": "four-on-the-floor",
            "breakbeat": "breakbeat", "swung": "swung rhythm",
        }
        for kw, desc in _rhythm_keywords.items():
            if kw in all_text:
                claims[AlignmentDimension.RHYTHM] = desc
                break

        for obs in all_observations:
            obs_lower = obs.lower()
            for dim in AlignmentDimension:
                if dim.value in obs_lower and dim not in claims:
                    claims[dim] = obs
                    break

        return claims

    def _claims_align(
        self, dim: AlignmentDimension, pipeline: str, human: str
    ) -> bool:
        """Determine whether two claims about the same dimension align.

        This is intentionally simple — keyword overlap. The alignment check
        is a signal, not a verdict. Nuanced judgment happens in divergence review.
        """
        p_words = set(pipeline.lower().split())
        h_words = set(human.lower().split())

        shared = p_words & h_words
        filler = {"the", "a", "an", "is", "and", "or", "of", "in", "to", "with"}
        meaningful_shared = shared - filler

        if meaningful_shared:
            return True

        _compatible = {
            ("bright", "bright"), ("dark", "dark"), ("warm", "warm"),
            ("fast", "driving"), ("driving", "fast"), ("fast", "uptempo"),
            ("high energy", "intense"), ("intense", "high energy"),
            ("building", "tension"), ("building", "crescendo"),
            ("complex", "layered"), ("layered", "complex"),
            ("compressed", "loud"), ("loud", "compressed"),
            ("groovy", "funky"), ("funky", "groovy"),
            ("polyrhythmic", "syncopated"),
            ("harmonic-dominant", "harmonic"), ("percussion-dominant", "percussive"),
        }

        p_lower = pipeline.lower()
        h_lower = human.lower()
        for p_term, h_term in _compatible:
            if p_term in p_lower and h_term in h_lower:
                return True

        return False

    def _alignment_confidence(
        self, dim: AlignmentDimension, pipeline: str, human: str
    ) -> float:
        """Estimate confidence in the alignment check.

        Concrete dimensions (tempo, key) get higher base confidence
        than subjective ones (mood, texture).
        """
        base_confidence = {
            AlignmentDimension.TEMPO: 0.9,
            AlignmentDimension.KEY: 0.8,
            AlignmentDimension.DYNAMICS: 0.7,
            AlignmentDimension.RHYTHM: 0.6,
            AlignmentDimension.STRUCTURE: 0.6,
            AlignmentDimension.ENERGY: 0.5,
            AlignmentDimension.PRODUCTION: 0.5,
            AlignmentDimension.TEXTURE: 0.4,
            AlignmentDimension.MOOD: 0.3,
            AlignmentDimension.GENRE: 0.3,
        }
        return base_confidence.get(dim, 0.5)
