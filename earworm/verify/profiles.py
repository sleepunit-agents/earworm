"""Category profiles for sample verification.

Each profile is a list of CheckDef objects. A CheckDef defines a named check with a
weight (0-1, importance) and a test function that returns (passed, detail_string).

Thresholds are based on audio physics principles:
- Spectral centroid reflects frequency balance (low=bass-heavy, high=bright/metallic)
- Harmonic ratio separates tonal sounds (>0.5) from percussive/noise sounds (<0.5)
- Crest factor measures punch — impulsive one-shots have high peak-to-RMS ratios (>6 dB)
- Spectral flatness separates noise (>0.3) from tonal/musical content (<0.2)
- Duration and onset_rate distinguish one-shots from loops
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from earworm.models import Layer1Features


@dataclass
class CheckDef:
    """Definition of a single profile check."""

    name: str
    weight: float
    test: Callable[[Layer1Features], tuple[bool, str]]


# --- Category aliases → canonical name ---

ALIASES: dict[str, str] = {
    "kick_drum": "kick",
    "kicks": "kick",
    "snaredrum": "snare",
    "snares": "snare",
    "hi_hat": "hihat",
    "hi-hat": "hihat",
    "hh": "hihat",
    "hats": "hihat",
    "hat": "hihat",
    "claps": "clap",
    "cymbals": "cymbal",
    "crash": "cymbal",
    "ride": "cymbal",
    "basses": "bass",
    "sub": "bass",
    "808": "bass",
    "pads": "pad",
    "synth_pad": "pad",
    "chords": "chord",
    "stab": "chord",
    "stabs": "chord",
    "leads": "melody",
    "lead": "melody",
    "synth": "melody",
    "vocals": "vocal",
    "vox": "vocal",
    "voice": "vocal",
    "vocal_chop": "vocal",
    "fx": "sfx",
    "effect": "sfx",
    "effects": "sfx",
    "sfx": "sfx",
    "drones": "drone",
    "atmos": "drone",
    "atmosphere": "drone",
    "texture": "drone",
    "loops": "loop",
    "break": "loop",
    "breaks": "loop",
    "one_shot": "oneshot",
    "one-shot": "oneshot",
    "oneshots": "oneshot",
    "noises": "noise",
    "white_noise": "noise",
}


def normalize_category(category: str) -> str:
    """Normalize a category label to its canonical form."""
    key = category.lower().strip().replace(" ", "_")
    return ALIASES.get(key, key)


# --- Profile factories ---


def _kick_profile() -> list[CheckDef]:
    return [
        CheckDef(
            name="low_frequency_content",
            weight=0.8,
            test=lambda f: (
                f.spectral.spectral_centroid_mean < 1500,
                f"spectral centroid {f.spectral.spectral_centroid_mean:.0f}Hz"
                f" — expected <1500Hz for kicks",
            ),
        ),
        CheckDef(
            name="percussive_character",
            weight=0.7,
            test=lambda f: (
                f.harmonic.harmonic_ratio < 0.55,
                f"harmonic ratio {f.harmonic.harmonic_ratio:.2f}"
                f" — expected <0.55 for percussive sounds",
            ),
        ),
        CheckDef(
            name="punch_crest",
            weight=0.6,
            test=lambda f: (
                f.loudness.crest_factor_db > 6.0,
                f"crest factor {f.loudness.crest_factor_db:.1f}dB"
                f" — expected >6dB for punchy one-shots",
            ),
        ),
        CheckDef(
            name="one_shot_length",
            weight=0.5,
            test=lambda f: (
                f.duration_seconds < 4.0,
                f"duration {f.duration_seconds:.2f}s — expected <4s for kick one-shots",
            ),
        ),
        CheckDef(
            name="low_rolloff",
            weight=0.4,
            test=lambda f: (
                f.spectral.spectral_rolloff_mean < 8000,
                f"spectral rolloff {f.spectral.spectral_rolloff_mean:.0f}Hz"
                f" — expected <8000Hz (energy concentrated in lows/mids)",
            ),
        ),
    ]


def _snare_profile() -> list[CheckDef]:
    return [
        CheckDef(
            name="mid_frequency_content",
            weight=0.7,
            test=lambda f: (
                500 < f.spectral.spectral_centroid_mean < 8000,
                f"spectral centroid {f.spectral.spectral_centroid_mean:.0f}Hz"
                f" — expected 500-8000Hz for snares",
            ),
        ),
        CheckDef(
            name="percussive_character",
            weight=0.7,
            test=lambda f: (
                f.harmonic.harmonic_ratio < 0.55,
                f"harmonic ratio {f.harmonic.harmonic_ratio:.2f}"
                f" — expected <0.55 for percussive sounds",
            ),
        ),
        CheckDef(
            name="punch_crest",
            weight=0.6,
            test=lambda f: (
                f.loudness.crest_factor_db > 6.0,
                f"crest factor {f.loudness.crest_factor_db:.1f}dB"
                f" — expected >6dB for punchy one-shots",
            ),
        ),
        CheckDef(
            name="one_shot_length",
            weight=0.5,
            test=lambda f: (
                f.duration_seconds < 4.0,
                f"duration {f.duration_seconds:.2f}s — expected <4s for snare one-shots",
            ),
        ),
        CheckDef(
            name="noise_content",
            weight=0.4,
            test=lambda f: (
                f.spectral.spectral_flatness_mean > 0.05,
                f"spectral flatness {f.spectral.spectral_flatness_mean:.3f}"
                f" — expected >0.05 (snares have noisy snare wire component)",
            ),
        ),
    ]


def _hihat_profile() -> list[CheckDef]:
    return [
        CheckDef(
            name="high_frequency_content",
            weight=0.8,
            test=lambda f: (
                f.spectral.spectral_centroid_mean > 4000,
                f"spectral centroid {f.spectral.spectral_centroid_mean:.0f}Hz"
                f" — expected >4000Hz for hihats",
            ),
        ),
        CheckDef(
            name="noisy_metallic_character",
            weight=0.7,
            test=lambda f: (
                f.spectral.spectral_flatness_mean > 0.1,
                f"spectral flatness {f.spectral.spectral_flatness_mean:.3f}"
                f" — expected >0.1 for metallic/noisy hihats",
            ),
        ),
        CheckDef(
            name="short_duration",
            weight=0.5,
            test=lambda f: (
                f.duration_seconds < 3.0,
                f"duration {f.duration_seconds:.2f}s — expected <3s for hihat one-shots",
            ),
        ),
        CheckDef(
            name="high_rolloff",
            weight=0.5,
            test=lambda f: (
                f.spectral.spectral_rolloff_mean > 5000,
                f"spectral rolloff {f.spectral.spectral_rolloff_mean:.0f}Hz"
                f" — expected >5000Hz (energy in high frequencies)",
            ),
        ),
    ]


def _clap_profile() -> list[CheckDef]:
    return [
        CheckDef(
            name="percussive_character",
            weight=0.7,
            test=lambda f: (
                f.harmonic.harmonic_ratio < 0.5,
                f"harmonic ratio {f.harmonic.harmonic_ratio:.2f}"
                f" — expected <0.5 for percussive claps",
            ),
        ),
        CheckDef(
            name="punch_crest",
            weight=0.6,
            test=lambda f: (
                f.loudness.crest_factor_db > 6.0,
                f"crest factor {f.loudness.crest_factor_db:.1f}dB"
                f" — expected >6dB for impulsive clap transients",
            ),
        ),
        CheckDef(
            name="mid_high_frequency",
            weight=0.5,
            test=lambda f: (
                f.spectral.spectral_centroid_mean > 1000,
                f"spectral centroid {f.spectral.spectral_centroid_mean:.0f}Hz"
                f" — expected >1000Hz for claps (broad mid-high spectrum)",
            ),
        ),
        CheckDef(
            name="short_duration",
            weight=0.5,
            test=lambda f: (
                f.duration_seconds < 2.0,
                f"duration {f.duration_seconds:.2f}s — expected <2s for clap one-shots",
            ),
        ),
        CheckDef(
            name="noise_content",
            weight=0.4,
            test=lambda f: (
                f.spectral.spectral_flatness_mean > 0.05,
                f"spectral flatness {f.spectral.spectral_flatness_mean:.3f}"
                f" — expected >0.05 (claps have broad noise spectrum)",
            ),
        ),
    ]


def _cymbal_profile() -> list[CheckDef]:
    return [
        CheckDef(
            name="high_frequency_content",
            weight=0.7,
            test=lambda f: (
                f.spectral.spectral_centroid_mean > 3000,
                f"spectral centroid {f.spectral.spectral_centroid_mean:.0f}Hz"
                f" — expected >3000Hz for cymbals",
            ),
        ),
        CheckDef(
            name="metallic_noise_character",
            weight=0.6,
            test=lambda f: (
                f.spectral.spectral_flatness_mean > 0.05,
                f"spectral flatness {f.spectral.spectral_flatness_mean:.3f}"
                f" — expected >0.05 for metallic cymbal sound",
            ),
        ),
        CheckDef(
            name="percussive_or_sustained",
            weight=0.4,
            test=lambda f: (
                f.duration_seconds < 10.0,
                f"duration {f.duration_seconds:.2f}s — expected <10s for cymbal one-shots",
            ),
        ),
    ]


def _bass_profile() -> list[CheckDef]:
    return [
        CheckDef(
            name="low_frequency_content",
            weight=0.8,
            test=lambda f: (
                f.spectral.spectral_centroid_mean < 1000,
                f"spectral centroid {f.spectral.spectral_centroid_mean:.0f}Hz"
                f" — expected <1000Hz for bass sounds",
            ),
        ),
        CheckDef(
            name="low_rolloff",
            weight=0.7,
            test=lambda f: (
                f.spectral.spectral_rolloff_mean < 4000,
                f"spectral rolloff {f.spectral.spectral_rolloff_mean:.0f}Hz"
                f" — expected <4000Hz (bass energy concentrated in lows)",
            ),
        ),
        CheckDef(
            name="not_pure_noise",
            weight=0.5,
            test=lambda f: (
                f.spectral.spectral_flatness_mean < 0.5,
                f"spectral flatness {f.spectral.spectral_flatness_mean:.3f}"
                f" — expected <0.5 (bass should have some tonal content)",
            ),
        ),
    ]


def _pad_profile() -> list[CheckDef]:
    return [
        CheckDef(
            name="harmonic_tonal",
            weight=0.7,
            test=lambda f: (
                f.harmonic.harmonic_ratio > 0.4,
                f"harmonic ratio {f.harmonic.harmonic_ratio:.2f}"
                f" — expected >0.4 for tonal pads",
            ),
        ),
        CheckDef(
            name="not_noisy",
            weight=0.6,
            test=lambda f: (
                f.spectral.spectral_flatness_mean < 0.4,
                f"spectral flatness {f.spectral.spectral_flatness_mean:.3f}"
                f" — expected <0.4 for tonal pad character",
            ),
        ),
        CheckDef(
            name="sustained_low_onset_rate",
            weight=0.5,
            test=lambda f: (
                f.temporal.onset_rate < 10,
                f"onset rate {f.temporal.onset_rate:.1f}/s"
                f" — expected <10/s for sustained pad textures",
            ),
        ),
        CheckDef(
            name="meaningful_duration",
            weight=0.4,
            test=lambda f: (
                f.duration_seconds > 0.5,
                f"duration {f.duration_seconds:.2f}s — expected >0.5s for pad sounds",
            ),
        ),
    ]


def _chord_profile() -> list[CheckDef]:
    return [
        CheckDef(
            name="harmonic_tonal",
            weight=0.7,
            test=lambda f: (
                f.harmonic.harmonic_ratio > 0.4,
                f"harmonic ratio {f.harmonic.harmonic_ratio:.2f}"
                f" — expected >0.4 for harmonic chord stabs",
            ),
        ),
        CheckDef(
            name="not_noisy",
            weight=0.6,
            test=lambda f: (
                f.spectral.spectral_flatness_mean < 0.4,
                f"spectral flatness {f.spectral.spectral_flatness_mean:.3f}"
                f" — expected <0.4 (chords should be tonal, not noise)",
            ),
        ),
        CheckDef(
            name="mid_frequency_presence",
            weight=0.5,
            test=lambda f: (
                200 < f.spectral.spectral_centroid_mean < 8000,
                f"spectral centroid {f.spectral.spectral_centroid_mean:.0f}Hz"
                f" — expected 200-8000Hz for chord stabs",
            ),
        ),
    ]


def _melody_profile() -> list[CheckDef]:
    return [
        CheckDef(
            name="harmonic_pitched",
            weight=0.7,
            test=lambda f: (
                f.harmonic.harmonic_ratio > 0.4,
                f"harmonic ratio {f.harmonic.harmonic_ratio:.2f}"
                f" — expected >0.4 for melodic pitched sounds",
            ),
        ),
        CheckDef(
            name="mid_frequency_range",
            weight=0.6,
            test=lambda f: (
                200 < f.spectral.spectral_centroid_mean < 6000,
                f"spectral centroid {f.spectral.spectral_centroid_mean:.0f}Hz"
                f" — expected 200-6000Hz for melodic sounds",
            ),
        ),
        CheckDef(
            name="not_noisy",
            weight=0.5,
            test=lambda f: (
                f.spectral.spectral_flatness_mean < 0.4,
                f"spectral flatness {f.spectral.spectral_flatness_mean:.3f}"
                f" — expected <0.4 for tonal melody",
            ),
        ),
    ]


def _vocal_profile() -> list[CheckDef]:
    return [
        CheckDef(
            name="vocal_frequency_range",
            weight=0.7,
            test=lambda f: (
                200 < f.spectral.spectral_centroid_mean < 5000,
                f"spectral centroid {f.spectral.spectral_centroid_mean:.0f}Hz"
                f" — expected 200-5000Hz for vocals",
            ),
        ),
        CheckDef(
            name="harmonic_voice",
            weight=0.6,
            test=lambda f: (
                f.harmonic.harmonic_ratio > 0.4,
                f"harmonic ratio {f.harmonic.harmonic_ratio:.2f}"
                f" — expected >0.4 (voice is a harmonic instrument)",
            ),
        ),
        CheckDef(
            name="not_pure_noise",
            weight=0.5,
            test=lambda f: (
                f.spectral.spectral_flatness_mean < 0.5,
                f"spectral flatness {f.spectral.spectral_flatness_mean:.3f}"
                f" — expected <0.5 for vocal recordings",
            ),
        ),
    ]


def _sfx_profile() -> list[CheckDef]:
    # FX/SFX is intentionally permissive — anything unusual can be an effect.
    # Just check it's non-silent (has some content).
    return [
        CheckDef(
            name="has_audio_content",
            weight=0.8,
            test=lambda f: (
                f.loudness.rms_db > -60,
                f"RMS {f.loudness.rms_db:.1f}dB — expected >-60dB (non-silent content)",
            ),
        ),
    ]


def _noise_profile() -> list[CheckDef]:
    return [
        CheckDef(
            name="noisy_character",
            weight=0.7,
            test=lambda f: (
                f.spectral.spectral_flatness_mean > 0.15,
                f"spectral flatness {f.spectral.spectral_flatness_mean:.3f}"
                f" — expected >0.15 for noise textures",
            ),
        ),
        CheckDef(
            name="low_harmonic_ratio",
            weight=0.6,
            test=lambda f: (
                f.harmonic.harmonic_ratio < 0.5,
                f"harmonic ratio {f.harmonic.harmonic_ratio:.2f}"
                f" — expected <0.5 for non-tonal noise",
            ),
        ),
    ]


def _drone_profile() -> list[CheckDef]:
    return [
        CheckDef(
            name="sustained_low_onset_rate",
            weight=0.7,
            test=lambda f: (
                f.temporal.onset_rate < 3,
                f"onset rate {f.temporal.onset_rate:.1f}/s"
                f" — expected <3/s for sustained drone textures",
            ),
        ),
        CheckDef(
            name="harmonic_tonal",
            weight=0.6,
            test=lambda f: (
                f.harmonic.harmonic_ratio > 0.3,
                f"harmonic ratio {f.harmonic.harmonic_ratio:.2f}"
                f" — expected >0.3 for tonal drone sounds",
            ),
        ),
        CheckDef(
            name="meaningful_duration",
            weight=0.5,
            test=lambda f: (
                f.duration_seconds > 1.0,
                f"duration {f.duration_seconds:.2f}s — expected >1s for drone textures",
            ),
        ),
    ]


def _loop_profile() -> list[CheckDef]:
    # Loops are diverse — main signal is they have some length and temporal structure.
    return [
        CheckDef(
            name="loop_length",
            weight=0.7,
            test=lambda f: (
                f.duration_seconds > 0.5,
                f"duration {f.duration_seconds:.2f}s — expected >0.5s for loops",
            ),
        ),
        CheckDef(
            name="has_rhythmic_activity",
            weight=0.5,
            test=lambda f: (
                f.temporal.onset_rate > 0.5,
                f"onset rate {f.temporal.onset_rate:.1f}/s"
                f" — expected >0.5/s for loops with rhythmic activity",
            ),
        ),
    ]


def _oneshot_profile() -> list[CheckDef]:
    return [
        CheckDef(
            name="short_duration",
            weight=0.8,
            test=lambda f: (
                f.duration_seconds < 5.0,
                f"duration {f.duration_seconds:.2f}s — expected <5s for one-shots",
            ),
        ),
        CheckDef(
            name="has_audio_content",
            weight=0.6,
            test=lambda f: (
                f.loudness.rms_db > -60,
                f"RMS {f.loudness.rms_db:.1f}dB — expected >-60dB (non-silent content)",
            ),
        ),
    ]


# --- Profile registry ---

_PROFILES: dict[str, list[CheckDef]] = {
    "kick": _kick_profile(),
    "snare": _snare_profile(),
    "hihat": _hihat_profile(),
    "clap": _clap_profile(),
    "cymbal": _cymbal_profile(),
    "bass": _bass_profile(),
    "pad": _pad_profile(),
    "chord": _chord_profile(),
    "melody": _melody_profile(),
    "vocal": _vocal_profile(),
    "sfx": _sfx_profile(),
    "noise": _noise_profile(),
    "drone": _drone_profile(),
    "loop": _loop_profile(),
    "oneshot": _oneshot_profile(),
}


def get_profile(category: str) -> list[CheckDef] | None:
    """Return the check definitions for a category, or None if unknown."""
    return _PROFILES.get(normalize_category(category))


def known_categories() -> list[str]:
    """Return all supported canonical category names."""
    return list(_PROFILES.keys())
