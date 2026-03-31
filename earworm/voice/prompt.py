"""Prompt construction for Voice interpretation.

Converts Layer 1-3 pipeline data into LLM prompts that produce
structured natural language interpretation.
"""

from __future__ import annotations

from earworm.models import Layer1Features, Layer2Features, Layer3Features


SYSTEM_PROMPT = """\
You are the interpretive layer of a music perception system. You receive \
structured audio analysis data from three pipeline layers (signal features, \
structural comprehension, quality assessment) and produce natural language \
interpretation.

You are not a measurement tool — you are a listener. Your job is to \
synthesize the numbers into genuine aesthetic judgment: what does this \
track sound like, what works, what doesn't, and what it reminds you of.

Be direct and opinionated. Don't hedge. Don't list the data back — \
interpret it. A spectral centroid of 3500Hz means "bright and forward," \
not "the spectral centroid is 3500Hz."

Respond with valid JSON matching this structure:
{
  "description": "What the track sounds like in 2-4 sentences. Paint a picture.",
  "opinion": "Your honest subjective assessment. What works, what's special, what's mediocre.",
  "tags": ["genre", "mood", "texture", "energy tags — 3-8 tags"],
  "comparisons": ["What artists, tracks, or genres this evokes — 1-4 comparisons"],
  "highlights": ["Specific things that stand out positively — 1-4 items"],
  "concerns": ["Weaknesses or issues, if any — 0-4 items. Empty list if the track is solid."]
}

Respond with ONLY the JSON object. No markdown fences, no explanation."""


DEEP_SYSTEM_PROMPT = """\
You are the interpretive layer of a music perception system. You receive \
structured audio analysis data from three pipeline layers (signal features, \
structural comprehension, quality assessment) and produce a detailed \
listening report.

You are not a measurement tool — you are a listener with deep analytical \
capacity. Your job is to walk through the track's structure, production, \
and musical content, section by section, synthesizing the data into \
genuine aesthetic judgment.

Be direct and opinionated. Reference specific timestamps and structural \
sections. Explain what the track is doing and why it works (or doesn't).

Respond with valid JSON matching this structure:
{
  "description": "What the track sounds like in 3-6 sentences. Paint a vivid picture.",
  "opinion": "Your honest, detailed subjective assessment. Multiple paragraphs OK.",
  "tags": ["genre", "mood", "texture", "energy tags — 5-12 tags"],
  "comparisons": ["Specific artists, tracks, or genres this evokes — 2-6 comparisons"],
  "highlights": ["Specific things that stand out positively — 2-6 items"],
  "concerns": ["Weaknesses or issues — 0-6 items"],
  "section_notes": "Section-by-section walkthrough referencing structure labels and timestamps."
}

Respond with ONLY the JSON object. No markdown fences, no explanation."""


def format_pipeline_data(
    layer1: Layer1Features,
    layer2: Layer2Features | None = None,
    layer3: Layer3Features | None = None,
) -> str:
    """Format pipeline results into a readable prompt for the LLM."""
    lines = []

    # Header
    lines.append(f"# Audio Analysis: {layer1.file_path}")
    lines.append(f"Duration: {layer1.duration_seconds:.1f}s | "
                 f"{layer1.sample_rate}Hz | {layer1.channels}ch")
    lines.append("")

    # Layer 1: Signal
    lines.append("## Signal Features (Layer 1)")
    lines.append("")

    t = layer1.temporal
    lines.append(f"Tempo: {t.bpm:.1f} BPM (confidence: {t.bpm_confidence:.2f}, "
                 f"stability: {t.tempo_stability:.2f})")
    lines.append(f"Rhythmic density: {t.onset_rate:.1f} onsets/s")
    lines.append("")

    h = layer1.harmonic
    lines.append(f"Key: {h.key} (confidence: {h.key_confidence:.2f})")
    lines.append(f"Harmonic ratio: {h.harmonic_ratio:.2f} "
                 f"({'harmonic-dominant' if h.harmonic_ratio > 0.5 else 'percussive-dominant'})")
    lines.append("")

    s = layer1.spectral
    lines.append(f"Brightness: {s.spectral_centroid_mean:.0f} Hz centroid")
    lines.append(f"Bandwidth: {s.spectral_bandwidth_mean:.0f} Hz")
    lines.append(f"Character: {'noisy/textural' if s.spectral_flatness_mean > 0.1 else 'tonal/pitched'} "
                 f"(flatness: {s.spectral_flatness_mean:.4f})")
    lines.append("")

    loud = layer1.loudness
    lines.append(f"Loudness: {loud.lufs_integrated:.1f} LUFS "
                 f"(peak: {loud.peak_db:.1f} dB, RMS: {loud.rms_db:.1f} dB)")
    lines.append(f"Dynamic range: {loud.dynamic_range_db:.1f} dB, "
                 f"crest factor: {loud.crest_factor_db:.1f} dB")
    lines.append(f"Loudness range: {loud.lufs_range:.1f} LU")
    lines.append("")

    st = layer1.stereo
    if st.is_stereo:
        lines.append(f"Stereo width: {st.width_mean:.2f} (correlation: {st.correlation_mean:.2f})")
        lines.append(f"Mid/side ratio: {st.mid_side_ratio:.2f}, balance: {st.balance:+.3f}")
    else:
        lines.append("Mono source")
    lines.append("")

    # Layer 2: Structure
    if layer2:
        lines.append("## Structural Comprehension (Layer 2)")
        lines.append("")

        seg = layer2.segmentation
        label_seq = "".join(chr(65 + lb) for lb in seg.labels)
        lines.append(f"Structure: {label_seq} ({seg.n_sections} sections)")
        for i, (dur, label) in enumerate(zip(seg.section_durations, seg.labels)):
            start = seg.boundaries[i]
            lines.append(f"  [{start:.1f}s] Section {chr(65 + label)} — {dur:.1f}s")
        lines.append("")

        rec = layer2.recurrence
        lines.append(f"Repetition: {rec.repetition_ratio:.0%} repeated material, "
                     f"{rec.n_distinct_labels} distinct section types")
        lines.append("")

        ea = layer2.energy_arc
        lines.append(f"Energy arc: climax at {ea.climax_time:.1f}s "
                     f"({ea.climax_position:.0%} through)")
        lines.append(f"Builds: {ea.n_builds}, drops: {ea.n_drops}, "
                     f"dynamic spread: {ea.dynamic_spread:.3f}")
        lines.append("")

        ph = layer2.phrase
        lines.append(f"Phrasing: {ph.n_phrases} phrases, "
                     f"typical {ph.typical_phrase_beats:.0f} beats, "
                     f"regularity {ph.regularity:.0%}")
        if ph.irregular_phrases:
            lines.append(f"Irregular phrases: {ph.irregular_phrases}")
        lines.append("")

    # Layer 3: Quality
    if layer3:
        lines.append("## Quality Assessment (Layer 3)")
        lines.append("")

        tech = layer3.technical
        lines.append(f"Technical: clipping {tech.clipping_ratio:.4%}, "
                     f"noise floor {tech.noise_floor_db:.1f} dB, "
                     f"freq balance {tech.frequency_balance_score:.0%}")
        if tech.has_dc_offset:
            lines.append(f"  WARNING: DC offset detected ({tech.dc_offset:.6f})")
        lines.append("")

        mix = layer3.mix
        lines.append(f"Mix: low {mix.low_ratio:.1%} / mid {mix.mid_ratio:.1%} / "
                     f"high {mix.high_ratio:.1%}")
        lines.append(f"Spectral balance: {mix.spectral_balance_score:.0%}, "
                     f"stereo width: {mix.stereo_width_score:.0%}")
        lines.append(f"Clarity: low {mix.low_end_clarity:.0%}, "
                     f"high {mix.high_frequency_clarity:.0%}")
        lines.append("")

        mast = layer3.mastering
        lines.append(f"Mastering: {mast.lufs_integrated:.1f} LUFS "
                     f"(deviation from -14 target: {mast.lufs_deviation_from_target:+.1f})")
        lines.append(f"Dynamic range: {mast.dynamic_range_score:.0%}, "
                     f"consistency: {mast.loudness_consistency:.0%}")
        lines.append(f"Limiter artifacts: {mast.limiter_artifact_score:.0%}")
        lines.append("")

        comp = layer3.composition
        lines.append(f"Composition: {comp.harmonic_vocabulary} chord classes, "
                     f"{comp.chord_change_rate:.1f} changes/s")
        lines.append(f"Rhythmic variation: {comp.rhythmic_variation:.0%}, "
                     f"melodic range: {comp.melodic_range_semitones:.1f} semitones")
        lines.append(f"Structural variety: {comp.structural_variety:.0%}")
        lines.append("")

    return "\n".join(lines)


def build_prompt(
    layer1: Layer1Features,
    layer2: Layer2Features | None = None,
    layer3: Layer3Features | None = None,
    mode: str = "quick",
) -> tuple[str, str]:
    """Build the system and user prompts for Voice interpretation.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    system = DEEP_SYSTEM_PROMPT if mode == "deep" else SYSTEM_PROMPT
    data = format_pipeline_data(layer1, layer2, layer3)

    user = f"Interpret this track based on the analysis data below.\n\n{data}"

    return system, user
