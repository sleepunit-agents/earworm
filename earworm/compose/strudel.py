"""Earworm → Strudel pattern generator.

Maps Layer 2 structural analysis (and optionally Layer 1 signal features)
to Strudel JavaScript code. The output is a self-contained snippet that can
be pasted into the Strudel REPL (strudel.cc) and played immediately.

Design philosophy: same as the WAV composer — the pattern *responds* to
the source track's structural DNA rather than imitating it. Section count,
energy arc, phrase rhythm, and label recurrence all transfer; the sonic
vocabulary is Strudel's own.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from earworm.models import Layer1Features, Layer2Features


# ─── Note / chord tables ──────────────────────────────────────────────────

# Chromatic note names (sharp convention — Strudel accepts both # and b)
_CHROMATIC = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]

# Scale intervals from root (semitones)
_SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
}

# Triad shapes: list of scale-degree offsets (0-indexed within the scale)
_TRIAD = [0, 2, 4]  # root, third, fifth


def _root_index(root: str) -> int:
    """Map a root note name to its chromatic index (0=C)."""
    return _CHROMATIC.index(root.lower().replace("♯", "#").replace("♭", "b"))


def _scale_notes(root: str, mode: str) -> list[str]:
    """Return the 7 note names of a scale."""
    base = _root_index(root)
    intervals = _SCALE_INTERVALS.get(mode, _SCALE_INTERVALS["minor"])
    return [_CHROMATIC[(base + iv) % 12] for iv in intervals]


def _triad_notes(scale: list[str], degree: int, octave: int = 3) -> list[str]:
    """Build a triad from a scale at a given degree, with octave numbers."""
    notes = []
    for offset in _TRIAD:
        idx = (degree + offset) % len(scale)
        note = scale[idx]
        # Wrap octave up when we cross C
        oct = octave
        if _CHROMATIC.index(note) < _CHROMATIC.index(scale[degree % len(scale)]):
            oct += 1
        notes.append(f"{note}{oct}")
    return notes


def _chord_mini(notes: list[str]) -> str:
    """Format a list of notes as a Strudel chord: [c3,e3,g3]."""
    return f"[{','.join(notes)}]"


# ─── Chord progression logic ──────────────────────────────────────────────

def _chord_progression_degrees(mode: str, n_chords: int) -> list[int]:
    """Mode-appropriate chord progression as scale degree indices."""
    if mode == "minor":
        base = [0, 2, 6, 3, 5, 2, 6, 0]
    else:
        base = [0, 3, 4, 5, 0, 3, 5, 4]
    return [base[i % len(base)] for i in range(n_chords)]


# ─── Energy → gain mapping ────────────────────────────────────────────────

def _section_gain(energy_curve: list[float], energy_timestamps: list[float],
                  section_start: float, section_end: float) -> float:
    """Average energy in a time window, mapped to a Strudel gain value (0.3–1.0)."""
    if not energy_curve:
        return 0.7
    ec = np.array(energy_curve)
    et = np.array(energy_timestamps)
    mask = (et >= section_start) & (et < section_end)
    if not mask.any():
        # Nearest sample
        idx = int(np.argmin(np.abs(et - (section_start + section_end) / 2)))
        avg = float(ec[idx])
    else:
        avg = float(ec[mask].mean())
    # Map 0–1 energy → 0.3–1.0 gain (never silent)
    return round(0.3 + avg * 0.7, 2)


# ─── Pattern builders ─────────────────────────────────────────────────────

def _build_section_pattern(
    scale: list[str],
    degree: int,
    phrase_beats: float,
    n_phrases: int,
    gain: float,
    synth: str = "supersaw",
) -> str:
    """Build a Strudel pattern string for one section.

    Returns something like:
      note("[c3,e3,g3] [c3,e3,g3] ~ [c3,e3,g3]")
        .s("supersaw").gain(0.7)
        .lpf(2000).attack(0.01).release(0.3)
    """
    chord = _chord_mini(_triad_notes(scale, degree, octave=3))

    # Build a rhythmic pattern: chord on downbeats, rest on some upbeats
    # Use a simple euclidean-ish approach: fill n_phrases slots
    slots = []
    for i in range(n_phrases):
        if i % 4 == 3 and n_phrases > 4:
            # Drop every 4th hit for breathing room
            slots.append("~")
        else:
            slots.append(chord)

    note_pattern = " ".join(slots)

    lines = [
        f'  note("{note_pattern}")',
        f'    .s("{synth}").gain({gain})',
        f"    .lpf({1200 + int(gain * 2000)}).attack(0.01).release(0.3)",
    ]
    return "\n".join(lines)


def _build_bass_pattern(
    scale: list[str],
    degree: int,
    n_phrases: int,
    gain: float,
) -> str:
    """Build a bass pattern — root note one octave below chord."""
    root = scale[degree % len(scale)]
    bass_note = f"{root}2"

    # Bass: hit on 1 and 3 of each phrase group, rest on 2 and 4
    slots = []
    for i in range(n_phrases):
        if i % 2 == 0:
            slots.append(bass_note)
        else:
            slots.append("~")

    note_pattern = " ".join(slots)
    lines = [
        f'  note("{note_pattern}")',
        f'    .s("sawtooth").gain({round(gain * 0.7, 2)})',
        "    .lpf(800).attack(0.005).release(0.2)",
    ]
    return "\n".join(lines)


def _build_hihat_pattern(gain: float) -> str:
    """Build a hi-hat pattern using Strudel's built-in samples."""
    return (
        f'  s("hh*8")\n'
        f"    .gain({round(gain * 0.4, 2)})"
    )


def _build_kick_pattern(gain: float) -> str:
    """Build a kick pattern."""
    return (
        f'  s("bd bd ~ bd")\n'
        f"    .gain({round(gain * 0.8, 2)})"
    )


# ─── Main entry points ────────────────────────────────────────────────────

def to_strudel(
    layer2: Layer2Features,
    layer1: Optional[Layer1Features] = None,
    bpm_override: Optional[float] = None,
    key_override: Optional[str] = None,
    include_drums: bool = True,
) -> str:
    """Generate Strudel REPL code from earworm analysis.

    Args:
        layer2: Layer 2 structural analysis.
        layer1: Optional Layer 1 signal features (provides BPM and key).
        bpm_override: Override BPM.
        key_override: Override root key (e.g. "C minor").
        include_drums: Whether to add drum patterns.

    Returns:
        A string of JavaScript code for the Strudel REPL.
    """
    # Resolve BPM
    bpm = bpm_override or (layer1.temporal.bpm if layer1 else 120.0)

    # Resolve key
    if key_override:
        key_str = key_override
    elif layer1:
        key_str = layer1.harmonic.key
    else:
        key_str = "A minor"

    root, mode = _parse_key(key_str)
    scale = _scale_notes(root, mode)

    # Build chord map: section label → scale degree
    label_seq = layer2.recurrence.label_sequence
    n_labels = layer2.recurrence.n_distinct_labels or len(set(label_seq)) or 1
    degrees = _chord_progression_degrees(mode, n_labels)
    label_to_degree = {
        label: degrees[i % len(degrees)]
        for i, label in enumerate(sorted(set(label_seq)))
    }

    # Section timing
    seg = layer2.segmentation
    boundaries = seg.boundaries
    durations = seg.section_durations
    labels = seg.labels

    # Compute cycles per section (1 cycle = 1 bar at 4 beats)
    beats_per_cycle = 4.0
    secs_per_cycle = beats_per_cycle / (bpm / 60.0)

    # Energy data
    ec = layer2.energy_arc.energy_curve
    et = layer2.energy_arc.energy_timestamps

    # Phrase info
    typical_phrase = layer2.phrase.typical_phrase_beats or 4.0
    # Build arrange() sections
    arrange_entries = []
    for i, (label, dur) in enumerate(zip(labels, durations)):
        degree = label_to_degree.get(label, 0)
        n_cycles = max(1, round(dur / secs_per_cycle))
        sec_start = boundaries[i] if i < len(boundaries) else 0.0
        sec_end = sec_start + dur
        gain = _section_gain(ec, et, sec_start, sec_end)

        # Phrases in this section
        n_phrases = max(1, round(dur / (typical_phrase / (bpm / 60.0))))

        # Synth varies by section type for timbral contrast
        synth = "supersaw" if label % 2 == 0 else "sawtooth"

        pad = _build_section_pattern(scale, degree, typical_phrase, n_phrases, gain, synth)
        bass = _build_bass_pattern(scale, degree, n_phrases, gain)

        layers = [pad, bass]
        if include_drums:
            layers.append(_build_kick_pattern(gain))
            layers.append(_build_hihat_pattern(gain))

        stack_body = ",\n".join(layers)
        arrange_entries.append(f"  [{n_cycles}, stack(\n{stack_body}\n  )]")

    # Assemble the full Strudel code
    sections_str = ",\n".join(arrange_entries)

    # Cycles per second: bpm / 60 / beats_per_cycle
    cps = round(bpm / 60.0 / beats_per_cycle, 4)

    code_lines = [
        f"// Earworm structural response — {key_str}, {bpm} BPM",
        f"// Sections: {_label_sequence_str(label_seq)}",
        f"// Energy arc: {layer2.energy_arc.n_builds} builds, {layer2.energy_arc.n_drops} drops",
        "",
        f"setcps({cps})",
        "",
        "arrange(",
        sections_str,
        ")",
        "  .room(0.3).roomsize(4)",
        "  .delay(0.15).delaytime(0.375).delayfeedback(0.3)",
    ]

    return "\n".join(code_lines)


def to_strudel_generative(
    bpm: float = 120.0,
    key: str = "A minor",
    n_sections: int = 8,
    section_pattern: str = "",
    energy_preset: str = "arc",
    duration_seconds: float = 180.0,
    include_drums: bool = True,
) -> str:
    """Generate Strudel code from parameters — no source audio required.

    Builds a synthetic Layer2Features and passes it to to_strudel().
    """
    from earworm.compose.composer import _build_synthetic_layer2

    layer2 = _build_synthetic_layer2(
        bpm=bpm,
        key=key,
        n_sections=n_sections,
        section_pattern=section_pattern,
        energy_preset=energy_preset,
        duration_seconds=duration_seconds,
    )

    return to_strudel(
        layer2,
        bpm_override=bpm,
        key_override=key,
        include_drums=include_drums,
    )


# ─── Helpers ───────────────────────────────────────────────────────────────

def _parse_key(key_str: str) -> tuple[str, str]:
    """Parse 'C minor' or 'F# major' into (root, mode)."""
    parts = key_str.strip().rsplit(" ", 1)
    if len(parts) == 2:
        root, mode_raw = parts
        mode = "minor" if "minor" in mode_raw.lower() else "major"
    else:
        root = parts[0]
        mode = "major"
    root = root[0].upper() + root[1:]
    return root, mode


def _label_sequence_str(labels: list[int]) -> str:
    """Convert [0, 1, 0, 2] to 'A B A C'."""
    return " ".join(chr(65 + label) for label in labels)
