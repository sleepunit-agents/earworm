"""Earworm compose — structural response pipeline.

Maps Layer 2 analysis (and optionally Layer 1) to a pytheory Score and
generates a WAV composition that responds to the source track's structural DNA.

The composition does not imitate the source — it responds:
- Same section count and label structure (which parts repeat, which are new)
- Same energy arc (where dynamics peak, where they drop)
- Same phrase rhythm (chord changes follow phrase boundaries)
- Different sonic vocabulary (synthetic pads and bass, not the source timbre)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel
from pytheory.play import Synth
from pytheory.rhythm import Score
from pytheory.scales import Key
from pytheory.tones import Tone

from earworm.compose.renderer import render_wav
from earworm.models import Layer1Features, Layer2Features


class ComposeManifest(BaseModel):
    """Record of musical choices made during composition."""

    source_file: str
    output_file: str
    bpm: float
    key: str
    mode: str
    n_sections: int
    label_sequence: list[int]
    chord_map: dict[int, str]  # label → chord name
    n_phrases: int
    duration_seconds: float
    style: str


def compose(
    layer2: Layer2Features,
    output: Path,
    layer1: Optional[Layer1Features] = None,
    style: str = "edm",
    duration_override: Optional[float] = None,
    bpm_override: Optional[float] = None,
    key_override: Optional[str] = None,
) -> ComposeManifest:
    """Generate a WAV response to a track's structural analysis.

    Args:
        layer2: Layer 2 structural analysis of the source track.
        output: Destination WAV file path.
        layer1: Optional Layer 1 signal features (provides BPM and key).
        style: Instrument palette. Currently only "edm" is supported.
        duration_override: Override composition length in seconds.
        bpm_override: Override BPM.
        key_override: Override root key (e.g. "C minor", "F# major").

    Returns:
        ComposeManifest describing the musical choices made.
    """
    # --- 1. Resolve BPM and key ---
    bpm = bpm_override or (layer1.temporal.bpm if layer1 else 120.0)
    bpm = float(bpm)

    if key_override:
        key_str = key_override
    elif layer1:
        key_str = layer1.harmonic.key  # e.g. "C minor" or "F# major"
    else:
        key_str = "A minor"

    root, mode = _parse_key(key_str)
    key = Key(root, mode)

    # --- 2. Build chord map: label → chord degree ---
    label_seq = layer2.recurrence.label_sequence
    n_labels = layer2.recurrence.n_distinct_labels or len(set(label_seq)) or 1
    chord_degrees = _chord_progression(mode, n_labels)
    chord_map = {label: chord_degrees[i % len(chord_degrees)] for i, label in enumerate(sorted(set(label_seq)))}

    # label → chord name for manifest
    chord_name_map = {label: str(key.triad(deg)) for label, deg in chord_map.items()}

    # --- 3. Map segmentation to chord sequence ---
    # One chord per *section* (boundaries from segmentation)
    seg = layer2.segmentation
    boundaries = seg.boundaries  # section start times in seconds
    seg_labels = seg.labels      # one label per section

    # --- 4. Map phrase boundaries to note events ---
    phrase_bounds = layer2.phrase.phrase_boundaries  # timestamps in seconds
    if not phrase_bounds:
        # Fallback: use section boundaries
        phrase_bounds = boundaries
    if not phrase_bounds:
        phrase_bounds = [0.0, layer2.duration_seconds]

    # Normalize: ensure we start at 0 and end at total_dur
    total_dur = duration_override or layer2.duration_seconds
    if phrase_bounds[0] > 0.1:
        phrase_bounds = [0.0] + phrase_bounds
    # Truncate to total_dur and append it as the final boundary
    phrase_bounds = [t for t in phrase_bounds if t < total_dur]
    phrase_bounds.append(total_dur)

    # --- 5. Map energy curve to velocity ---
    energy_curve = np.array(layer2.energy_arc.energy_curve)
    energy_times = np.array(layer2.energy_arc.energy_timestamps)

    def energy_at(t: float) -> float:
        if len(energy_curve) == 0:
            return 0.5
        idx = int(np.searchsorted(energy_times, t, side="right")) - 1
        idx = max(0, min(idx, len(energy_curve) - 1))
        return float(energy_curve[idx])

    def velocity_at(t: float) -> int:
        e = energy_at(t)
        return max(50, min(127, int(50 + e * 77)))

    # --- 6. Determine the chord at each phrase boundary ---
    # Match each phrase to a section label
    def section_label_at(t: float) -> int:
        if not boundaries or not seg_labels:
            return 0
        idx = int(np.searchsorted(boundaries, t, side="right")) - 1
        idx = max(0, min(idx, len(seg_labels) - 1))
        return seg_labels[idx]

    # --- 7. Build Score ---
    score = Score(bpm=int(bpm))

    bass_part = score.part("bass", synth=Synth.SAW, volume=0.45)
    pad_part = score.part("pad", synth=Synth.SUPERSAW, volume=0.35)

    beats_per_sec = bpm / 60.0

    for i, t_start in enumerate(phrase_bounds[:-1]):
        t_end = phrase_bounds[i + 1]
        dur_secs = t_end - t_start
        dur_beats = max(0.25, dur_secs * beats_per_sec)

        vel = velocity_at(t_start)
        label = section_label_at(t_start)
        degree = chord_map.get(label, 0)
        chord = key.triad(degree)

        # Pad: full chord in mid range (octave 4)
        pad_part.add(chord, dur_beats, velocity=vel)

        # Bass: root note two octaves below chord root
        root_tone = chord.tones[0]  # root is first tone e.g. 'C4'
        bass_tone = _transpose_octaves(root_tone, -2)
        bass_part.add(bass_tone, dur_beats, velocity=min(vel, 90))

    # --- 8. Render ---
    output.parent.mkdir(parents=True, exist_ok=True)
    render_wav(score, output)

    return ComposeManifest(
        source_file=layer2.file_path,
        output_file=str(output),
        bpm=bpm,
        key=root,
        mode=mode,
        n_sections=layer2.segmentation.n_sections,
        label_sequence=label_seq,
        chord_map=chord_name_map,
        n_phrases=len(phrase_bounds) - 1,
        duration_seconds=total_dur,
        style=style,
    )


def compose_from_json(
    analysis_path: Path,
    output: Path,
    **kwargs,
) -> ComposeManifest:
    """Load analysis JSON and compose. Accepts L1, L2, or combined JSON."""
    with open(analysis_path) as f:
        data = json.load(f)

    layer2 = Layer2Features.model_validate(data)

    layer1 = None
    if "temporal" in data and "harmonic" in data:
        try:
            layer1 = Layer1Features.model_validate(data)
        except Exception:
            pass

    return compose(layer2, output, layer1=layer1, **kwargs)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _parse_key(key_str: str) -> tuple[str, str]:
    """Parse 'C minor' or 'F# major' into (root, mode)."""
    parts = key_str.strip().rsplit(" ", 1)
    if len(parts) == 2:
        root, mode_raw = parts
        mode = "minor" if "minor" in mode_raw.lower() else "major"
    else:
        root = parts[0]
        mode = "major"
    # Capitalize root properly (e.g. 'f#' → 'F#')
    root = root[0].upper() + root[1:]
    return root, mode


def _chord_progression(mode: str, n_chords: int) -> list[int]:
    """Return a list of scale-degree indices for a chord progression.

    The progression is mode-appropriate and long enough for *n_chords* sections.
    Degrees repeat cyclically if n_chords > len(base_progression).
    """
    if mode == "minor":
        # i – III – VII – iv – VI – III – VII – i  (dark, driving — good for EDM)
        base = [0, 2, 6, 3, 5, 2, 6, 0]
    else:
        # I – IV – V – vi – I – IV – vi – V  (bright, anthemic)
        base = [0, 3, 4, 5, 0, 3, 5, 4]
    # Return exactly n_chords degrees (cycling if needed)
    return [base[i % len(base)] for i in range(n_chords)]


def _transpose_octaves(tone: Tone, octaves: int) -> Tone:
    """Return a new Tone shifted by *octaves* octaves."""
    current = str(tone)  # e.g. 'C4'
    # Extract letter + accidentals + octave number
    import re
    m = re.match(r"([A-G][#b]?)(\d+)", current)
    if not m:
        return tone
    name, oct_str = m.group(1), m.group(2)
    new_oct = max(0, int(oct_str) + octaves)
    return Tone.from_string(f"{name}{new_oct}")
