"""CLI entry point for earworm analysis."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from earworm.pipeline import analyze_layer1, analyze_layer2


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="earworm",
        description="Art's perceptual system for music comprehension",
    )
    subparsers = parser.add_subparsers(dest="command")

    # analyze command
    analyze = subparsers.add_parser("analyze", help="Run analysis on an audio file")
    analyze.add_argument("file", type=Path, help="Path to audio file (WAV, FLAC, MP3, etc.)")
    analyze.add_argument("--json", action="store_true", help="Output raw JSON")
    analyze.add_argument("-o", "--output", type=Path, help="Write JSON to file instead of stdout")
    analyze.add_argument(
        "--layer",
        type=int,
        choices=[1, 2],
        default=1,
        help="Analysis layer: 1=signal features, 2=structural comprehension (default: 1)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "analyze":
        _cmd_analyze(args)


def _cmd_analyze(args: argparse.Namespace) -> None:
    path = args.file
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    layer = args.layer
    print(f"Analyzing (Layer {layer}): {path}", file=sys.stderr)
    start = time.time()

    if layer == 1:
        result = analyze_layer1(path)
    else:
        # Layer 2 runs Layer 1 first if needed for beat data
        layer1 = analyze_layer1(path)
        result = analyze_layer2(path, layer1=layer1)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s", file=sys.stderr)

    if args.json:
        output = result.model_dump_json(indent=2)
    else:
        if layer == 1:
            output = _format_human_l1(result)
        else:
            output = _format_human_l2(result)

    if args.output:
        args.output.write_text(output)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


def _format_human_l1(result) -> str:
    """Format Layer 1 analysis results for human reading."""
    lines = []
    lines.append(f"{'=' * 60}")
    lines.append("  Earworm Layer 1 Analysis")
    lines.append(f"{'=' * 60}")
    lines.append(f"  File:     {result.file_path}")
    lines.append(f"  Duration: {result.duration_seconds:.1f}s")
    lines.append(f"  Format:   {result.sample_rate}Hz, {result.channels}ch")
    lines.append("")

    # Temporal
    t = result.temporal
    lines.append("  Rhythm")
    lines.append(f"    BPM:            {t.bpm:.1f} (confidence: {t.bpm_confidence:.2f})")
    lines.append(f"    Tempo stability: {t.tempo_stability:.2f}")
    lines.append(f"    Onset rate:     {t.onset_rate:.1f}/s")
    lines.append(f"    Beats detected: {len(t.beat_times)}")
    lines.append("")

    # Harmonic
    h = result.harmonic
    lines.append("  Tonality")
    lines.append(f"    Key:            {h.key} (confidence: {h.key_confidence:.2f})")
    lines.append(f"    Harmonic ratio: {h.harmonic_ratio:.2f}")
    lines.append("")

    # Spectral
    s = result.spectral
    lines.append("  Spectral")
    lines.append(f"    Brightness:     {s.spectral_centroid_mean:.0f} Hz")
    lines.append(f"    Bandwidth:      {s.spectral_bandwidth_mean:.0f} Hz")
    lines.append(f"    Rolloff:        {s.spectral_rolloff_mean:.0f} Hz")
    lines.append(f"    Flatness:       {s.spectral_flatness_mean:.4f} ({'noisy' if s.spectral_flatness_mean > 0.1 else 'tonal'})")
    lines.append("")

    # Loudness
    loud = result.loudness
    lines.append("  Loudness")
    lines.append(f"    LUFS:           {loud.lufs_integrated:.1f}")
    lines.append(f"    Peak:           {loud.peak_db:.1f} dB")
    lines.append(f"    RMS:            {loud.rms_db:.1f} dB")
    lines.append(f"    Crest factor:   {loud.crest_factor_db:.1f} dB")
    lines.append(f"    Dynamic range:  {loud.lufs_range:.1f} LU")
    lines.append("")

    # Stereo
    st = result.stereo
    lines.append("  Stereo")
    if st.is_stereo:
        lines.append(f"    Width:          {st.width_mean:.2f} (std: {st.width_std:.2f})")
        lines.append(f"    Correlation:    {st.correlation_mean:.2f} (min: {st.correlation_min:.2f})")
        lines.append(f"    M/S ratio:      {st.mid_side_ratio:.2f}")
        lines.append(f"    Balance:        {st.balance:+.3f}")
    else:
        lines.append("    Mono source")
    lines.append("")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def _format_human_l2(result) -> str:
    """Format Layer 2 analysis results for human reading."""
    lines = []
    lines.append(f"{'=' * 60}")
    lines.append("  Earworm Layer 2 Analysis — Structural Comprehension")
    lines.append(f"{'=' * 60}")
    lines.append(f"  File:     {result.file_path}")
    lines.append(f"  Duration: {result.duration_seconds:.1f}s")
    lines.append("")

    # Segmentation
    seg = result.segmentation
    lines.append("  Segmentation")
    lines.append(f"    Sections:       {seg.n_sections}")
    label_seq = "".join(chr(65 + lb) for lb in seg.labels)  # 0→A, 1→B, etc.
    lines.append(f"    Structure:      {label_seq}")
    for i, (dur, label) in enumerate(zip(seg.section_durations, seg.labels)):
        start = seg.boundaries[i]
        lines.append(f"    [{start:6.1f}s] Section {chr(65 + label)} ({dur:.1f}s)")
    lines.append("")

    # Recurrence
    rec = result.recurrence
    lines.append("  Recurrence")
    lines.append(f"    Distinct types: {rec.n_distinct_labels}")
    lines.append(f"    Repetition:     {rec.repetition_ratio:.0%} of track is repeated material")
    lines.append("")

    # Energy arc
    ea = result.energy_arc
    lines.append("  Energy Arc")
    lines.append(f"    Climax at:      {ea.climax_time:.1f}s ({ea.climax_position:.0%} through)")
    lines.append(f"    Builds:         {ea.n_builds}")
    lines.append(f"    Drops:          {ea.n_drops}")
    lines.append(f"    Dynamic spread: {ea.dynamic_spread:.3f}")
    lines.append("")

    # Phrase structure
    ph = result.phrase
    lines.append("  Phrase Structure")
    lines.append(f"    Phrases:        {ph.n_phrases}")
    lines.append(f"    Typical length: {ph.typical_phrase_beats:.0f} beats")
    lines.append(f"    Regularity:     {ph.regularity:.0%}")
    if ph.irregular_phrases:
        lines.append(f"    Irregular:      phrases {ph.irregular_phrases}")
    lines.append("")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
