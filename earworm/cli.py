"""CLI entry point for earworm analysis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from earworm.pipeline import analyze_layer1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="earworm",
        description="Art's perceptual system for music comprehension",
    )
    subparsers = parser.add_subparsers(dest="command")

    # analyze command
    analyze = subparsers.add_parser("analyze", help="Run Layer 1 analysis on an audio file")
    analyze.add_argument("file", type=Path, help="Path to audio file (WAV, FLAC, MP3, etc.)")
    analyze.add_argument("--json", action="store_true", help="Output raw JSON")
    analyze.add_argument("-o", "--output", type=Path, help="Write JSON to file instead of stdout")

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

    print(f"Analyzing: {path}", file=sys.stderr)
    start = time.time()

    result = analyze_layer1(path)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s", file=sys.stderr)

    if args.json:
        output = result.model_dump_json(indent=2)
    else:
        output = _format_human(result)

    if args.output:
        args.output.write_text(output)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


def _format_human(result) -> str:
    """Format analysis results for human reading."""
    lines = []
    lines.append(f"{'=' * 60}")
    lines.append(f"  Earworm Layer 1 Analysis")
    lines.append(f"{'=' * 60}")
    lines.append(f"  File:     {result.file_path}")
    lines.append(f"  Duration: {result.duration_seconds:.1f}s")
    lines.append(f"  Format:   {result.sample_rate}Hz, {result.channels}ch")
    lines.append("")

    # Temporal
    t = result.temporal
    lines.append(f"  Rhythm")
    lines.append(f"    BPM:            {t.bpm:.1f} (confidence: {t.bpm_confidence:.2f})")
    lines.append(f"    Tempo stability: {t.tempo_stability:.2f}")
    lines.append(f"    Onset rate:     {t.onset_rate:.1f}/s")
    lines.append(f"    Beats detected: {len(t.beat_times)}")
    lines.append("")

    # Harmonic
    h = result.harmonic
    lines.append(f"  Tonality")
    lines.append(f"    Key:            {h.key} (confidence: {h.key_confidence:.2f})")
    lines.append(f"    Harmonic ratio: {h.harmonic_ratio:.2f}")
    lines.append("")

    # Spectral
    s = result.spectral
    lines.append(f"  Spectral")
    lines.append(f"    Brightness:     {s.spectral_centroid_mean:.0f} Hz")
    lines.append(f"    Bandwidth:      {s.spectral_bandwidth_mean:.0f} Hz")
    lines.append(f"    Rolloff:        {s.spectral_rolloff_mean:.0f} Hz")
    lines.append(f"    Flatness:       {s.spectral_flatness_mean:.4f} ({'noisy' if s.spectral_flatness_mean > 0.1 else 'tonal'})")
    lines.append("")

    # Loudness
    l = result.loudness
    lines.append(f"  Loudness")
    lines.append(f"    LUFS:           {l.lufs_integrated:.1f}")
    lines.append(f"    Peak:           {l.peak_db:.1f} dB")
    lines.append(f"    RMS:            {l.rms_db:.1f} dB")
    lines.append(f"    Crest factor:   {l.crest_factor_db:.1f} dB")
    lines.append(f"    Dynamic range:  {l.lufs_range:.1f} LU")
    lines.append("")

    # Stereo
    st = result.stereo
    lines.append(f"  Stereo")
    if st.is_stereo:
        lines.append(f"    Width:          {st.width_mean:.2f} (std: {st.width_std:.2f})")
        lines.append(f"    Correlation:    {st.correlation_mean:.2f} (min: {st.correlation_min:.2f})")
        lines.append(f"    M/S ratio:      {st.mid_side_ratio:.2f}")
        lines.append(f"    Balance:        {st.balance:+.3f}")
    else:
        lines.append(f"    Mono source")
    lines.append("")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
