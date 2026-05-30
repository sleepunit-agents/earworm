"""CLI entry point for earworm analysis."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from earworm.pipeline import analyze_layer1, analyze_layer2, analyze_layer3
from earworm.compose import compose, compose_generative
from earworm.compose.composer import compose_from_json
from earworm.compose.strudel import to_strudel, to_strudel_generative


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
        choices=[1, 2, 3],
        default=1,
        help="Analysis layer: 1=signal, 2=structural, 3=quality (default: 1)",
    )

    # voice command
    voice = subparsers.add_parser("voice", help="Interpret a track — natural language opinion")
    voice.add_argument("file", type=Path, help="Path to audio file (WAV, FLAC, MP3, etc.)")
    voice.add_argument("--json", action="store_true", help="Output raw JSON")
    voice.add_argument("-o", "--output", type=Path, help="Write output to file")
    voice.add_argument(
        "--mode",
        choices=["quick", "deep"],
        default="quick",
        help="Interpretation depth: quick (2-3 sentences) or deep (full walkthrough)",
    )
    voice.add_argument(
        "--provider",
        choices=["anthropic", "ollama"],
        help="LLM provider (default: EARWORM_LLM_PROVIDER env or ollama)",
    )
    voice.add_argument(
        "--model",
        help="Override the LLM model name",
    )
    voice.add_argument(
        "--find-samples",
        action="store_true",
        help="Enrich output with related samples from samplebank",
    )
    voice.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Max related samples to include (default: 10)",
    )

    # bridge command
    bridge_parser = subparsers.add_parser(
        "bridge", help="Find related samples in samplebank via CLAP"
    )
    bridge_parser.add_argument(
        "file", type=Path, help="Path to audio file to analyze and search"
    )
    bridge_parser.add_argument("--json", action="store_true", help="Output raw JSON")
    bridge_parser.add_argument(
        "-o", "--output", type=Path, help="Write output to file"
    )
    bridge_parser.add_argument(
        "--limit", type=int, default=20, help="Max results to return (default: 20)"
    )
    bridge_parser.add_argument(
        "--mode",
        choices=["text", "audio", "combined"],
        default="text",
        help="Search mode: text (Voice→samples), audio (track→samples), combined (default: text)",
    )
    bridge_parser.add_argument(
        "--voice-mode",
        choices=["quick", "deep"],
        default="quick",
        help="Voice interpretation depth for text search (default: quick)",
    )
    bridge_parser.add_argument(
        "--samplebank-url", help="Samplebank API URL (default: env or localhost:8000)"
    )
    bridge_parser.add_argument(
        "--clap-url", help="CLAP service URL (default: env or localhost:8100)"
    )
    bridge_parser.add_argument(
        "--qdrant-url", help="Qdrant URL (default: env or localhost:6333)"
    )
    bridge_parser.add_argument(
        "--status", action="store_true", help="Check samplebank bridge status and exit"
    )

    # compose command
    compose_parser = subparsers.add_parser(
        "compose", help="Generate a structural response composition from analysis"
    )
    compose_parser.add_argument(
        "file", type=Path, nargs="?",
        help="Audio file to analyze and compose from (omit with --generative)"
    )
    compose_parser.add_argument(
        "-o", "--output", type=Path, help="Output WAV file (default: <file>_response.wav or output.wav)"
    )
    compose_parser.add_argument(
        "--from-json", type=Path, dest="from_json",
        help="Skip analysis — use existing JSON analysis file"
    )
    compose_parser.add_argument(
        "--generative", action="store_true",
        help="Pure generative mode — no source audio required"
    )
    compose_parser.add_argument(
        "--sections", type=int, dest="sections", default=8,
        help="Number of sections in generative mode (default: 8)"
    )
    compose_parser.add_argument(
        "--pattern", dest="pattern", default="",
        help="Section letter pattern for generative mode, e.g. AABABCABC"
    )
    compose_parser.add_argument(
        "--energy",
        choices=["arc", "peak-drop", "flat", "pulse"],
        default="arc",
        help="Energy arc preset for generative mode (default: arc)"
    )
    compose_parser.add_argument(
        "--style",
        choices=["edm"],
        default="edm",
        help="Instrument palette (default: edm)",
    )
    compose_parser.add_argument(
        "--bpm", type=float, help="BPM (generative default: 120)"
    )
    compose_parser.add_argument(
        "--key", help="Key signature (e.g. 'C minor', 'F# major')"
    )
    compose_parser.add_argument(
        "--duration", type=float, dest="duration",
        help="Composition length in seconds (generative default: 180)"
    )
    compose_parser.add_argument(
        "--json", action="store_true", help="Output manifest JSON to stdout"
    )

    # generate — alias for compose --generative
    generate_parser = subparsers.add_parser(
        "generate", help="Alias for 'compose --generative' — original composition without source audio"
    )
    generate_parser.add_argument(
        "-o", "--output", type=Path, default=Path("output.wav"),
        help="Output WAV file (default: output.wav)"
    )
    generate_parser.add_argument(
        "--sections", type=int, dest="sections", default=8,
        help="Number of sections (default: 8)"
    )
    generate_parser.add_argument(
        "--pattern", dest="pattern", default="",
        help="Section letter pattern, e.g. AABABCABC"
    )
    generate_parser.add_argument(
        "--energy",
        choices=["arc", "peak-drop", "flat", "pulse"],
        default="arc",
        help="Energy arc preset (default: arc)"
    )
    generate_parser.add_argument(
        "--style",
        choices=["edm"],
        default="edm",
        help="Instrument palette (default: edm)",
    )
    generate_parser.add_argument(
        "--bpm", type=float, default=120.0, help="BPM (default: 120)"
    )
    generate_parser.add_argument(
        "--key", default="A minor", help="Key signature (default: 'A minor')"
    )
    generate_parser.add_argument(
        "--duration", type=float, dest="duration", default=180.0,
        help="Composition length in seconds (default: 180)"
    )
    generate_parser.add_argument(
        "--json", action="store_true", help="Output manifest JSON to stdout"
    )

    # strudel command — generate Strudel REPL code
    strudel_parser = subparsers.add_parser(
        "strudel", help="Generate Strudel REPL code from analysis or parameters"
    )
    strudel_parser.add_argument(
        "file", type=Path, nargs="?",
        help="Audio file to analyze (omit with --generative)"
    )
    strudel_parser.add_argument(
        "--from-json", type=Path, dest="from_json",
        help="Use existing analysis JSON instead of analyzing audio"
    )
    strudel_parser.add_argument(
        "--generative", action="store_true",
        help="Pure generative mode — no source audio required"
    )
    strudel_parser.add_argument(
        "--sections", type=int, default=8,
        help="Number of sections in generative mode (default: 8)"
    )
    strudel_parser.add_argument(
        "--pattern", default="",
        help="Section letter pattern, e.g. AABABCABC"
    )
    strudel_parser.add_argument(
        "--energy",
        choices=["arc", "peak-drop", "flat", "pulse"],
        default="arc",
        help="Energy arc preset for generative mode (default: arc)"
    )
    strudel_parser.add_argument(
        "--bpm", type=float, help="BPM override"
    )
    strudel_parser.add_argument(
        "--key", help="Key signature (e.g. 'C minor', 'F# major')"
    )
    strudel_parser.add_argument(
        "--duration", type=float, default=180.0,
        help="Duration in seconds for generative mode (default: 180)"
    )
    strudel_parser.add_argument(
        "--no-drums", action="store_true",
        help="Omit drum patterns"
    )
    strudel_parser.add_argument(
        "-o", "--output", type=Path,
        help="Write Strudel code to file instead of stdout"
    )

    # calibrate command group
    calibrate = subparsers.add_parser("calibrate", help="Phase 3 calibration tools")
    cal_sub = calibrate.add_subparsers(dest="cal_command")

    cal_init = cal_sub.add_parser("init", help="Initialize corpus with seed tracks")
    cal_init.add_argument(
        "--corpus-dir", type=Path, help="Corpus directory (default: ./calibration)"
    )

    cal_list = cal_sub.add_parser("list", help="List tracks in the calibration corpus")
    cal_list.add_argument(
        "--corpus-dir", type=Path, help="Corpus directory (default: ./calibration)"
    )

    cal_add = cal_sub.add_parser("add", help="Add an audio file to a corpus track")
    cal_add.add_argument("track_id", help="Track ID (e.g. talking-heads-born-under-punches)")
    cal_add.add_argument("file", type=Path, help="Path to audio file")
    cal_add.add_argument(
        "--corpus-dir", type=Path, help="Corpus directory (default: ./calibration)"
    )

    cal_run = cal_sub.add_parser("run", help="Run pipeline on pending corpus tracks")
    cal_run.add_argument(
        "--corpus-dir", type=Path, help="Corpus directory (default: ./calibration)"
    )
    cal_run.add_argument("--voice", action="store_true", help="Include Voice interpretation")
    cal_run.add_argument(
        "--track", help="Run only this track ID (default: all pending)"
    )

    cal_check = cal_sub.add_parser("check", help="Run alignment checks on analyzed tracks")
    cal_check.add_argument(
        "--corpus-dir", type=Path, help="Corpus directory (default: ./calibration)"
    )

    cal_report = cal_sub.add_parser("report", help="Generate calibration report")
    cal_report.add_argument(
        "--corpus-dir", type=Path, help="Corpus directory (default: ./calibration)"
    )
    cal_report.add_argument("--json", action="store_true", help="Output raw JSON")

    # journal subcommands
    cal_journal = cal_sub.add_parser("journal", help="View or manage the taste journal")
    journal_sub = cal_journal.add_subparsers(dest="journal_command")

    journal_show = journal_sub.add_parser("show", help="Show journal summary")
    journal_show.add_argument(
        "--corpus-dir", type=Path, help="Corpus directory (default: ./calibration)"
    )
    journal_show.add_argument("--json", action="store_true", help="Output raw JSON")

    journal_observe = journal_sub.add_parser(
        "observe", help="Record an observation about a track"
    )
    journal_observe.add_argument("track_id", help="Track ID to observe")
    journal_observe.add_argument("--noticed", required=True, help="What you noticed")
    journal_observe.add_argument("--stood-out", required=True, help="What stood out")
    journal_observe.add_argument("--missed", default="", help="What you missed (if anything)")
    journal_observe.add_argument("--reaction", default="", help="Raw gut reaction")
    journal_observe.add_argument(
        "--samples", nargs="*", default=[],
        help="Sample references as 'id:filename:score:why' (e.g. '42:kick_808.wav:0.9:matches the low end')",
    )
    journal_observe.add_argument(
        "--corpus-dir", type=Path, help="Corpus directory (default: ./calibration)"
    )

    journal_pattern = journal_sub.add_parser(
        "pattern", help="Record or update a taste pattern"
    )
    journal_pattern.add_argument("name", help="Pattern name")
    journal_pattern.add_argument("--description", required=True, help="Pattern description")
    journal_pattern.add_argument(
        "--tracks", nargs="*", default=[], help="Supporting track IDs"
    )
    journal_pattern.add_argument(
        "--confidence", type=float, default=0.5, help="Pattern confidence (0-1)"
    )
    journal_pattern.add_argument(
        "--corpus-dir", type=Path, help="Corpus directory (default: ./calibration)"
    )

    journal_diverge = journal_sub.add_parser(
        "diverge", help="Classify a divergence"
    )
    journal_diverge.add_argument("track_id", help="Track ID")
    journal_diverge.add_argument("dimension", help="Alignment dimension")
    journal_diverge.add_argument("--pipeline", required=True, help="What the pipeline perceived")
    journal_diverge.add_argument("--human", required=True, help="Human consensus")
    journal_diverge.add_argument(
        "--classify",
        required=True,
        choices=["gap", "taste", "unclear"],
        help="Classification: gap (pipeline limitation), taste (genuine difference), unclear",
    )
    journal_diverge.add_argument("--reasoning", default="", help="Why this classification")
    journal_diverge.add_argument(
        "--corpus-dir", type=Path, help="Corpus directory (default: ./calibration)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "voice":
        _cmd_voice(args)
    elif args.command == "bridge":
        _cmd_bridge(args)
    elif args.command in ("compose", "generate"):
        _cmd_compose(args)
    elif args.command == "strudel":
        _cmd_strudel(args)
    elif args.command == "calibrate":
        _cmd_calibrate(args)


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
    elif layer == 2:
        layer1 = analyze_layer1(path)
        result = analyze_layer2(path, layer1=layer1)
    else:
        layer1 = analyze_layer1(path)
        layer2 = analyze_layer2(path, layer1=layer1)
        result = analyze_layer3(path, layer1=layer1, layer2=layer2)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s", file=sys.stderr)

    if args.json:
        output = result.model_dump_json(indent=2)
    else:
        if layer == 1:
            output = _format_human_l1(result)
        elif layer == 2:
            output = _format_human_l2(result)
        else:
            output = _format_human_l3(result)

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


def _format_human_l3(result) -> str:
    """Format Layer 3 analysis results for human reading."""
    lines = []
    lines.append(f"{'=' * 60}")
    lines.append("  Earworm Layer 3 Analysis — Quality Assessment")
    lines.append(f"{'=' * 60}")
    lines.append(f"  File:     {result.file_path}")
    lines.append(f"  Duration: {result.duration_seconds:.1f}s")
    lines.append("")

    # Technical
    tech = result.technical
    lines.append("  Technical Quality")
    lines.append(f"    Clipping:       {tech.clipping_ratio:.4%} ({tech.clipping_regions} regions)")
    lines.append(f"    DC offset:      {tech.dc_offset:.6f} ({'WARNING' if tech.has_dc_offset else 'OK'})")
    lines.append(f"    Noise floor:    {tech.noise_floor_db:.1f} dB")
    lines.append(f"    Freq balance:   {tech.frequency_balance_score:.0%}")
    lines.append("")

    # Mix
    mix = result.mix
    lines.append("  Mix Quality")
    lines.append(f"    Low (<200Hz):   {mix.low_ratio:.1%}")
    lines.append(f"    Mid (200-4k):   {mix.mid_ratio:.1%}")
    lines.append(f"    High (>4kHz):   {mix.high_ratio:.1%}")
    lines.append(f"    Balance:        {mix.spectral_balance_score:.0%}")
    lines.append(f"    Stereo width:   {mix.stereo_width_score:.0%}")
    lines.append(f"    Low clarity:    {mix.low_end_clarity:.0%}")
    lines.append(f"    High clarity:   {mix.high_frequency_clarity:.0%}")
    lines.append("")

    # Mastering
    mast = result.mastering
    lines.append("  Mastering Quality")
    lines.append(f"    LUFS:           {mast.lufs_integrated:.1f} (target: -14, deviation: {mast.lufs_deviation_from_target:+.1f})")
    lines.append(f"    Dynamic range:  {mast.dynamic_range_score:.0%}")
    lines.append(f"    Crest factor:   {mast.crest_factor_db:.1f} dB")
    lines.append(f"    Consistency:    {mast.loudness_consistency:.0%}")
    lines.append(f"    Limiter:        {mast.limiter_artifact_score:.0%} artifacts")
    lines.append("")

    # Composition
    comp = result.composition
    lines.append("  Composition Quality")
    lines.append(f"    Harm. vocab:    {comp.harmonic_vocabulary} chord classes")
    lines.append(f"    Chord changes:  {comp.chord_change_rate:.1f}/s")
    lines.append(f"    Rhythmic var:   {comp.rhythmic_variation:.0%}")
    lines.append(f"    Melodic range:  {comp.melodic_range_semitones:.1f} semitones")
    lines.append(f"    Struct variety: {comp.structural_variety:.0%}")
    lines.append("")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def _cmd_compose(args: argparse.Namespace) -> None:
    import json as _json

    # Detect generative mode: either --generative flag, or `generate` alias, or no file given
    # Note: --from-json takes priority over the "no file" heuristic
    is_generative = (
        getattr(args, "generative", False)
        or args.command == "generate"
        or (getattr(args, "file", None) is None and not getattr(args, "from_json", None))
    )

    if is_generative:
        output = args.output or Path("output.wav")
        bpm = args.bpm or 120.0
        key = args.key or "A minor"
        duration = args.duration or 180.0
        n_sections = getattr(args, "sections", 8)
        pattern = getattr(args, "pattern", "")
        energy = getattr(args, "energy", "arc")

        print(f"Generating (generative mode): BPM={bpm:.0f} Key={key} Energy={energy}", file=sys.stderr)
        print(f"Sections={n_sections}{f' pattern={pattern}' if pattern else ''}  Duration={duration:.0f}s", file=sys.stderr)
        print(f"Output: {output}", file=sys.stderr)

        manifest = compose_generative(
            output=output,
            bpm=bpm,
            key=key,
            n_sections=n_sections,
            section_pattern=pattern,
            energy_preset=energy,
            duration_seconds=duration,
            style=args.style,
        )
    elif args.from_json:
        output = args.output or Path("output.wav")
        # Use existing analysis JSON
        if not args.from_json.exists():
            print(f"Error: analysis file not found: {args.from_json}", file=sys.stderr)
            sys.exit(1)
        print(f"Composing from: {args.from_json}", file=sys.stderr)
        print(f"Output: {output}", file=sys.stderr)
        manifest = compose_from_json(
            args.from_json,
            output,
            style=args.style,
            bpm_override=args.bpm,
            key_override=args.key,
            duration_override=args.duration,
        )
    else:
        file_path = args.file
        if file_path is None:
            print("Error: provide a file argument or use --generative", file=sys.stderr)
            sys.exit(1)

        # Determine output path
        output = args.output
        if output is None:
            output = file_path.parent / (file_path.stem + "_response.wav")

        # Run full analysis first
        if not file_path.exists():
            print(f"Error: file not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Analyzing (Layer 1+2): {file_path}", file=sys.stderr)
        start = time.time()
        layer1 = analyze_layer1(file_path)
        layer2 = analyze_layer2(file_path, layer1=layer1)
        elapsed = time.time() - start
        print(f"Analysis complete in {elapsed:.1f}s", file=sys.stderr)
        print(f"Composing → {output}", file=sys.stderr)
        manifest = compose(
            layer2,
            output,
            layer1=layer1,
            style=args.style,
            bpm_override=args.bpm,
            key_override=args.key,
            duration_override=args.duration,
        )

    if args.json:
        print(_json.dumps(manifest.model_dump(), indent=2))
    else:
        print(f"Written: {manifest.output_file}")
        print(f"BPM: {manifest.bpm:.0f}  Key: {manifest.key} {manifest.mode}")
        print(f"Sections: {manifest.n_sections}  Phrases: {manifest.n_phrases}")
        print(f"Chord map: {manifest.chord_map}")
        print(f"Duration: {manifest.duration_seconds:.1f}s")


def _cmd_strudel(args: argparse.Namespace) -> None:
    import json as _json

    is_generative = (
        getattr(args, "generative", False)
        or getattr(args, "file", None) is None
    )
    include_drums = not getattr(args, "no_drums", False)

    if is_generative:
        bpm = args.bpm or 120.0
        key = args.key or "A minor"
        print(f"Generating Strudel (generative): BPM={bpm:.0f} Key={key}", file=sys.stderr)

        code = to_strudel_generative(
            bpm=bpm,
            key=key,
            n_sections=getattr(args, "sections", 8),
            section_pattern=getattr(args, "pattern", ""),
            energy_preset=getattr(args, "energy", "arc"),
            duration_seconds=args.duration,
            include_drums=include_drums,
        )
    elif args.from_json:
        if not args.from_json.exists():
            print(f"Error: analysis file not found: {args.from_json}", file=sys.stderr)
            sys.exit(1)
        print(f"Generating Strudel from: {args.from_json}", file=sys.stderr)

        with open(args.from_json) as f:
            data = _json.load(f)

        from earworm.models import Layer1Features, Layer2Features
        layer2 = Layer2Features.model_validate(data)
        layer1 = None
        if "temporal" in data and "harmonic" in data:
            try:
                layer1 = Layer1Features.model_validate(data)
            except Exception:
                pass

        code = to_strudel(
            layer2, layer1=layer1,
            bpm_override=args.bpm, key_override=args.key,
            include_drums=include_drums,
        )
    else:
        file_path = args.file
        if not file_path.exists():
            print(f"Error: file not found: {file_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Analyzing (Layer 1+2): {file_path}", file=sys.stderr)
        start = time.time()
        layer1 = analyze_layer1(file_path)
        layer2 = analyze_layer2(file_path, layer1=layer1)
        elapsed = time.time() - start
        print(f"Analysis complete in {elapsed:.1f}s", file=sys.stderr)

        code = to_strudel(
            layer2, layer1=layer1,
            bpm_override=args.bpm, key_override=args.key,
            include_drums=include_drums,
        )

    if args.output:
        args.output.write_text(code)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(code)


def _cmd_voice(args: argparse.Namespace) -> None:
    from earworm.voice import interpret_from_file
    from earworm.voice.provider import get_provider

    path = args.file
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    # Build provider kwargs
    provider_kwargs = {}
    if args.model:
        provider_kwargs["model"] = args.model

    provider = get_provider(provider_name=args.provider, **provider_kwargs)

    print(f"Interpreting ({args.mode}): {path}", file=sys.stderr)
    print(f"Provider: {type(provider).__name__}", file=sys.stderr)
    start = time.time()

    result = interpret_from_file(path, mode=args.mode, provider=provider)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s", file=sys.stderr)

    if args.find_samples:
        from earworm.bridge.enrich import enrich_voice_with_samples

        print("Finding related samples...", file=sys.stderr)
        start = time.time()
        result = enrich_voice_with_samples(
            result, audio_path=str(path), limit=args.sample_limit, mode="text"
        )
        elapsed = time.time() - start
        n = len(result.related_samples) if result.related_samples else 0
        print(f"Found {n} related samples in {elapsed:.1f}s", file=sys.stderr)

    if args.json:
        output = result.model_dump_json(indent=2)
    else:
        output = _format_human_voice(result)

    if args.output:
        args.output.write_text(output)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


def _format_human_voice(result) -> str:
    """Format Voice interpretation for human reading."""
    lines = []
    lines.append(f"{'=' * 60}")
    mode_label = "Quick Take" if result.mode == "quick" else "Deep Listen"
    lines.append(f"  Earworm Voice — {mode_label}")
    lines.append(f"{'=' * 60}")
    lines.append(f"  File: {result.file_path}")
    lines.append("")

    lines.append("  Description")
    for line in result.description.split("\n"):
        lines.append(f"    {line}")
    lines.append("")

    lines.append("  Opinion")
    for line in result.opinion.split("\n"):
        lines.append(f"    {line}")
    lines.append("")

    if result.tags:
        lines.append(f"  Tags: {', '.join(result.tags)}")
        lines.append("")

    if result.comparisons:
        lines.append("  Reminds me of")
        for comp in result.comparisons:
            lines.append(f"    • {comp}")
        lines.append("")

    if result.highlights:
        lines.append("  Highlights")
        for h in result.highlights:
            lines.append(f"    + {h}")
        lines.append("")

    if result.concerns:
        lines.append("  Concerns")
        for c in result.concerns:
            lines.append(f"    - {c}")
        lines.append("")

    if result.section_notes:
        lines.append("  Section Notes")
        for line in result.section_notes.split("\n"):
            lines.append(f"    {line}")
        lines.append("")

    if result.related_samples:
        lines.append("  Related Samples")
        for i, s in enumerate(result.related_samples, 1):
            lines.append(f"    {i:3d}. [{s.score:.3f}] {s.filename}")
            lines.append(f"         {s.path}")
        lines.append("")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def _cmd_bridge(args: argparse.Namespace) -> None:
    from earworm.bridge import SamplebankBridge

    bridge = SamplebankBridge(
        samplebank_url=args.samplebank_url,
        clap_url=args.clap_url,
        qdrant_url=args.qdrant_url,
    )

    if args.status:
        status = bridge.check_status()
        if args.json:
            print(status.model_dump_json(indent=2))
        else:
            reachable = "yes" if status.samplebank_reachable else "NO"
            print(f"Samplebank reachable: {reachable}")
            print(f"Indexed: {status.indexed_samples}/{status.total_samples} ({status.coverage_pct:.1f}%)")
        return

    path = args.file
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    mode = args.mode
    limit = args.limit

    if mode in ("text", "combined"):
        from earworm.voice import interpret_from_file
        from earworm.voice.provider import get_provider

        provider = get_provider()
        print(f"Running Voice interpretation ({args.voice_mode}): {path}", file=sys.stderr)
        start = time.time()
        voice_result = interpret_from_file(path, mode=args.voice_mode, provider=provider)
        elapsed = time.time() - start
        print(f"Voice done in {elapsed:.1f}s", file=sys.stderr)
    else:
        voice_result = None

    print(f"Searching samplebank ({mode} mode)...", file=sys.stderr)
    start = time.time()

    if mode == "text":
        results = bridge.search_by_voice(voice_result, limit=limit)
    elif mode == "audio":
        results = bridge.search_by_audio(str(path), limit=limit)
    else:
        results = bridge.search_combined(voice_result, str(path), limit=limit)

    elapsed = time.time() - start
    print(f"Found {len(results)} matches in {elapsed:.1f}s", file=sys.stderr)

    if args.json:
        import json as json_mod

        output = json_mod.dumps(
            {
                "mode": mode,
                "query_file": str(path),
                "results": [r.model_dump() for r in results],
            },
            indent=2,
        )
    else:
        output = _format_bridge_results(results, mode)

    if args.output:
        args.output.write_text(output)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


def _format_bridge_results(results: list, mode: str) -> str:
    lines = []
    lines.append(f"{'=' * 60}")
    lines.append(f"  Earworm Bridge — {mode} search")
    lines.append(f"{'=' * 60}")

    if not results:
        lines.append("  No matching samples found.")
    else:
        lines.append(f"  {len(results)} matches:")
        lines.append("")
        for i, r in enumerate(results, 1):
            lines.append(f"  {i:3d}. [{r.score:.3f}] {r.filename}")
            lines.append(f"       {r.path}")
            if r.match_source != mode:
                lines.append(f"       (via {r.match_source})")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def _cmd_calibrate(args: argparse.Namespace) -> None:
    from earworm.calibration.corpus import Corpus

    corpus = Corpus(corpus_dir=args.corpus_dir) if hasattr(args, "corpus_dir") and args.corpus_dir else Corpus()

    if args.cal_command is None:
        print("Usage: earworm calibrate {init,list,add,run,check,report,journal}", file=sys.stderr)
        sys.exit(1)

    if args.cal_command == "init":
        from earworm.calibration.seeds import seed_corpus

        seed_corpus(corpus)
        tracks = corpus.list_tracks()
        print(f"Corpus initialized with {len(tracks)} tracks:", file=sys.stderr)
        for t in tracks:
            descs = len(t.human_descriptions)
            print(f"  {t.track_id}: {t.artist} — {t.title} ({descs} descriptions)", file=sys.stderr)

    elif args.cal_command == "list":
        tracks = corpus.list_tracks()
        if not tracks:
            print("Corpus is empty. Run 'earworm calibrate init' first.", file=sys.stderr)
            sys.exit(1)

        for t in tracks:
            status = []
            if t.audio_path:
                status.append("audio")
            if t.layer1:
                status.append("L1")
            if t.layer2:
                status.append("L2")
            if t.layer3:
                status.append("L3")
            if t.voice_result:
                status.append("voice")
            if t.alignments:
                aligned = sum(1 for a in t.alignments if a.aligned)
                status.append(f"aligned:{aligned}/{len(t.alignments)}")
            if t.divergences:
                status.append(f"div:{len(t.divergences)}")

            status_str = ", ".join(status) if status else "pending"
            descs = len(t.human_descriptions)
            print(f"  {t.track_id}")
            print(f"    {t.artist} — {t.title} ({t.year})")
            print(f"    {descs} descriptions | {status_str}")

    elif args.cal_command == "add":
        path = args.file.resolve()
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

        entry = corpus.get_track(args.track_id)
        if entry is None:
            print(f"Error: track not found: {args.track_id}", file=sys.stderr)
            print("Available tracks:", file=sys.stderr)
            for t in corpus.list_tracks():
                print(f"  {t.track_id}", file=sys.stderr)
            sys.exit(1)

        entry.audio_path = str(path)
        corpus.save()
        print(f"Audio linked: {entry.track_id} → {path}", file=sys.stderr)

    elif args.cal_command == "run":
        from earworm.calibration.runner import CalibrationRunner

        runner = CalibrationRunner(corpus)

        if args.track:
            entry = corpus.get_track(args.track)
            if entry is None:
                print(f"Error: track not found: {args.track}", file=sys.stderr)
                sys.exit(1)
            if not entry.audio_path:
                print("Error: no audio file linked. Use 'earworm calibrate add'", file=sys.stderr)
                sys.exit(1)

            print(f"Running: {entry.artist} — {entry.title}", file=sys.stderr)
            runner.run_track(entry, include_voice=args.voice)
            corpus.save_result(entry)
            corpus.save()
            print("Done.", file=sys.stderr)
        else:
            results = runner.run_pending(include_voice=args.voice)
            if results:
                print(f"\nAnalyzed {len(results)} tracks.", file=sys.stderr)

    elif args.cal_command == "check":
        from earworm.calibration.alignment import AlignmentChecker

        checker = AlignmentChecker()
        pending = corpus.tracks_needing_alignment()

        if not pending:
            all_checked = [e for e in corpus.list_tracks() if e.alignments]
            if all_checked:
                print("All analyzed tracks already checked.", file=sys.stderr)
            else:
                print("No tracks ready for alignment check.", file=sys.stderr)
                print("Tracks need both pipeline results and human descriptions.", file=sys.stderr)
            return

        for entry in pending:
            print(f"\nChecking: {entry.artist} — {entry.title}", file=sys.stderr)
            results = checker.check(entry)

            for r in results:
                mark = "✓" if r.aligned else "✗"
                print(f"  {mark} {r.dimension.value}: pipeline='{r.pipeline_says}' | human='{r.human_says}'")

            if entry.divergences:
                print(f"\n  Divergences ({len(entry.divergences)}):")
                for d in entry.divergences:
                    print(f"    → {d}")

        corpus.save()

    elif args.cal_command == "report":
        from earworm.calibration.alignment import AlignmentChecker

        checker = AlignmentChecker()
        entries = corpus.list_tracks()
        report = checker.generate_report(entries)

        if args.json:
            print(report.model_dump_json(indent=2))
        else:
            print(f"{'=' * 60}")
            print("  Earworm Calibration Report")
            print(f"{'=' * 60}")
            print(f"  Tracks: {report.total_tracks} total, "
                  f"{report.analyzed_tracks} analyzed, "
                  f"{report.checked_tracks} checked")
            print(f"  Alignment rate: {report.alignment_rate:.0%}")
            print()
            if report.strongest_dimensions:
                print(f"  Strongest: {', '.join(report.strongest_dimensions)}")
            if report.weakest_dimensions:
                print(f"  Weakest:   {', '.join(report.weakest_dimensions)}")
            if report.notable_divergences:
                print()
                print("  Notable divergences:")
                for d in report.notable_divergences:
                    print(f"    → {d}")
            print(f"{'=' * 60}")

    elif args.cal_command == "journal":
        _cmd_journal(args)


def _cmd_journal(args: argparse.Namespace) -> None:
    from earworm.calibration.journal import JournalManager

    corpus_dir = args.corpus_dir if hasattr(args, "corpus_dir") and args.corpus_dir else None
    if corpus_dir:
        journal_path = corpus_dir / "journal.json"
    else:
        journal_path = None

    mgr = JournalManager(journal_path=journal_path)

    if args.journal_command is None:
        print("Usage: earworm calibrate journal {show,observe,pattern,diverge}", file=sys.stderr)
        sys.exit(1)

    if args.journal_command == "show":
        if args.json:
            print(mgr.journal.model_dump_json(indent=2))
        else:
            summary = mgr.summary()
            print(f"{'=' * 60}")
            print("  Earworm Taste Journal")
            print(f"{'=' * 60}")
            print(f"  Observations: {summary['total_observations']} across {summary['tracks_observed']} tracks")
            print(f"  Patterns:     {summary['patterns_identified']} identified")
            print(f"  Divergences:  {summary['total_divergences']} total")
            print(f"    Gaps:       {summary['gaps']}")
            print(f"    Taste:      {summary['taste_differences']}")
            print(f"    Unclear:    {summary['unclear']}")
            print()
            if summary["strongest_patterns"]:
                print("  Strongest patterns:")
                for p in summary["strongest_patterns"]:
                    print(f"    • {p}")
            print(f"{'=' * 60}")

    elif args.journal_command == "observe":
        from earworm.calibration.journal import JournalSampleRef

        sample_refs = []
        for s in args.samples:
            parts = s.split(":", 3)
            if len(parts) >= 2:
                sample_refs.append(JournalSampleRef(
                    sample_id=int(parts[0]),
                    filename=parts[1],
                    score=float(parts[2]) if len(parts) > 2 else 0.0,
                    why=parts[3] if len(parts) > 3 else "",
                ))

        obs = mgr.add_observation(
            track_id=args.track_id,
            what_i_noticed=args.noticed,
            what_stood_out=args.stood_out,
            what_i_missed=args.missed,
            raw_reaction=args.reaction,
            sample_references=sample_refs if sample_refs else None,
        )
        mgr.save()
        n_samples = len(obs.sample_references)
        suffix = f" ({n_samples} sample refs)" if n_samples else ""
        print(f"Observation recorded for {obs.track_id}{suffix}", file=sys.stderr)

    elif args.journal_command == "pattern":
        pattern = mgr.add_pattern(
            name=args.name,
            description=args.description,
            supporting_tracks=args.tracks,
            confidence=args.confidence,
        )
        mgr.save()
        existing = "Updated" if len(pattern.supporting_tracks) > len(args.tracks) else "Recorded"
        print(f"{existing} pattern: {pattern.name} (confidence: {pattern.confidence:.0%})", file=sys.stderr)

    elif args.journal_command == "diverge":
        div = mgr.classify_divergence(
            track_id=args.track_id,
            dimension=args.dimension,
            pipeline_perception=args.pipeline,
            human_consensus=args.human,
            classification=args.classify,
            reasoning=args.reasoning,
        )
        mgr.save()
        print(f"Divergence classified: {div.dimension} [{div.classification}] for {div.track_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
