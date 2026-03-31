"""CLI entry point for earworm analysis."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from earworm.pipeline import analyze_layer1, analyze_layer2, analyze_layer3


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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "voice":
        _cmd_voice(args)
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

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def _cmd_calibrate(args: argparse.Namespace) -> None:
    from earworm.calibration.corpus import Corpus

    corpus = Corpus(corpus_dir=args.corpus_dir) if hasattr(args, "corpus_dir") and args.corpus_dir else Corpus()

    if args.cal_command is None:
        print("Usage: earworm calibrate {init,list,add,run,check,report}", file=sys.stderr)
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


if __name__ == "__main__":
    main()
