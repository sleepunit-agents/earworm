"""Calibration runner — executes pipeline on corpus tracks.

Runs Layer 1-3 analysis on tracks in the calibration corpus,
optionally including Voice interpretation.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

from earworm.calibration.corpus import Corpus
from earworm.calibration.models import CalibrationEntry
from earworm.pipeline import analyze_layer1, analyze_layer2, analyze_layer3


class CalibrationRunner:
    """Runs the earworm pipeline on calibration corpus tracks."""

    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def run_track(
        self,
        entry: CalibrationEntry,
        include_voice: bool = False,
        verbose: bool = True,
    ) -> CalibrationEntry:
        """Run L1-L3 pipeline on a single track.

        Args:
            entry: The calibration entry to analyze.
            include_voice: Also run Voice interpretation.
            verbose: Print progress to stderr.

        Returns:
            The updated entry with pipeline results.
        """
        if not entry.audio_path:
            raise ValueError(f"No audio path for track {entry.track_id}")

        path = Path(entry.audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if verbose:
            print("  L1 signal extraction...", file=sys.stderr, end="", flush=True)

        start = time.time()
        entry.layer1 = analyze_layer1(path)
        l1_time = time.time() - start

        if verbose:
            print(f" {l1_time:.1f}s", file=sys.stderr)
            print("  L2 structural comprehension...", file=sys.stderr, end="", flush=True)

        start = time.time()
        entry.layer2 = analyze_layer2(path, layer1=entry.layer1)
        l2_time = time.time() - start

        if verbose:
            print(f" {l2_time:.1f}s", file=sys.stderr)
            print("  L3 quality assessment...", file=sys.stderr, end="", flush=True)

        start = time.time()
        entry.layer3 = analyze_layer3(path, layer1=entry.layer1, layer2=entry.layer2)
        l3_time = time.time() - start

        if verbose:
            print(f" {l3_time:.1f}s", file=sys.stderr)

        if include_voice:
            if verbose:
                print("  Voice interpretation...", file=sys.stderr, end="", flush=True)

            from earworm.voice import interpret

            start = time.time()
            entry.voice_result = interpret(
                entry.layer1, entry.layer2, entry.layer3, mode="deep"
            )
            voice_time = time.time() - start

            if verbose:
                print(f" {voice_time:.1f}s", file=sys.stderr)

        entry.analyzed_at = datetime.now()
        return entry

    def run_pending(
        self,
        include_voice: bool = False,
        verbose: bool = True,
    ) -> list[CalibrationEntry]:
        """Run pipeline on all tracks that need analysis.

        Returns:
            List of entries that were analyzed.
        """
        pending = self.corpus.tracks_needing_analysis()
        if not pending and verbose:
            print("No tracks pending analysis.", file=sys.stderr)
            return []

        results = []
        for i, entry in enumerate(pending, 1):
            if verbose:
                print(
                    f"[{i}/{len(pending)}] {entry.artist} — {entry.title}",
                    file=sys.stderr,
                )

            try:
                self.run_track(entry, include_voice=include_voice, verbose=verbose)
                self.corpus.save_result(entry)
                results.append(entry)
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}", file=sys.stderr)

        self.corpus.save()
        return results
