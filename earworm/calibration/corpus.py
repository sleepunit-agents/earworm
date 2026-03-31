"""Calibration corpus management.

Handles loading, saving, and querying the calibration corpus — the collection
of tracks with known human descriptions that earworm uses to validate and
develop its perception.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from earworm.calibration.models import (
    CalibrationEntry,
    CorpusManifest,
    HumanDescription,
)


DEFAULT_CORPUS_DIR = Path(__file__).parent.parent.parent / "calibration"


class Corpus:
    """Manages the calibration corpus on disk.

    The corpus is a directory containing:
    - manifest.json — the corpus manifest with all entries
    - results/     — per-track JSON results from pipeline runs
    """

    def __init__(self, corpus_dir: Path | str | None = None):
        self.corpus_dir = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
        self.results_dir = self.corpus_dir / "results"
        self._manifest: CorpusManifest | None = None

    def _ensure_dirs(self) -> None:
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.corpus_dir / "manifest.json"

    @property
    def manifest(self) -> CorpusManifest:
        if self._manifest is None:
            self._manifest = self._load_manifest()
        return self._manifest

    def _load_manifest(self) -> CorpusManifest:
        if self.manifest_path.exists():
            data = json.loads(self.manifest_path.read_text())
            return CorpusManifest.model_validate(data)
        return CorpusManifest()

    def save(self) -> None:
        """Persist the manifest to disk."""
        self._ensure_dirs()
        self.manifest.updated_at = datetime.now()
        self.manifest_path.write_text(
            self.manifest.model_dump_json(indent=2, exclude_none=True)
        )

    def add_track(
        self,
        artist: str,
        title: str,
        album: str = "",
        year: int = 0,
        audio_path: str | None = None,
        track_id: str | None = None,
    ) -> CalibrationEntry:
        """Add a track to the corpus. Returns the new or existing entry."""
        tid = track_id or _slugify(f"{artist}-{title}")

        existing = self.get_track(tid)
        if existing is not None:
            if audio_path:
                existing.audio_path = audio_path
            return existing

        entry = CalibrationEntry(
            track_id=tid,
            artist=artist,
            title=title,
            album=album,
            year=year,
            audio_path=audio_path,
        )
        self.manifest.entries.append(entry)
        return entry

    def get_track(self, track_id: str) -> CalibrationEntry | None:
        """Look up a track by ID."""
        for entry in self.manifest.entries:
            if entry.track_id == track_id:
                return entry
        return None

    def list_tracks(self) -> list[CalibrationEntry]:
        """Return all corpus entries."""
        return list(self.manifest.entries)

    def add_human_description(
        self,
        track_id: str,
        source: str,
        text: str,
        tags: list[str] | None = None,
        key_observations: list[str] | None = None,
    ) -> HumanDescription:
        """Add a human description to a track."""
        entry = self.get_track(track_id)
        if entry is None:
            raise ValueError(f"Track not found: {track_id}")

        desc = HumanDescription(
            source=source,
            text=text,
            tags=tags or [],
            key_observations=key_observations or [],
        )
        entry.human_descriptions.append(desc)
        return desc

    def tracks_needing_analysis(self) -> list[CalibrationEntry]:
        """Return tracks that have audio paths but no pipeline results."""
        return [
            e for e in self.manifest.entries
            if e.audio_path and e.layer1 is None
        ]

    def tracks_needing_alignment(self) -> list[CalibrationEntry]:
        """Return tracks with pipeline results and human descriptions but no alignment."""
        return [
            e for e in self.manifest.entries
            if e.layer1 is not None
            and e.human_descriptions
            and not e.alignments
        ]

    def save_result(self, entry: CalibrationEntry) -> None:
        """Save detailed results for a track to a separate JSON file."""
        self._ensure_dirs()
        result_path = self.results_dir / f"{entry.track_id}.json"
        result_path.write_text(
            entry.model_dump_json(indent=2, exclude_none=True)
        )

    def load_result(self, track_id: str) -> CalibrationEntry | None:
        """Load detailed results from disk for a track."""
        result_path = self.results_dir / f"{track_id}.json"
        if not result_path.exists():
            return None
        data = json.loads(result_path.read_text())
        return CalibrationEntry.model_validate(data)


def _slugify(text: str) -> str:
    """Convert text to a URL/filesystem-safe slug."""
    import re

    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")
