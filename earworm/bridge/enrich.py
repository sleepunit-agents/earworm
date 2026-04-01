"""Voice enrichment — append related samples to a VoiceResult.

Connects Voice interpretation output to samplebank's sample library by
running the text search path (Voice fields → CLAP semantic search).
Optionally includes audio-to-audio results via combined mode.
"""

from __future__ import annotations

import logging

from earworm.bridge.samplebank import SamplebankBridge
from earworm.models import SampleReference, VoiceResult

logger = logging.getLogger(__name__)


def enrich_voice_with_samples(
    voice: VoiceResult,
    bridge: SamplebankBridge | None = None,
    audio_path: str | None = None,
    limit: int = 10,
    mode: str = "text",
) -> VoiceResult:
    """Attach related samples from samplebank to a VoiceResult.

    Args:
        voice: Existing VoiceResult to enrich.
        bridge: SamplebankBridge instance. Created with defaults if not provided.
        audio_path: Path to audio file (required for audio/combined modes).
        limit: Max samples to attach.
        mode: "text" (Voice fields only), "audio" (audio file only),
              or "combined" (both).

    Returns:
        New VoiceResult with related_samples populated.
    """
    if bridge is None:
        bridge = SamplebankBridge()

    try:
        if mode == "text":
            matches = bridge.search_by_voice(voice, limit=limit)
        elif mode == "audio":
            if not audio_path:
                logger.warning("Audio mode requested but no audio_path provided")
                return voice
            matches = bridge.search_by_audio(audio_path, limit=limit)
        elif mode == "combined":
            if not audio_path:
                logger.warning("Combined mode requested but no audio_path; falling back to text")
                matches = bridge.search_by_voice(voice, limit=limit)
            else:
                matches = bridge.search_combined(voice, audio_path, limit=limit)
        else:
            logger.warning("Unknown enrichment mode: %s", mode)
            return voice
    except Exception as e:
        logger.warning("Sample enrichment failed: %s", e)
        return voice

    refs = [
        SampleReference(
            sample_id=m.sample_id,
            filename=m.filename,
            path=m.path,
            score=m.score,
            match_source=m.match_source,
        )
        for m in matches
    ]

    return voice.model_copy(update={"related_samples": refs})
