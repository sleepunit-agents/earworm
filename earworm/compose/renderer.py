"""Score → WAV renderer using pytheory's synthesis engine.

Renders a pytheory Score to a WAV file by synthesizing each part's notes
with the assigned synth waveform, mixing all parts together, and writing
the result using scipy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io.wavfile

from pytheory.live import SAMPLE_RATE, _SYNTH_FUNCTIONS
from pytheory.play import Synth
from pytheory.rhythm import Score

# Attack/release envelope constants — applied per note to eliminate click artifacts
# that occur when raw oscillator samples are abruptly switched on/off.
_ATTACK_SAMPLES = int(0.020 * SAMPLE_RATE)   # 20 ms linear ramp-in
_RELEASE_SAMPLES = int(0.050 * SAMPLE_RATE)  # 50 ms linear ramp-out


def render_wav(score: Score, output: Path) -> None:
    """Render a pytheory Score to a WAV file at *output*.

    Each Part in the score is synthesized using its assigned synth, then all
    parts are mixed together. The output is normalized to -1 dBFS and saved
    as 16-bit stereo WAV at 44100 Hz.
    """
    beats_per_sec = score.bpm / 60.0
    total_samples = int(score.duration_ms / 1000.0 * SAMPLE_RATE) + SAMPLE_RATE  # 1s tail
    buf = np.zeros(total_samples, dtype=np.float64)

    for part in score.parts.values():
        synth_key = part.synth.value if isinstance(part.synth, Synth) else str(part.synth)
        synth_fn = _SYNTH_FUNCTIONS.get(synth_key)
        if synth_fn is None:
            synth_fn = _SYNTH_FUNCTIONS["sine"]

        volume = getattr(part, "volume", 0.5)
        pos_beats = 0.0

        for note in part.notes:
            dur_samples = max(1, int(note.beats / beats_per_sec * SAMPLE_RATE))
            pos_samples = int(pos_beats / beats_per_sec * SAMPLE_RATE)

            if note.tone is not None and hasattr(note.tone, "pitch"):
                # Single tone
                hz = note.tone.pitch()
                samples = _apply_envelope(synth_fn(hz, n_samples=dur_samples).astype(np.float64))
                _mix_into(buf, samples, pos_samples, volume)
            elif note.tone is not None and hasattr(note.tone, "tones"):
                # Chord — render each tone and mix at reduced volume
                chord_vol = volume / max(1, len(note.tone.tones))
                for tone in note.tone.tones:
                    if hasattr(tone, "pitch"):
                        hz = tone.pitch()
                        samples = _apply_envelope(synth_fn(hz, n_samples=dur_samples).astype(np.float64))
                        _mix_into(buf, samples, pos_samples, chord_vol)

            pos_beats += note.beats

    # Normalize to -1 dBFS
    peak = np.max(np.abs(buf))
    if peak > 0:
        buf = buf / peak * 0.9

    audio = (buf * 32767).astype(np.int16)

    # Write stereo (duplicate mono channel)
    stereo = np.column_stack([audio, audio])
    scipy.io.wavfile.write(str(output), SAMPLE_RATE, stereo)


def _apply_envelope(samples: np.ndarray) -> np.ndarray:
    """Apply a linear attack/release fade to eliminate click artifacts.

    The first _ATTACK_SAMPLES frames ramp from 0 → 1; the last
    _RELEASE_SAMPLES frames ramp from 1 → 0.  Each ramp is capped at n//4
    so very short notes still get a proportional fade without overlap.
    """
    n = len(samples)
    env = np.ones(n, dtype=np.float64)
    att = min(_ATTACK_SAMPLES, n // 4)
    rel = min(_RELEASE_SAMPLES, n // 4)
    if att > 0:
        env[:att] = np.linspace(0.0, 1.0, att)
    if rel > 0:
        env[n - rel:] = np.linspace(1.0, 0.0, rel)
    return samples * env


def _mix_into(buf: np.ndarray, samples: np.ndarray, offset: int, volume: float) -> None:
    """Add *samples* scaled by *volume* into *buf* starting at *offset*."""
    end = min(offset + len(samples), len(buf))
    if end <= offset:
        return
    chunk = samples[: end - offset].astype(np.float64)
    # Normalize to [-1, 1] range (pytheory returns int16)
    chunk = chunk / 32768.0
    buf[offset:end] += chunk * volume
