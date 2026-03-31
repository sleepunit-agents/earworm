"""Energy arc analysis — builds, drops, climaxes, dynamic shape."""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

from earworm.models import EnergyArcFeatures


def extract_energy_arc(y: np.ndarray, sr: int) -> EnergyArcFeatures:
    """Analyze how energy changes over the track's duration.

    Computes a smoothed energy envelope, finds builds (sustained increases),
    drops (sharp decreases), and the overall climax position.
    """
    duration = len(y) / sr

    # RMS energy at frame level
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    n_frames = len(rms)
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    # Normalize to 0-1
    rms_max = rms.max()
    if rms_max > 0:
        energy = rms / rms_max
    else:
        energy = rms.copy()

    # Smooth for arc detection (2-second window)
    frames_per_sec = sr / hop_length
    smooth_window = max(1, int(frames_per_sec * 2))
    energy_smooth = uniform_filter1d(energy, size=smooth_window)

    # Downsample to ~2Hz for output
    target_rate = 2.0
    target_n = max(1, int(duration * target_rate))

    if n_frames > target_n:
        indices = np.linspace(0, n_frames - 1, target_n, dtype=int)
        energy_ds = energy_smooth[indices].tolist()
        times_ds = times[indices].tolist()
    else:
        energy_ds = energy_smooth.tolist()
        times_ds = times.tolist()

    # Climax: peak of smoothed energy
    climax_frame = int(np.argmax(energy_smooth))
    climax_time = float(times[climax_frame]) if climax_frame < len(times) else 0.0
    climax_position = climax_time / duration if duration > 0 else 0.0

    # Detect builds and drops from smoothed energy derivative
    # Use a coarser resolution for build/drop detection (~1 second windows)
    analysis_window = max(1, int(frames_per_sec))
    n_windows = max(1, n_frames // analysis_window)
    windowed_energy = np.array([
        energy_smooth[i * analysis_window : (i + 1) * analysis_window].mean()
        for i in range(n_windows)
    ])

    builds, drops = _detect_builds_and_drops(windowed_energy, analysis_window, times)

    # Dynamic spread
    dynamic_spread = float(energy_smooth.max() - energy_smooth.min())

    return EnergyArcFeatures(
        energy_curve=[round(v, 4) for v in energy_ds],
        energy_timestamps=[round(t, 3) for t in times_ds],
        climax_time=round(climax_time, 3),
        climax_position=round(climax_position, 3),
        n_builds=len(builds),
        n_drops=len(drops),
        build_times=[round(t, 3) for t in builds],
        drop_times=[round(t, 3) for t in drops],
        dynamic_spread=round(dynamic_spread, 4),
    )


def _detect_builds_and_drops(
    windowed_energy: np.ndarray,
    analysis_window: int,
    times: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Detect sustained energy increases (builds) and sharp decreases (drops).

    A build is 3+ consecutive windows of increasing energy.
    A drop is a decrease of >30% of dynamic range in 1-2 windows.
    """
    if len(windowed_energy) < 3:
        return [], []

    diff = np.diff(windowed_energy)
    dynamic_range = windowed_energy.max() - windowed_energy.min()
    if dynamic_range < 0.01:
        return [], []

    builds: list[float] = []
    drops: list[float] = []

    # Detect builds: 3+ consecutive positive differences
    run_start = None
    run_length = 0
    for i, d in enumerate(diff):
        if d > 0.005:  # Small positive threshold
            if run_start is None:
                run_start = i
            run_length += 1
        else:
            if run_length >= 3 and run_start is not None:
                frame_idx = min(run_start * analysis_window, len(times) - 1)
                builds.append(float(times[frame_idx]))
            run_start = None
            run_length = 0
    # Check final run
    if run_length >= 3 and run_start is not None:
        frame_idx = min(run_start * analysis_window, len(times) - 1)
        builds.append(float(times[frame_idx]))

    # Detect drops: sharp decrease > 30% of dynamic range
    drop_threshold = -0.3 * dynamic_range
    for i, d in enumerate(diff):
        if d < drop_threshold:
            frame_idx = min((i + 1) * analysis_window, len(times) - 1)
            drops.append(float(times[frame_idx]))

    return builds, drops
