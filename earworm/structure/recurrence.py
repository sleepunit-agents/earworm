"""Recurrence analysis — self-similarity, repetition structure, novelty."""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import median_filter

from earworm.models import RecurrenceFeatures


def extract_recurrence(
    y: np.ndarray,
    sr: int,
    boundaries: list[float],
    labels: list[int],
) -> RecurrenceFeatures:
    """Analyze repetition structure using segmentation results.

    Takes boundaries and labels from segmentation to avoid recomputing.
    Adds novelty curve and repetition statistics.
    """
    duration = len(y) / sr
    hop_length = 512

    # Novelty curve from chroma recurrence
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_norm = librosa.util.normalize(chroma, axis=1)

    rec = librosa.segment.recurrence_matrix(
        chroma_norm, width=3, mode="affinity", sym=True
    )

    # Checkerboard novelty
    novelty = _checkerboard_novelty(rec)
    if len(novelty) > 5:
        novelty = median_filter(novelty, size=5).astype(float)

    # Downsample novelty to ~2Hz for reasonable output size
    n_frames = len(novelty)
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    target_rate = 2.0  # samples per second
    target_n = max(1, int(duration * target_rate))

    if n_frames > target_n:
        indices = np.linspace(0, n_frames - 1, target_n, dtype=int)
        novelty_ds = novelty[indices].tolist()
        times_ds = times[indices].tolist()
    else:
        novelty_ds = novelty.tolist()
        times_ds = times.tolist()

    # Repetition statistics from labels
    n_distinct = len(set(labels))

    # Compute duration per label
    label_durations: dict[int, float] = {}
    for i, label in enumerate(labels):
        seg_dur = boundaries[i + 1] - boundaries[i] if i + 1 < len(boundaries) else 0.0
        label_durations[label] = label_durations.get(label, 0.0) + seg_dur

    # Repetition ratio: fraction of track covered by labels that appear more than once
    from collections import Counter

    label_counts = Counter(labels)
    repeated_duration = sum(
        dur for label, dur in label_durations.items() if label_counts[label] > 1
    )
    repetition_ratio = repeated_duration / duration if duration > 0 else 0.0

    return RecurrenceFeatures(
        n_distinct_labels=n_distinct,
        repetition_ratio=min(repetition_ratio, 1.0),
        label_sequence=labels,
        label_durations={k: round(v, 3) for k, v in label_durations.items()},
        novelty_curve=[round(v, 4) for v in novelty_ds],
        novelty_timestamps=[round(t, 3) for t in times_ds],
    )


def _checkerboard_novelty(rec: np.ndarray, kernel_size: int = 16) -> np.ndarray:
    """Compute novelty curve from recurrence matrix using checkerboard kernel."""
    n = rec.shape[0]
    half = kernel_size // 2
    novelty = np.zeros(n)

    kernel = np.ones((kernel_size, kernel_size))
    kernel[:half, :half] = -1
    kernel[half:, half:] = -1

    for i in range(half, n - half):
        patch = rec[i - half : i + half, i - half : i + half]
        if patch.shape == kernel.shape:
            novelty[i] = np.sum(patch * kernel)

    novelty = np.maximum(novelty, 0)
    max_val = novelty.max()
    if max_val > 0:
        novelty /= max_val

    return novelty
