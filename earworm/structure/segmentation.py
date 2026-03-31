"""Structural segmentation — detect section boundaries and label similar sections."""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import median_filter

from earworm.models import SegmentationFeatures


def extract_segmentation(
    y: np.ndarray, sr: int, n_sections: int | None = None
) -> SegmentationFeatures:
    """Detect section boundaries using spectral novelty and cluster similar sections.

    Approach:
    1. Compute CQT-based chroma features (good for harmonic structure)
    2. Stack with MFCCs (good for timbral changes)
    3. Build recurrence matrix and apply checkerboard kernel for novelty
    4. Peak-pick on novelty curve for boundaries
    5. Cluster segments by feature similarity to assign labels
    """
    duration = len(y) / sr

    # Short audio: return single section
    if duration < 2.0:
        return SegmentationFeatures(
            boundaries=[0.0, duration],
            labels=[0],
            n_sections=1,
            section_durations=[duration],
        )

    # Feature extraction — chroma CQT + MFCCs stacked
    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

    # Normalize and stack
    chroma_norm = librosa.util.normalize(chroma, axis=1)
    mfcc_norm = librosa.util.normalize(mfcc, axis=1)
    features = np.vstack([chroma_norm, mfcc_norm])

    # Build recurrence matrix (self-similarity)
    rec = librosa.segment.recurrence_matrix(
        features, width=3, mode="affinity", sym=True
    )

    # Checkerboard kernel novelty detection
    novelty = _checkerboard_novelty(rec)

    # Smooth the novelty curve
    if len(novelty) > 5:
        novelty = median_filter(novelty, size=5)

    # Determine number of sections if not specified
    if n_sections is None:
        # Adaptive: aim for sections of ~8-30 seconds
        target_section_dur = 15.0
        n_sections = max(2, min(int(duration / target_section_dur), 20))

    # Peak-pick for boundaries
    n_peaks = max(1, n_sections - 1)
    boundary_frames = _pick_peaks(novelty, n_peaks=n_peaks, min_distance=10)

    # Convert frames to times
    boundary_times = librosa.frames_to_time(boundary_frames, sr=sr, hop_length=hop_length)
    boundary_times = sorted(boundary_times.tolist())

    # Ensure start and end are included
    if len(boundary_times) == 0 or boundary_times[0] > 0.1:
        boundary_times = [0.0] + boundary_times
    else:
        boundary_times[0] = 0.0
    if boundary_times[-1] < duration - 0.1:
        boundary_times.append(duration)
    else:
        boundary_times[-1] = duration

    # Compute section durations
    section_durations = [
        boundary_times[i + 1] - boundary_times[i]
        for i in range(len(boundary_times) - 1)
    ]

    # Cluster segments by feature similarity
    n_actual_sections = len(section_durations)
    labels = _cluster_segments(features, boundary_frames, sr, hop_length, n_actual_sections)

    return SegmentationFeatures(
        boundaries=[round(t, 3) for t in boundary_times],
        labels=labels,
        n_sections=n_actual_sections,
        section_durations=[round(d, 3) for d in section_durations],
    )


def _checkerboard_novelty(rec: np.ndarray, kernel_size: int = 16) -> np.ndarray:
    """Compute novelty curve from recurrence matrix using checkerboard kernel."""
    n = rec.shape[0]
    half = kernel_size // 2
    novelty = np.zeros(n)

    # Build checkerboard kernel
    kernel = np.ones((kernel_size, kernel_size))
    kernel[:half, :half] = -1
    kernel[half:, half:] = -1

    for i in range(half, n - half):
        patch = rec[i - half : i + half, i - half : i + half]
        if patch.shape == kernel.shape:
            novelty[i] = np.sum(patch * kernel)

    # Normalize to 0-1
    novelty = np.maximum(novelty, 0)
    max_val = novelty.max()
    if max_val > 0:
        novelty /= max_val

    return novelty


def _pick_peaks(
    signal: np.ndarray, n_peaks: int, min_distance: int = 10
) -> np.ndarray:
    """Pick the top n_peaks from a signal with minimum distance constraint."""
    if len(signal) == 0:
        return np.array([], dtype=int)

    # Find all local maxima
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            peaks.append((signal[i], i))

    if not peaks:
        # Fallback: evenly space boundaries
        indices = np.linspace(0, len(signal) - 1, n_peaks + 2, dtype=int)
        return indices[1:-1]

    # Sort by height, pick top peaks with distance constraint
    peaks.sort(reverse=True)
    selected = []
    for _, idx in peaks:
        if len(selected) >= n_peaks:
            break
        if all(abs(idx - s) >= min_distance for s in selected):
            selected.append(idx)

    return np.sort(np.array(selected, dtype=int))


def _cluster_segments(
    features: np.ndarray,
    boundary_frames: np.ndarray,
    sr: int,
    hop_length: int,
    n_sections: int,
) -> list[int]:
    """Assign labels to segments based on feature similarity."""
    if n_sections <= 1:
        return [0]

    n_frames = features.shape[1]

    # Build segment start/end frames
    all_frames = [0] + sorted(boundary_frames.tolist()) + [n_frames]

    # Compute mean feature vector per segment
    segment_features = []
    for i in range(len(all_frames) - 1):
        start = int(all_frames[i])
        end = int(all_frames[i + 1])
        if end <= start:
            end = start + 1
        end = min(end, n_frames)
        seg_feat = features[:, start:end].mean(axis=1)
        segment_features.append(seg_feat)

    # Trim to actual section count
    segment_features = segment_features[:n_sections]

    if len(segment_features) < 2:
        return list(range(len(segment_features)))

    seg_matrix = np.array(segment_features)

    # Cosine similarity between segments
    norms = np.linalg.norm(seg_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    seg_normed = seg_matrix / norms
    sim = seg_normed @ seg_normed.T

    # Simple greedy labeling: similar segments get the same label
    labels = [-1] * len(segment_features)
    next_label = 0
    similarity_threshold = 0.85

    for i in range(len(segment_features)):
        if labels[i] >= 0:
            continue
        labels[i] = next_label
        for j in range(i + 1, len(segment_features)):
            if labels[j] < 0 and sim[i, j] >= similarity_threshold:
                labels[j] = next_label
        next_label += 1

    return labels
