"""Compositional quality assessment — harmonic vocabulary, rhythmic variety, structure."""

from __future__ import annotations

import librosa
import numpy as np

from earworm.models import CompositionQuality


def extract_composition(
    y: np.ndarray,
    sr: int,
    segment_labels: list[int] | None = None,
) -> CompositionQuality:
    """Assess compositional quality: harmonic vocabulary, rhythmic variety, melody.

    Optionally accepts segment labels from Layer 2 for structural variety scoring.
    """
    # Harmonic vocabulary — count distinct chord classes from chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    vocab, change_rate = _harmonic_vocabulary(chroma, sr)

    # Rhythmic variation — entropy of onset pattern
    rhythmic_var = _rhythmic_variation(y, sr)

    # Melodic range — pitch contour from predominant frequency
    melodic_range = _melodic_range(y, sr)

    # Structural variety — how different sections are from each other
    structural_var = _structural_variety(chroma, segment_labels)

    return CompositionQuality(
        harmonic_vocabulary=vocab,
        chord_change_rate=round(change_rate, 3),
        rhythmic_variation=round(rhythmic_var, 3),
        melodic_range_semitones=round(melodic_range, 1),
        structural_variety=round(structural_var, 3),
    )


def _harmonic_vocabulary(chroma: np.ndarray, sr: int) -> tuple[int, float]:
    """Count distinct chord classes and chord change rate.

    Quantizes each chroma frame to its strongest pitch class, then counts
    unique pitch class patterns across time.
    """
    n_frames = chroma.shape[1]
    if n_frames == 0:
        return 0, 0.0

    # Reduce to chord-like representation: top 3 pitch classes per frame
    chord_frames = []
    for i in range(n_frames):
        frame = chroma[:, i]
        if frame.sum() == 0:
            chord_frames.append((-1, -1, -1))
            continue
        top3 = tuple(sorted(np.argsort(frame)[-3:]))
        chord_frames.append(top3)

    # Count distinct chord classes
    distinct = set(chord_frames) - {(-1, -1, -1)}
    vocabulary = len(distinct)

    # Chord change rate: count transitions between different chords
    changes = 0
    for i in range(1, len(chord_frames)):
        if chord_frames[i] != chord_frames[i - 1]:
            changes += 1

    hop_length = 512
    duration = n_frames * hop_length / sr
    change_rate = changes / duration if duration > 0 else 0.0

    return vocabulary, change_rate


def _rhythmic_variation(y: np.ndarray, sr: int) -> float:
    """Measure rhythmic variation via onset pattern entropy.

    Higher entropy = more varied rhythmic patterns = score closer to 1.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if len(onset_env) == 0:
        return 0.0

    # Quantize onset strength into bins
    n_bins = 8
    max_val = onset_env.max()
    if max_val == 0:
        return 0.0

    quantized = np.clip(np.floor(onset_env / max_val * n_bins), 0, n_bins - 1).astype(int)

    # Compute entropy of the distribution
    counts = np.bincount(quantized, minlength=n_bins).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log2(probs)))

    # Normalize by max entropy (log2(n_bins))
    max_entropy = np.log2(n_bins)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _melodic_range(y: np.ndarray, sr: int) -> float:
    """Estimate melodic range in semitones from the predominant pitch contour."""
    # Use piptrack for pitch estimation
    S = np.abs(librosa.stft(y))
    pitches, magnitudes = librosa.piptrack(S=S, sr=sr)

    # Extract predominant pitch per frame (highest magnitude)
    pitch_track = []
    for i in range(pitches.shape[1]):
        mag_frame = magnitudes[:, i]
        if mag_frame.max() > 0:
            best_idx = mag_frame.argmax()
            pitch = pitches[best_idx, i]
            if pitch > 50:  # Ignore very low frequencies (noise)
                pitch_track.append(pitch)

    if len(pitch_track) < 2:
        return 0.0

    pitches_hz = np.array(pitch_track)

    # Filter outliers (keep 10th-90th percentile)
    p10 = np.percentile(pitches_hz, 10)
    p90 = np.percentile(pitches_hz, 90)
    filtered = pitches_hz[(pitches_hz >= p10) & (pitches_hz <= p90)]

    if len(filtered) < 2 or filtered.min() <= 0:
        return 0.0

    # Convert Hz range to semitones
    range_semitones = 12 * np.log2(filtered.max() / filtered.min())
    return float(range_semitones)


def _structural_variety(
    chroma: np.ndarray, segment_labels: list[int] | None
) -> float:
    """Score how different sections are from each other.

    If segment labels are provided, computes inter-section chroma distance.
    Otherwise, uses global chroma variance as a proxy.
    """
    if segment_labels is None or len(set(segment_labels)) <= 1:
        # Fallback: overall chroma variance as structural variety proxy
        if chroma.shape[1] == 0:
            return 0.0
        var = float(chroma.var(axis=1).mean())
        return min(1.0, var * 10.0)

    # Compute mean chroma per unique label
    n_frames = chroma.shape[1]
    n_sections = len(segment_labels)
    frames_per_section = max(1, n_frames // n_sections)

    section_chromas = []
    for i, label in enumerate(segment_labels):
        start = i * frames_per_section
        end = min((i + 1) * frames_per_section, n_frames)
        if end > start:
            section_chromas.append((label, chroma[:, start:end].mean(axis=1)))

    if len(section_chromas) < 2:
        return 0.0

    # Group by label and compute inter-label distances
    label_means = {}
    for label, feat in section_chromas:
        if label not in label_means:
            label_means[label] = []
        label_means[label].append(feat)

    # Average the features per label
    for label in label_means:
        label_means[label] = np.mean(label_means[label], axis=0)

    if len(label_means) < 2:
        return 0.0

    # Compute pairwise cosine distances between label means
    labels = list(label_means.keys())
    distances = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = label_means[labels[i]]
            b = label_means[labels[j]]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
                distances.append(1.0 - cos_sim)  # Convert similarity to distance

    if not distances:
        return 0.0

    # Average distance, capped at 1.0
    avg_dist = np.mean(distances)
    return min(1.0, float(avg_dist * 2.0))  # Scale up since chroma distances tend to be small
