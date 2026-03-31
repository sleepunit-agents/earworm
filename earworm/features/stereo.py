"""Stereo feature extraction — width, correlation, spatial distribution."""

from __future__ import annotations

import numpy as np

from earworm.models import StereoFeatures


def extract_stereo(y_stereo: np.ndarray, sr: int) -> StereoFeatures:
    """Extract stereo field features.

    Args:
        y_stereo: Audio signal — shape (channels, samples) or (samples,) for mono.
        sr: Sample rate.
    """
    if y_stereo.ndim == 1 or y_stereo.shape[0] == 1:
        return StereoFeatures(
            is_stereo=False,
            width_mean=0.0,
            width_std=0.0,
            correlation_mean=1.0,
            correlation_min=1.0,
            mid_side_ratio=1.0,
            balance=0.0,
        )

    left = y_stereo[0].astype(np.float64)
    right = y_stereo[1].astype(np.float64)

    # Mid/Side decomposition
    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    mid_energy = float(np.sum(mid**2))
    side_energy = float(np.sum(side**2))
    total_energy = mid_energy + side_energy

    mid_side_ratio = mid_energy / total_energy if total_energy > 0 else 1.0

    # Stereo width — ratio of side to total energy, windowed
    hop = int(sr * 0.5)  # 500ms windows
    widths = []
    correlations = []
    balances = []

    for i in range(0, len(left) - hop, hop):
        l_chunk = left[i : i + hop]
        r_chunk = right[i : i + hop]
        m_chunk = (l_chunk + r_chunk) / 2.0
        s_chunk = (l_chunk - r_chunk) / 2.0

        m_e = np.sum(m_chunk**2)
        s_e = np.sum(s_chunk**2)
        total = m_e + s_e
        if total > 0:
            widths.append(s_e / total)
        else:
            widths.append(0.0)

        # Pearson correlation between L and R
        l_std = np.std(l_chunk)
        r_std = np.std(r_chunk)
        if l_std > 0 and r_std > 0:
            corr = float(np.corrcoef(l_chunk, r_chunk)[0, 1])
            correlations.append(corr)
        else:
            correlations.append(1.0)

        # Balance — energy difference
        l_e = np.sum(l_chunk**2)
        r_e = np.sum(r_chunk**2)
        total_lr = l_e + r_e
        if total_lr > 0:
            balances.append((r_e - l_e) / total_lr)
        else:
            balances.append(0.0)

    widths_arr = np.array(widths) if widths else np.array([0.0])
    corr_arr = np.array(correlations) if correlations else np.array([1.0])
    bal_arr = np.array(balances) if balances else np.array([0.0])

    return StereoFeatures(
        is_stereo=True,
        width_mean=float(widths_arr.mean()),
        width_std=float(widths_arr.std()),
        correlation_mean=float(corr_arr.mean()),
        correlation_min=float(corr_arr.min()),
        mid_side_ratio=mid_side_ratio,
        balance=float(bal_arr.mean()),
    )
