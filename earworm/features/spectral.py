"""Spectral feature extraction — brightness, texture, pitch class distribution."""

from __future__ import annotations

import librosa
import numpy as np

from earworm.models import SpectralFeatures


def extract_spectral(y: np.ndarray, sr: int) -> SpectralFeatures:
    """Extract spectral features from a mono audio signal."""

    # MFCCs — timbral texture fingerprint
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Spectral shape descriptors
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    # Chromagram — pitch class energy distribution
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    return SpectralFeatures(
        mfcc_mean=mfccs.mean(axis=1).tolist(),
        mfcc_std=mfccs.std(axis=1).tolist(),
        spectral_centroid_mean=float(centroid.mean()),
        spectral_centroid_std=float(centroid.std()),
        spectral_bandwidth_mean=float(bandwidth.mean()),
        spectral_rolloff_mean=float(rolloff.mean()),
        spectral_contrast_mean=contrast.mean(axis=1).tolist(),
        spectral_flatness_mean=float(flatness.mean()),
        chroma_mean=chroma.mean(axis=1).tolist(),
        chroma_std=chroma.std(axis=1).tolist(),
    )
