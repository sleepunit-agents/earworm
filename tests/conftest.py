"""Shared test fixtures for earworm tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sine_440_mono() -> tuple[np.ndarray, int]:
    """5 seconds of 440Hz sine wave, mono, 44100Hz."""
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    return y, sr


@pytest.fixture
def sine_440_stereo() -> tuple[np.ndarray, int]:
    """5 seconds of 440Hz sine wave, stereo (identical channels), 44100Hz."""
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = 0.5 * np.sin(2 * np.pi * 440 * t)
    y = np.stack([mono, mono])
    return y, sr


@pytest.fixture
def wide_stereo() -> tuple[np.ndarray, int]:
    """5 seconds of stereo audio with different L/R content."""
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    y = np.stack([left, right])
    return y, sr


@pytest.fixture
def silence_mono() -> tuple[np.ndarray, int]:
    """5 seconds of silence, mono."""
    sr = 44100
    y = np.zeros(int(sr * 5.0))
    return y, sr


@pytest.fixture
def noise_mono() -> tuple[np.ndarray, int]:
    """5 seconds of white noise, mono."""
    sr = 44100
    rng = np.random.default_rng(42)
    y = 0.5 * rng.standard_normal(int(sr * 5.0))
    return y, sr
