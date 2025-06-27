"""
Pytest configuration for TTSDS tests.

This module contains fixtures and configuration for testing TTSDS components.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import numpy as np
import pytest


# Simplified MockDataset that doesn't inherit from ttsds.util.dataset.Dataset
class MockDataset:
    """Mock dataset for testing benchmarks."""

    def __init__(
        self,
        name: str = "mock_dataset",
        n_samples: int = 10,
        sample_rate: int = 16000,
        sample_length: float = 1.0,
    ):
        """
        Initialize a mock dataset.

        Args:
            name: Name of the dataset
            n_samples: Number of audio samples to generate
            sample_rate: Sample rate of the audio samples
            sample_length: Length of each audio sample in seconds
        """
        self.name = name
        self.sample_rate = sample_rate
        self._n_samples = n_samples
        self._sample_length = sample_length
        self._generate_samples()

    def sample(self, n_samples=None):
        """
        Return a sampled subset of the dataset.

        Args:
            n_samples: Number of samples to return. If None, return entire dataset.

        Returns:
            A new MockDataset with the sampled data
        """
        if n_samples is None or n_samples >= self._n_samples:
            return self

        sampled_dataset = MockDataset(
            name=f"{self.name}_sampled",
            n_samples=n_samples,
            sample_rate=self.sample_rate,
            sample_length=self._sample_length,
        )

        # Use the first n_samples
        indices = np.random.choice(self._n_samples, n_samples, replace=False)
        sampled_dataset._samples = [self._samples[i] for i in indices]

        return sampled_dataset

    def _generate_samples(self) -> None:
        """Generate random audio samples."""
        self._samples = []
        for i in range(self._n_samples):
            # Generate random audio with sine wave at different frequencies
            t = np.linspace(
                0, self._sample_length, int(self._sample_length * self.sample_rate)
            )
            freq = 440 * (1 + 0.1 * i)  # Different frequency for each sample
            audio = 0.5 * np.sin(2 * np.pi * freq * t)

            # Add some noise
            noise = np.random.normal(0, 0.01, len(audio))
            audio = audio + noise

            self._samples.append((audio, f"sample_{i}.wav"))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._n_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """Get a sample from the dataset."""
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )
        return self._samples[idx]

    def iter_with_progress(self, benchmark):
        """Iterator with progress tracking, simplified for testing."""
        for i in range(len(self)):
            yield self[i]


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    return MockDataset()


@pytest.fixture
def mock_reference_dataset():
    """Create a mock reference dataset for testing."""
    return MockDataset(name="mock_reference")


@pytest.fixture
def mock_noise_dataset():
    """Create a mock noise dataset for testing."""
    # Use higher frequency range to simulate noise
    dataset = MockDataset(name="mock_noise")
    for i in range(len(dataset._samples)):
        # Replace with actual noise
        t = np.linspace(
            0, dataset._sample_length, int(dataset._sample_length * dataset.sample_rate)
        )
        noise = np.random.normal(0, 0.1, len(t))
        dataset._samples[i] = (noise, f"noise_{i}.wav")
    return dataset


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for caching test results."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir
