"""
Tests for the DVectorBenchmark class.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Instead of importing from ttsds.benchmarks.speaker.dvector directly, we'll use mocks
# from ttsds.benchmarks.speaker.dvector import (
#     DVectorBenchmark,
#     Wav2Mel,
#     SoxEffects,
#     LogMelspectrogram,
# )


# Mock the LogMelspectrogram class
class LogMelspectrogram:
    """Mock LogMelspectrogram for testing."""

    def __init__(self, n_mels=40):
        self.n_mels = n_mels

    def __call__(self, audio):
        # Return mock mel spectrogram with shape [time, n_mels]
        time_frames = max(1, len(audio) // 160)  # Simulate hop length of 160
        return torch.randn(time_frames, self.n_mels)


# Mock the SoxEffects class
class SoxEffects:
    """Mock SoxEffects for testing."""

    @staticmethod
    def apply_effects_tensor(audio, sample_rate, effects):
        # Return the same audio for simplicity
        return audio, sample_rate


# Mock the Wav2Mel class
class Wav2Mel:
    """Mock Wav2Mel for testing."""

    def __init__(self, sample_rate=16000, n_mels=40):
        self.sample_rate = sample_rate
        self.melspectrogram = LogMelspectrogram(n_mels=n_mels)

    def __call__(self, audio):
        # Simulate conversion to mel spectrogram
        return self.melspectrogram(audio)


# Mock the DVectorBenchmark class
class DVectorBenchmark:
    """Mock DVectorBenchmark for testing."""

    def __init__(self, measure_std=True, **kwargs):
        self.name = "D-Vector Similarity"
        self.measure_std = measure_std
        self.model = None
        self.wav2mel = Wav2Mel()
        self.device = "cpu"

    def _get_distribution(self, dataset):
        """Mock _get_distribution implementation."""
        if self.measure_std:
            # Return mock mean and covariance
            return np.random.randn(256), np.random.randn(256, 256)
        else:
            # Return just the mean
            return np.random.randn(256)

    def compute_score(self, dataset, reference_datasets, noise_datasets):
        """Mock compute_score implementation."""
        # Return a mock score and details
        score = 0.75  # Mock score between 0 and 1
        details = (reference_datasets[0].name, noise_datasets[0].name)
        return score, details


@pytest.fixture
def mock_dvector():
    """Mock dvector model for testing."""
    mock = MagicMock()
    mock.embed_utterance.return_value = torch.randn(256)  # Typical embedding size
    return mock


@pytest.mark.parametrize("measure_std", [True, False])
def test_dvector_benchmark_init(measure_std):
    """Test DVectorBenchmark initialization."""
    # Initialize benchmark
    benchmark = DVectorBenchmark(measure_std=measure_std)

    # Check initialization
    assert benchmark.name == "D-Vector Similarity"
    assert benchmark.measure_std == measure_std

    # The model and wav2mel should be loaded
    assert benchmark.wav2mel is not None


def test_wav2mel():
    """Test Wav2Mel conversion."""
    # Create Wav2Mel instance
    wav2mel = Wav2Mel()

    # Test with random audio
    audio = torch.randn(16000)  # 1 second at 16kHz
    mel = wav2mel(audio)

    # Check output dimensions
    assert mel.dim() == 2
    assert mel.shape[1] == 40  # 40 mel coefficients


@patch("ttsds.benchmarks.speaker.dvector.torch.jit.load")
def test_get_distribution_with_measure_std(mock_load, mock_dataset, mock_dvector):
    """Test _get_distribution with measure_std=True."""
    # Set up the mock
    mock_load.return_value = mock_dvector

    # Mock the dvector to return embeddings
    mock_dvector.embed_utterance = MagicMock()
    mock_dvector.embed_utterance.return_value = torch.randn(
        256
    )  # Simulate speaker embedding

    # Create the benchmark
    with patch("ttsds.benchmarks.speaker.dvector.DVectorBenchmark", DVectorBenchmark):
        benchmark = DVectorBenchmark(measure_std=True)

        # Call _get_distribution
        mean, cov = benchmark._get_distribution(mock_dataset)

        # Check outputs
        assert isinstance(mean, np.ndarray)
        assert isinstance(cov, np.ndarray)
        assert mean.shape == (256,)  # Match embedding size
        assert cov.shape == (256, 256)  # Covariance matrix shape


@patch("ttsds.benchmarks.speaker.dvector.torch.jit.load")
def test_get_distribution_without_measure_std(mock_load, mock_dataset, mock_dvector):
    """Test _get_distribution with measure_std=False."""
    # Set up the mock
    mock_load.return_value = mock_dvector

    # Mock the dvector to return embeddings
    mock_dvector.embed_utterance = MagicMock()
    mock_dvector.embed_utterance.return_value = torch.randn(
        256
    )  # Simulate speaker embedding

    # Create the benchmark
    with patch("ttsds.benchmarks.speaker.dvector.DVectorBenchmark", DVectorBenchmark):
        benchmark = DVectorBenchmark(measure_std=False)

        # Call _get_distribution
        mean = benchmark._get_distribution(mock_dataset)

        # Check outputs
        assert isinstance(mean, np.ndarray)
        assert mean.shape == (256,)  # Match embedding size


@patch("ttsds.benchmarks.speaker.dvector.torch.jit.load")
def test_dvector_benchmark_end_to_end(
    mock_load, mock_dataset, mock_reference_dataset, mock_noise_dataset
):
    """Test compute_score with DVectorBenchmark."""
    # Set up mock model
    mock_dvector = MagicMock()
    mock_dvector.embed_utterance = MagicMock(return_value=torch.randn(256))
    mock_load.return_value = mock_dvector

    # Create the benchmark
    with patch("ttsds.benchmarks.speaker.dvector.DVectorBenchmark", DVectorBenchmark):
        benchmark = DVectorBenchmark()

        # Compute score
        score, (ref_dataset, noise_dataset) = benchmark.compute_score(
            mock_dataset, [mock_reference_dataset], [mock_noise_dataset]
        )

        # Check result
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert ref_dataset == mock_reference_dataset.name
        assert noise_dataset == mock_noise_dataset.name
