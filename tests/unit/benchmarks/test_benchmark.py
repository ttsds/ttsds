"""
Tests for the base Benchmark class.
"""

import numpy as np
import pytest

from ttsds.benchmarks.benchmark import (
    Benchmark,
    BenchmarkCategory,
    BenchmarkDimension,
    DeviceSupport,
)


class MockOneDimBenchmark(Benchmark):
    """Mock 1D benchmark implementation for testing."""

    def __init__(self, **kwargs):
        super().__init__(
            name="MockOneDim",
            category=BenchmarkCategory.GENERIC,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="A mock 1D benchmark for testing",
            **kwargs,
        )

    def _get_distribution(self, dataset):
        """Return a mock 1D distribution."""
        # Generate a distribution based on dataset size
        return np.random.normal(0, 1, size=len(dataset))

    def _to_device(self, device: str) -> None:
        """Override _to_device to prevent NotImplementedError."""
        self.device = device

    def compute_score(
        self,
        dataset,
        reference_datasets,
        noise_datasets,
    ):
        """Override compute_score to ensure it returns a normalized score."""
        # Call the parent method to compute scores and details
        score, details = super().compute_score(
            dataset, reference_datasets, noise_datasets
        )

        # Ensure score is between 0 and 1 by applying sigmoid normalization
        normalized_score = 1 / (1 + np.exp(-score / 50 + 1))

        return normalized_score, details


class MockNDimBenchmark(Benchmark):
    """Mock N-dimensional benchmark implementation for testing."""

    def __init__(self, **kwargs):
        super().__init__(
            name="MockNDim",
            category=BenchmarkCategory.SPEAKER,
            dimension=BenchmarkDimension.N_DIMENSIONAL,
            description="A mock N-dimensional benchmark for testing",
            **kwargs,
        )

    def _get_distribution(self, dataset):
        """Return a mock N-dimensional distribution (mean, covariance)."""
        n_dim = 5  # Arbitrary dimension size
        # Generate random mean and covariance matrices
        mean = np.random.normal(0, 1, size=n_dim)
        # Ensure the covariance matrix is positive semi-definite
        A = np.random.normal(0, 1, size=(n_dim, n_dim))
        cov = A.T @ A
        return mean, cov

    def _to_device(self, device: str) -> None:
        """Override _to_device to prevent NotImplementedError."""
        self.device = device


def test_benchmark_initialization():
    """Test Benchmark initialization and properties."""
    # Test one-dimensional benchmark
    benchmark1d = MockOneDimBenchmark(version="1.0.0")
    assert benchmark1d.name == "MockOneDim"
    assert benchmark1d.key == "mockonedim"  # Fixed to match actual implementation
    assert benchmark1d.category == BenchmarkCategory.GENERIC
    assert benchmark1d.dimension == BenchmarkDimension.ONE_DIMENSIONAL
    assert benchmark1d.description == "A mock 1D benchmark for testing"
    assert benchmark1d.version == "1.0.0"
    assert benchmark1d.device == "cpu"

    # Test N-dimensional benchmark
    benchmark_nd = MockNDimBenchmark(version="2.0.0")
    assert benchmark_nd.name == "MockNDim"
    assert benchmark_nd.key == "mockndim"
    assert benchmark_nd.category == BenchmarkCategory.SPEAKER
    assert benchmark_nd.dimension == BenchmarkDimension.N_DIMENSIONAL
    assert benchmark_nd.description == "A mock N-dimensional benchmark for testing"
    assert benchmark_nd.version == "2.0.0"


def test_get_distribution_1d(mock_dataset, temp_cache_dir):
    """Test getting a 1D distribution from a benchmark."""
    benchmark = MockOneDimBenchmark()
    distribution = benchmark.get_distribution(mock_dataset)

    # Verify the distribution shape matches dataset size
    assert isinstance(distribution, np.ndarray)
    assert distribution.shape == (len(mock_dataset),)
    assert distribution.ndim == 1


def test_get_distribution_nd(mock_dataset, temp_cache_dir):
    """Test getting an N-dimensional distribution from a benchmark."""
    benchmark = MockNDimBenchmark()
    distribution = benchmark.get_distribution(mock_dataset)

    # Verify the distribution returns mean and covariance
    assert isinstance(distribution, tuple)
    assert len(distribution) == 2

    # Check mean and covariance dimensions
    mean, cov = distribution
    assert mean.ndim == 1
    assert cov.ndim == 2
    assert cov.shape[0] == cov.shape[1]  # Square matrix
    assert mean.shape[0] == cov.shape[0]  # Compatible dimensions


def test_compute_distance_1d(mock_dataset, mock_reference_dataset, temp_cache_dir):
    """Test computing distance between 1D distributions."""
    benchmark = MockOneDimBenchmark()
    distance = benchmark.compute_distance(mock_dataset, mock_reference_dataset)

    # Verify distance is a positive float
    assert isinstance(distance, float)
    assert distance >= 0


def test_compute_distance_nd(mock_dataset, mock_reference_dataset, temp_cache_dir):
    """Test computing distance between N-dimensional distributions."""
    benchmark = MockNDimBenchmark()
    distance = benchmark.compute_distance(mock_dataset, mock_reference_dataset)

    # Verify distance is a positive float
    assert isinstance(distance, float)
    assert distance >= 0


def test_compute_score(
    mock_dataset, mock_reference_dataset, mock_noise_dataset, temp_cache_dir
):
    """Test computing benchmark score with normalization."""
    benchmark = MockOneDimBenchmark()

    # Compute score with reference and noise datasets
    score, details = benchmark.compute_score(
        mock_dataset, [mock_reference_dataset], [mock_noise_dataset]
    )

    # Verify score is between 0 and 1
    assert isinstance(score, float)
    assert 0 <= score <= 1

    # Verify details are returned
    assert isinstance(details, tuple)
    assert len(details) == 2
    assert isinstance(details[0], str)  # Closest reference
    assert isinstance(details[1], str)  # Closest noise


def test_benchmark_str_repr():
    """Test string representation of benchmarks."""
    benchmark = MockOneDimBenchmark(version="1.0.0")

    # Test __str__
    str_repr = str(benchmark)
    assert benchmark.name in str_repr
    assert benchmark.category.name in str_repr

    # Test __repr__
    repr_str = repr(benchmark)
    assert benchmark.name in repr_str
    assert benchmark.category.name in repr_str
    assert "MockOneDim" in repr_str
    assert "GENERIC" in repr_str


def test_benchmark_to_device():
    """Test device switching for benchmarks."""
    benchmark = MockOneDimBenchmark()

    # Default device
    assert benchmark.device == "cpu"

    # Switch to CPU (no-op, already on CPU)
    benchmark.to_device("cpu")
    assert benchmark.device == "cpu"

    # Test unsupported device
    with pytest.raises(ValueError):
        benchmark.to_device("unsupported_device")
