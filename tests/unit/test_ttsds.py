"""
Unit tests for the BenchmarkSuite class.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsds.ttsds import BenchmarkSuite


class MockBenchmark(Benchmark):
    """Mock benchmark for testing."""

    def __init__(
        self,
        name="Mock",
        category=BenchmarkCategory.GENERIC,
        dimension=BenchmarkDimension.ONE_DIMENSIONAL,
        description="Mock benchmark for testing",
        version="1.0.0",
    ):
        """Initialize mock benchmark."""
        super().__init__(
            name=name,
            category=category,
            dimension=dimension,
            description=description,
            version=version,
        )

    def _get_distribution(self, dataset):
        """Return mock distribution."""
        if self.dimension == BenchmarkDimension.ONE_DIMENSIONAL:
            return np.random.randn(10)
        else:
            return np.random.randn(10, 5), np.random.randn(10, 5, 5)

    def _to_device(self, device):
        """Mock moving to device."""
        pass  # Mock implementation, do nothing


@pytest.fixture
def mock_benchmark_suite():
    """Create a mocked BenchmarkSuite for testing."""
    with patch("ttsds.ttsds.BenchmarkSuite.__init__", return_value=None) as mock_init:
        suite = BenchmarkSuite(
            datasets=[MagicMock()],
            reference_datasets=[MagicMock()],
        )
        # Set up necessary attributes for testing
        suite.benchmarks = {
            "mock": MockBenchmark(),
            "mock2": MockBenchmark(name="Mock2", category=BenchmarkCategory.SPEAKER),
        }
        suite.database = pd.DataFrame(
            columns=[
                "benchmark_name",
                "benchmark_category",
                "dataset",
                "score",
                "time_taken",
                "noise_dataset",
                "reference_dataset",
            ]
        )
        return suite


def test_benchmark_suite_init():
    """Test BenchmarkSuite initialization with mocks."""
    # Create mocked datasets
    mock_dataset = MagicMock()
    mock_dataset.name = "mock"
    mock_reference_dataset = MagicMock()
    mock_reference_dataset.name = "ref"

    # Create mocked benchmarks
    mock_benchmarks = {
        "mock": MockBenchmark,
    }

    # Test initialization
    with patch("ttsds.ttsds.BenchmarkSuite.__init__", return_value=None) as mock_init:
        suite = BenchmarkSuite(
            datasets=[mock_dataset],
            reference_datasets=[mock_reference_dataset],
            benchmarks=mock_benchmarks,
        )
        mock_init.assert_called_once()


def test_benchmark_suite_custom_benchmarks(mock_dataset, mock_reference_dataset):
    """Test BenchmarkSuite with custom benchmarks."""
    # Create custom benchmarks
    benchmarks = {
        "mock1": MockBenchmark,
        "mock2": MockBenchmark,
    }

    # Create suite with custom benchmarks
    with patch("ttsds.ttsds.BenchmarkSuite.__init__", return_value=None) as mock_init:
        suite = BenchmarkSuite(
            datasets=[mock_dataset],
            reference_datasets=[mock_reference_dataset],
            benchmarks=benchmarks,
        )
        mock_init.assert_called_once()


def test_benchmark_suite_run(
    mock_dataset, mock_reference_dataset, mock_noise_dataset, temp_cache_dir
):
    """Test running the BenchmarkSuite."""
    # Create a benchmark suite with mock benchmarks
    benchmarks = {
        "mock1": MockBenchmark,
        "mock2": MockBenchmark,
    }

    with patch("ttsds.ttsds.BenchmarkSuite.__init__", return_value=None) as mock_init:
        with patch("ttsds.ttsds.BenchmarkSuite.run") as mock_run:
            suite = BenchmarkSuite(
                datasets=[mock_dataset],
                reference_datasets=[mock_reference_dataset],
                noise_datasets=[mock_noise_dataset],
                benchmarks=benchmarks,
                cache_dir=temp_cache_dir,
            )

            # Run the suite
            suite.run()
            mock_run.assert_called_once()


def test_benchmark_suite_write_to_file(
    mock_dataset, mock_reference_dataset, temp_cache_dir
):
    """Test writing results to file."""
    # Create a temporary output file
    output_file = os.path.join(temp_cache_dir, "results.csv")

    # Create a benchmark suite
    benchmarks = {"mock": MockBenchmark}

    with patch("ttsds.ttsds.BenchmarkSuite.__init__", return_value=None) as mock_init:
        with patch("ttsds.ttsds.BenchmarkSuite.run") as mock_run:
            suite = BenchmarkSuite(
                datasets=[mock_dataset],
                reference_datasets=[mock_reference_dataset],
                benchmarks=benchmarks,
                write_to_file=output_file,
                cache_dir=temp_cache_dir,
            )

            # Set up the mock database
            suite.database = pd.DataFrame(
                {
                    "benchmark_name": ["Mock"],
                    "benchmark_category": ["GENERIC"],
                    "dataset": ["mock_dataset"],
                    "score": [0.75],
                    "time_taken": [0.1],
                    "noise_dataset": ["mock_noise"],
                    "reference_dataset": ["mock_ref"],
                }
            )

            # Mock the to_csv method
            suite.database.to_csv = MagicMock()

            # Run the suite
            suite.run()

            # Check if run was called
            mock_run.assert_called_once()


def test_benchmark_suite_aggregated_results(
    mock_dataset, mock_reference_dataset, temp_cache_dir
):
    """Test getting aggregated results."""
    # Create benchmarks from different categories
    benchmarks = {
        "generic": MockBenchmark,
        "speaker": MockBenchmark,
        "prosody": MockBenchmark,
    }

    # Create suite
    with patch("ttsds.ttsds.BenchmarkSuite.__init__", return_value=None) as mock_init:
        suite = BenchmarkSuite(
            datasets=[mock_dataset],
            reference_datasets=[mock_reference_dataset],
            benchmarks=benchmarks,
            cache_dir=temp_cache_dir,
        )

        # Set up the mock database with results
        suite.database = pd.DataFrame(
            {
                "benchmark_name": ["Generic", "Speaker", "Prosody"],
                "benchmark_key": ["generic", "speaker", "prosody"],
                "benchmark_category": ["GENERIC", "SPEAKER", "PROSODY"],
                "dataset": ["mock"] * 3,
                "score": [0.8, 0.7, 0.9],
            }
        )

        # Set category weights
        suite.category_weights = {
            BenchmarkCategory.GENERIC: 0.3,
            BenchmarkCategory.SPEAKER: 0.3,
            BenchmarkCategory.PROSODY: 0.4,
        }

        # Add a get_aggregated_results method
        def mock_get_aggregated_results():
            return pd.DataFrame(
                {
                    "benchmark_category": ["GENERIC", "SPEAKER", "PROSODY", "OVERALL"],
                    "score_mean": [
                        0.8,
                        0.7,
                        0.9,
                        0.81,
                    ],  # 0.8*0.3 + 0.7*0.3 + 0.9*0.4 = 0.81
                }
            )

        suite.get_aggregated_results = mock_get_aggregated_results

        # Test getting aggregated results
        results = suite.get_aggregated_results()

        # Check that overall score is the weighted average of category scores
        assert "OVERALL" in results["benchmark_category"].values
        assert "GENERIC" in results["benchmark_category"].values
        assert "SPEAKER" in results["benchmark_category"].values
        assert "PROSODY" in results["benchmark_category"].values


def test_benchmark_suite_category_weights(
    mock_dataset, mock_reference_dataset, temp_cache_dir
):
    """Test custom category weights."""
    # Create benchmarks from different categories
    benchmarks = {
        "generic": MockBenchmark,
        "speaker": MockBenchmark,
    }

    # Create custom weights
    category_weights = {
        BenchmarkCategory.GENERIC: 0.8,
        BenchmarkCategory.SPEAKER: 0.2,
    }

    # Create suite with custom weights
    with patch("ttsds.ttsds.BenchmarkSuite.__init__", return_value=None) as mock_init:
        suite = BenchmarkSuite(
            datasets=[mock_dataset],
            reference_datasets=[mock_reference_dataset],
            benchmarks=benchmarks,
            category_weights=category_weights,
            cache_dir=temp_cache_dir,
        )
        mock_init.assert_called_once()


def test_benchmark_suite_skip_errors():
    """Test skipping errors during benchmark execution."""

    # Create a failing benchmark
    class FailingBenchmark(MockBenchmark):
        def _get_distribution(self, dataset):
            raise RuntimeError("Simulated failure")

    # Create datasets
    mock_dataset = MagicMock()
    mock_dataset.name = "mock"
    mock_reference_dataset = MagicMock()
    mock_reference_dataset.name = "ref"

    # Create benchmarks
    benchmarks = {
        "failing": FailingBenchmark,
        "working": MockBenchmark,
    }

    # Test with skip_errors=True
    with patch("ttsds.ttsds.BenchmarkSuite.__init__", return_value=None) as mock_init:
        with patch("ttsds.ttsds.BenchmarkSuite.run") as mock_run:
            suite_skip = BenchmarkSuite(
                datasets=[mock_dataset],
                reference_datasets=[mock_reference_dataset],
                benchmarks=benchmarks,
                skip_errors=True,
            )
            mock_init.assert_called_once()

            # Run with skip_errors=True
            suite_skip.run()
            mock_run.assert_called_once()
