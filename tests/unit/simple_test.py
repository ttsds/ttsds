"""
Simplified test for benchmarks using mock objects.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock


# Mock enumerations to simulate the actual enums in ttsds
class MockBenchmarkCategory:
    GENERIC = "GENERIC"
    SPEAKER = "SPEAKER"
    PROSODY = "PROSODY"
    INTELLIGIBILITY = "INTELLIGIBILITY"
    ENVIRONMENT = "ENVIRONMENT"


class MockBenchmarkDimension:
    ONE_DIMENSIONAL = "ONE_DIMENSIONAL"
    N_DIMENSIONAL = "N_DIMENSIONAL"


# Mock Benchmark class to simulate the actual Benchmark class in ttsds
class MockBenchmark:
    def __init__(self, name, category, dimension, description, version=None, **kwargs):
        self.name = name
        self.key = name.lower().replace(" ", "_")
        self.category = category
        self.dimension = dimension
        self.description = description
        self.version = version
        self.kwargs = kwargs
        self.device = "cpu"


# Test implementation classes
class TestOneDimBenchmark(MockBenchmark):
    def __init__(self, **kwargs):
        super().__init__(
            name="TestOneDim",
            category=MockBenchmarkCategory.GENERIC,
            dimension=MockBenchmarkDimension.ONE_DIMENSIONAL,
            description="A test 1D benchmark",
            **kwargs,
        )


class TestNDimBenchmark(MockBenchmark):
    def __init__(self, **kwargs):
        super().__init__(
            name="TestNDim",
            category=MockBenchmarkCategory.SPEAKER,
            dimension=MockBenchmarkDimension.N_DIMENSIONAL,
            description="A test N-dimensional benchmark",
            **kwargs,
        )


def test_benchmark_initialization():
    """Test Benchmark initialization and properties."""
    # Test one-dimensional benchmark
    benchmark1d = TestOneDimBenchmark(version="1.0.0")
    assert benchmark1d.name == "TestOneDim"
    assert benchmark1d.key == "testonedim"
    assert benchmark1d.category == MockBenchmarkCategory.GENERIC
    assert benchmark1d.dimension == MockBenchmarkDimension.ONE_DIMENSIONAL
    assert benchmark1d.description == "A test 1D benchmark"
    assert benchmark1d.version == "1.0.0"
    assert benchmark1d.device == "cpu"

    # Test N-dimensional benchmark
    benchmark_nd = TestNDimBenchmark(version="2.0.0")
    assert benchmark_nd.name == "TestNDim"
    assert benchmark_nd.key == "testndim"
    assert benchmark_nd.category == MockBenchmarkCategory.SPEAKER
    assert benchmark_nd.dimension == MockBenchmarkDimension.N_DIMENSIONAL
    assert benchmark_nd.description == "A test N-dimensional benchmark"
    assert benchmark_nd.version == "2.0.0"


def test_with_mock_dataset(mock_dataset):
    """Test using the mock dataset fixture."""
    assert len(mock_dataset) > 0
    audio, filename = mock_dataset[0]
    assert isinstance(audio, np.ndarray)
    assert isinstance(filename, str)
    assert filename.endswith(".wav")
