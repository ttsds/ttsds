"""
Test importing from the ttsds package.
"""

import pytest


def test_import_version():
    """Test importing version directly from ttsds.__about__."""
    from ttsds.__about__ import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)


def test_import_benchmark():
    """Test importing the Benchmark class."""
    from ttsds.benchmarks.benchmark import (
        Benchmark,
        BenchmarkCategory,
        BenchmarkDimension,
    )

    assert Benchmark is not None
    assert BenchmarkCategory is not None
    assert BenchmarkDimension is not None
