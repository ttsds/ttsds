"""
Benchmark implementations for evaluating audio datasets.

This module provides various benchmark classes organized by category:
- General: General-purpose benchmarks for audio analysis
- Prosody: Benchmarks focused on speech prosody characteristics
- Speaker: Benchmarks for speaker verification and identification
- Intelligibility: Benchmarks related to speech intelligibility
- Environment: Benchmarks measuring environmental factors
"""

from ttsds.benchmarks.benchmark import (
    Benchmark,
    BenchmarkCategory,
    BenchmarkDimension,
    DeviceSupport,
)

__all__ = [
    "Benchmark",
    "BenchmarkCategory",
    "BenchmarkDimension",
    "DeviceSupport",
]
