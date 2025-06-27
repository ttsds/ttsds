"""
TTSDS - A toolkit for measuring text-to-speech (TTS) system degradation.

This package provides a suite of benchmarks for evaluating the quality
of text-to-speech systems by comparing their output to reference datasets.
"""

from ttsds.__about__ import __version__
from ttsds.ttsds import BenchmarkSuite

__all__ = ["BenchmarkSuite", "__version__"]
