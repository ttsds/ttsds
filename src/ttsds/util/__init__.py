"""
Utility modules for the TTSDS package.

This package provides various utility functions and classes used by the TTSDS
benchmarking system, including:
- dataset: Classes for loading and managing audio datasets
- cache: Utilities for caching computation results
- distances: Functions for measuring distribution distances
- parallel_distances: Utilities for parallel distance computation
- mpm: Implementations of Mel Pitch Metrics
"""

from ttsds.util.cache import cache, load_cache, check_cache, hash_md5
from ttsds.util.dataset import Dataset, DirectoryDataset, TarDataset, WavListDataset
from ttsds.util.distances import wasserstein_distance, frechet_distance
from ttsds.util.parallel_distances import DistanceCalculator

__all__ = [
    # Cache utilities
    "cache",
    "load_cache",
    "check_cache",
    "hash_md5",
    # Dataset classes
    "Dataset",
    "DirectoryDataset",
    "TarDataset",
    "WavListDataset",
    # Distance utilities
    "wasserstein_distance",
    "frechet_distance",
    "DistanceCalculator",
]
