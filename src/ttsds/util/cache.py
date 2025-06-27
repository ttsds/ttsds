"""
Cache utilities for storing and retrieving computed values.

This module provides functions for caching numpy arrays to disk
to speed up repeated computations. It is primarily used by the
benchmark system to avoid recomputing distributions.
"""

import os
from pathlib import Path
from hashlib import md5
from typing import Any, Optional, Union, TypeVar, cast, Tuple

import numpy as np

# Configure cache directory
CACHE_DIR = os.getenv("TTSDS_CACHE_DIR", os.path.expanduser("~/.cache/ttsds"))
CACHE_DIR = Path(CACHE_DIR)
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

# For potential multiprocessing lock
lock: Optional[Any] = None

# Type variable for array-like objects
ArrayLike = TypeVar("ArrayLike", bound=Union[np.ndarray, Tuple[np.ndarray, np.ndarray]])


def hash_md5(obj: Any) -> str:
    """
    Generate an MD5 hash for an object.

    Args:
        obj: The object to hash (must implement __hash__)

    Returns:
        str: Hexadecimal MD5 hash string
    """
    h = md5()
    h.update(str(obj.__hash__()).encode())
    return h.hexdigest()


def cache(obj: ArrayLike, name: str) -> ArrayLike:
    """
    Cache a numpy array or tuple of arrays to disk.

    Args:
        obj: The numpy array or tuple of arrays to cache
        name: The name/key for the cached data

    Returns:
        The original numpy array or tuple (for chaining)

    Note:
        Creates directory structure if it doesn't exist
        If obj is a tuple, each element is saved separately with a suffix
    """
    # Handle tuple of arrays (e.g., mean and covariance)
    if isinstance(obj, tuple) and len(obj) == 2:
        mean, cov = obj
        cache_mean_file = CACHE_DIR / f"{name}_mean.npy"
        cache_cov_file = CACHE_DIR / f"{name}_cov.npy"

        cache_mean_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_mean_file, mean)
        np.save(cache_cov_file, cov)
        return obj

    # Handle single array
    cache_file = CACHE_DIR / f"{name}.npy"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, obj)
    return obj


def load_cache(name: str) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Load a cached numpy array or tuple of arrays from disk.

    Args:
        name: The name/key of the cached data

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            The cached numpy array or tuple of arrays

    Raises:
        Exception: If loading the cache file fails
    """
    # Check if this is a tuple (mean, covariance)
    cache_mean_file = CACHE_DIR / f"{name}_mean.npy"
    cache_cov_file = CACHE_DIR / f"{name}_cov.npy"

    if cache_mean_file.exists() and cache_cov_file.exists():
        try:
            mean = np.load(cache_mean_file, allow_pickle=True)
            cov = np.load(cache_cov_file, allow_pickle=True)
            return (mean, cov)
        except Exception as e:
            print(f"Failed to load cache {cache_mean_file} or {cache_cov_file}: {e}")
            raise e

    # Otherwise, load single array
    cache_file = CACHE_DIR / f"{name}.npy"
    try:
        return np.load(cache_file, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load cache {cache_file}: {e}")
        raise e


def check_cache(name: str) -> bool:
    """
    Check if a cache file exists.

    Args:
        name: The name/key of the cached data

    Returns:
        bool: True if the cache file exists, False otherwise
    """
    # Check for tuple (mean, covariance) files
    cache_mean_file = CACHE_DIR / f"{name}_mean.npy"
    cache_cov_file = CACHE_DIR / f"{name}_cov.npy"

    if cache_mean_file.exists() and cache_cov_file.exists():
        return True

    # Check for single array file
    cache_file = CACHE_DIR / f"{name}.npy"
    if not cache_file.exists():
        print(f"Cache file {cache_file} does not exist")

    return cache_file.exists()
