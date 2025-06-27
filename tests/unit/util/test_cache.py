"""
Tests for the cache utilities.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import patch

# Import the hash function directly
from ttsds.util.cache import hash_md5


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for caching tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the cache directory fixture
        yield temp_dir


def test_hash_md5():
    """Test the hash_md5 function."""

    # Create objects with consistent hash values
    class MockObject:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            return hash(self.value)

    # Test with different objects
    obj1 = MockObject("test1")
    obj2 = MockObject("test2")
    obj3 = MockObject("test1")  # Same hash as obj1

    # Generate hashes
    hash1 = hash_md5(obj1)
    hash2 = hash_md5(obj2)
    hash3 = hash_md5(obj3)

    # Check that hashes are strings
    assert isinstance(hash1, str)
    assert isinstance(hash2, str)
    assert isinstance(hash3, str)

    # Check that different objects have different hashes
    assert hash1 != hash2

    # Check that objects with the same hash value have the same MD5 hash
    assert hash1 == hash3


def test_cache_and_load_array(temp_cache_dir):
    """Test caching and loading a numpy array."""
    # Create a test array
    test_array = np.random.randn(10, 5)

    # Mock the cache directory
    with patch("ttsds.util.cache.CACHE_DIR", Path(temp_cache_dir)):
        # Import functions here after patching
        from ttsds.util.cache import cache, load_cache, check_cache

        # Cache the array
        name = "test_array"
        cached_array = cache(test_array, name)

        # Check that the cache file exists
        cache_file = Path(temp_cache_dir) / f"{name}.npy"
        assert cache_file.exists()

        # Check that the function returns the original array
        assert np.array_equal(cached_array, test_array)

        # Load the array from cache
        loaded_array = load_cache(name)

        # Check that the loaded array matches the original
        assert np.array_equal(loaded_array, test_array)


def test_cache_and_load_tuple(temp_cache_dir):
    """Test caching and loading a tuple of arrays."""
    # Create test arrays
    mean = np.random.randn(5)
    cov = np.random.randn(5, 5)
    tuple_arrays = (mean, cov)

    # Mock the cache directory
    with patch("ttsds.util.cache.CACHE_DIR", Path(temp_cache_dir)):
        # Import functions here after patching
        from ttsds.util.cache import cache, load_cache, check_cache

        # Cache the tuple
        name = "test_tuple"
        cached_tuple = cache(tuple_arrays, name)

        # Check that the cache files exist
        mean_file = Path(temp_cache_dir) / f"{name}_mean.npy"
        cov_file = Path(temp_cache_dir) / f"{name}_cov.npy"
        assert mean_file.exists()
        assert cov_file.exists()

        # Check that the function returns the original tuple
        assert isinstance(cached_tuple, tuple)
        assert len(cached_tuple) == 2
        assert np.array_equal(cached_tuple[0], mean)
        assert np.array_equal(cached_tuple[1], cov)

        # Load the tuple from cache
        loaded_tuple = load_cache(name)

        # Check that the loaded tuple matches the original
        assert isinstance(loaded_tuple, tuple)
        assert len(loaded_tuple) == 2
        assert np.array_equal(loaded_tuple[0], mean)
        assert np.array_equal(loaded_tuple[1], cov)


def test_check_cache(temp_cache_dir):
    """Test checking if a cache file exists."""
    # Mock the cache directory
    with patch("ttsds.util.cache.CACHE_DIR", Path(temp_cache_dir)):
        # Import functions here after patching
        from ttsds.util.cache import cache, load_cache, check_cache

        # Create a test array
        test_array = np.random.randn(10, 5)

        # Cache the array
        name_exists = "existing_array"
        cache(test_array, name_exists)

        # Check that the cache file exists
        assert check_cache(name_exists)

        # Check that a non-existent cache file doesn't exist
        assert not check_cache("nonexistent_array")

        # Test with tuple cache
        mean = np.random.randn(5)
        cov = np.random.randn(5, 5)
        tuple_arrays = (mean, cov)

        name_tuple = "existing_tuple"
        cache(tuple_arrays, name_tuple)

        # Check that the tuple cache exists
        assert check_cache(name_tuple)


def test_load_cache_error(temp_cache_dir):
    """Test error handling when loading a non-existent cache file."""
    # Mock the cache directory
    with patch("ttsds.util.cache.CACHE_DIR", Path(temp_cache_dir)):
        # Import function after patching
        from ttsds.util.cache import load_cache

        # Try to load a non-existent cache file
        with pytest.raises(Exception):
            load_cache("nonexistent_file")
