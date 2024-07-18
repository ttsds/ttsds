import os
from pathlib import Path
from hashlib import md5

import numpy as np

CACHE_DIR = os.getenv("TTSDS_CACHE_DIR", os.path.expanduser("~/.cache/ttsds"))
CACHE_DIR = Path(CACHE_DIR)
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def hash_md5(obj) -> str:
    """
    Hash an object.

    Args:
        obj: The object to hash.

    Returns:
        str: The hash of the object.
    """
    h = md5()
    h.update(str(obj.__hash__()).encode())
    return h.hexdigest()


def cache(obj: np.ndarray, name: str) -> np.ndarray:
    """
    Cache a numpy array to disk.

    Args:
        obj (np.ndarray): The numpy array to cache.
        name (str): The name of the cache file.

    Returns:
        np.ndarray: The cached numpy array.
    """
    cache_file = CACHE_DIR / f"{name}.npy"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, obj)
    return obj


def load_cache(name: str) -> np.ndarray:
    """
    Load a cached numpy array from disk.

    Args:
        name (str): The name of the cache file.

    Returns:
        np.ndarray: The cached numpy array.
    """
    cache_file = CACHE_DIR / f"{name}.npy"
    return np.load(cache_file)


def check_cache(name: str) -> bool:
    """
    Check if a cache file exists.

    Args:
        name (str): The name of the cache file.

    Returns:
        bool: True if the cache file exists, False otherwise.
    """
    cache_file = CACHE_DIR / f"{name}.npy"
    return cache_file.exists()
