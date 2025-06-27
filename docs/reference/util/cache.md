---
layout: default
title: Caching Utilities
---

# Caching Utilities

TTSDS includes caching utilities to speed up repeated benchmark runs by storing and retrieving intermediate computation results.

## Functions

### `cache(obj, name)`

Cache a numpy array to disk.

```python
from ttsds.util.cache import cache
import numpy as np

# Create a numpy array
data = np.array([1.0, 2.0, 3.0])

# Cache the array
cache(data, "my_data")
```

### `load_cache(name)`

Load a cached numpy array from disk.

```python
from ttsds.util.cache import load_cache

# Load cached data
data = load_cache("my_data")
```

### `check_cache(name)`

Check if a cache file exists.

```python
from ttsds.util.cache import check_cache

# Check if cache exists
if check_cache("my_data"):
    data = load_cache("my_data")
else:
    # Compute data
    pass
```

### `hash_md5(obj)`

Generate an MD5 hash for an object.

```python
from ttsds.util.cache import hash_md5

# Create a hash for an object
obj_hash = hash_md5(my_object)
```

## Configuration

The cache directory is configured using the `TTSDS_CACHE_DIR` environment variable:

```bash
export TTSDS_CACHE_DIR=/path/to/cache
```

If not set, it defaults to `~/.cache/ttsds`.

## Cache Structure

The cache directory structure is organized as follows:

```
~/.cache/ttsds/
├── benchmarks/
│   ├── benchmark_name/
│   │   └── hash_values.npy
└── other_cached_data.npy
```

Each benchmark stores its computed distributions in its own subdirectory, using hash values to uniquely identify datasets. 