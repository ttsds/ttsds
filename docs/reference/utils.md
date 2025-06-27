---
layout: default
title: Utility Modules
---

# Utility Modules

TTSDS includes several utility modules that provide functionality for dataset handling, caching, feature extraction, and statistical calculations.

## Dataset Handling

The dataset utilities provide classes and functions for loading and managing audio datasets.

```python
from ttsds.util.dataset import DirectoryDataset, TarDataset, WavListDataset
```

These classes allow you to load audio data from various sources:
- `DirectoryDataset`: Load audio from a directory of WAV files
- `TarDataset`: Load audio from a TAR archive
- `WavListDataset`: Load audio from explicit lists of WAV files

[Read more about dataset utilities](util/dataset.md)

## Caching

The caching utilities provide functions for storing and retrieving intermediate computation results.

```python
from ttsds.util.cache import cache, load_cache, check_cache, hash_md5
```

These functions help speed up repeated computations by caching results to disk.

[Read more about caching utilities](util/cache.md)

## Distance Metrics

The distance metric utilities provide functions for calculating distances between distributions.

```python
from ttsds.util.distances import wasserstein_distance, frechet_distance
```

These functions measure the similarity between probability distributions, which is essential for comparing synthetic and reference speech features.

[Read more about distance metrics](util/distances.md)

## Parallel Distance Computation

The parallel distance utilities provide functions for efficient computation of distances.

```python
from ttsds.util.parallel_distances import DistanceCalculator
```

The `DistanceCalculator` class handles computing distances between multiple distributions efficiently.

[Read more about parallel distance utilities](util/parallel_distances.md)

## MPM (McLeod Pitch Method)

The MPM utilities provide functions for pitch detection using the McLeod Pitch Method.

```python
from ttsds.util.mpm import pitch_mpm
from ttsds.util.mpm_modules import nsdf, peak_picking
```

These modules implement the McLeod Pitch Method for accurate pitch detection in speech signals.

[Read more about MPM utilities](util/mpm.md)  
[Read more about MPM modules](util/mpm_modules.md)

## Environment Variables

TTSDS uses the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `TTSDS_CACHE_DIR` | Directory for cached data | `~/.cache/ttsds`