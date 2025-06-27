---
layout: default
title: Distances Utilities
---

# Distances Utilities

This module provides functions for calculating distances between distributions, which are used in benchmarks to compare feature distributions between synthetic and reference speech.

## Distribution Distance Metrics

```python
from ttsds.util.distances import wasserstein_distance, frechet_distance, frechet_distance_fast
```

### `wasserstein_distance(x, y)`

Calculates the 2-Wasserstein distance between two 1D distributions.

```python
import numpy as np
from ttsds.util.distances import wasserstein_distance

# Create two 1D distributions
x = np.random.normal(0, 1, 1000)
y = np.random.normal(0.5, 1.2, 1000)

# Calculate Wasserstein distance
dist = wasserstein_distance(x, y)
print(f"Wasserstein distance: {dist:.4f}")
```

### `frechet_distance(x, y, eps=1e-6)`

Calculates the Fréchet distance between two multivariate Gaussian distributions.

The Fréchet distance (also known as Wasserstein-2 distance) measures the similarity between two probability distributions over a feature space.

```python
from ttsds.util.distances import frechet_distance
import numpy as np

# Create two distributions (samples)
x = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 1000)
y = np.random.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], 1000)

# Calculate Fréchet distance
dist = frechet_distance(x, y)
print(f"Fréchet distance: {dist:.4f}")

# Or with pre-computed statistics
mu1 = np.mean(x, axis=0)
sigma1 = np.cov(x, rowvar=False)
mu2 = np.mean(y, axis=0)
sigma2 = np.cov(y, rowvar=False)

dist = frechet_distance((mu1, sigma1), (mu2, sigma2))
```

### `frechet_distance_fast(x, y, eps=1e-6)`

A faster implementation of the Fréchet distance using optimized matrix operations.

```python
from ttsds.util.distances import frechet_distance_fast

# Calculate Fréchet distance with the faster implementation
dist = frechet_distance_fast(x, y)
print(f"Fréchet distance (fast): {dist:.4f}")
```

## Parallel Distance Computation

For efficiently computing multiple distances, use the `DistanceCalculator` class:

```python
from ttsds.util.parallel_distances import DistanceCalculator

# Create a calculator
calculator = DistanceCalculator(n_workers=4)

# Compute multiple distances
distances = calculator.compute_distances(
    target_distribution=x,
    comparison_distributions=[y1, y2, y3],
    dimension_type="N_DIMENSIONAL",
    names=["dist1", "dist2", "dist3"]
)

print(distances["dist1"])
``` 