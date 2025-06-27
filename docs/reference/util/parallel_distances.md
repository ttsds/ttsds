---
layout: default
title: Parallel Distances
---

# Parallel Distances

This module provides a class for calculating distances between distributions efficiently, with support for both sequential and potentially parallel execution.

## DistanceCalculator Class

```python
from ttsds.util.parallel_distances import DistanceCalculator
```

The `DistanceCalculator` class provides methods to compute various distance metrics between probability distributions, with support for both one-dimensional and multi-dimensional distributions.

### Initialization

```python
# Create a calculator with default settings
calculator = DistanceCalculator()

# Create a calculator with custom settings
calculator = DistanceCalculator(
    logger=my_logger_function,
    n_workers=4
)
```

#### Parameters

- `logger`: Optional function that takes a string message for logging (default: None)
- `n_workers`: Number of worker processes for parallel computation (default: 1)

### Methods

#### `compute_distances()`

Compute distances between a target distribution and multiple comparison distributions.

```python
import numpy as np
from ttsds.util.parallel_distances import DistanceCalculator

# Create distributions
target_dist = np.random.normal(0, 1, 1000)  # 1D distribution
comparison_dists = [
    np.random.normal(0.2, 1, 1000),
    np.random.normal(0.5, 1.2, 1000),
    np.random.normal(-0.3, 0.8, 1000)
]

# Create calculator
calculator = DistanceCalculator()

# Compute Wasserstein distances for 1D distributions
results = calculator.compute_distances(
    target_distribution=target_dist,
    comparison_distributions=comparison_dists,
    dimension_type="ONE_DIMENSIONAL",
    names=["dist1", "dist2", "dist3"]
)
print(results)  # {'dist1': 0.123, 'dist2': 0.456, 'dist3': 0.789}

# For multi-dimensional distributions
target_2d = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 1000)
comparison_2d = [
    np.random.multivariate_normal([1, 0], [[1, 0.2], [0.2, 1]], 1000)
]

# Compute Fréchet distances for multi-dimensional distributions
results_2d = calculator.compute_distances(
    target_distribution=target_2d,
    comparison_distributions=comparison_2d,
    dimension_type="N_DIMENSIONAL",
    names=["dist_2d"]
)
```

#### Parameters

- `target_distribution`: The target distribution to compare against
- `comparison_distributions`: List of distributions to compare with target
- `dimension_type`: Type of dimension, either "ONE_DIMENSIONAL" or "N_DIMENSIONAL"
- `names`: Optional list of names for comparison distributions (defaults to indices)

#### Returns

- Dictionary mapping names (or indices) to distance values

## Implementation Details

The DistanceCalculator internally uses the `wasserstein_distance` function from the `ttsds.util.distances` module for one-dimensional distributions and the `frechet_distance` function for multi-dimensional distributions.

Future versions of this module may implement true parallel computation for improved performance on multi-core systems. 