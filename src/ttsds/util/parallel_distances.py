"""
Utilities for parallel computation of distribution distances.

This module provides the DistanceCalculator class, which handles efficient
computation of Wasserstein and Fréchet distances between multiple distributions.
It supports both sequential and potentially parallel execution.
"""

import numpy as np
from scipy import linalg
import time
from typing import Tuple, List, Dict, Any, Union, Callable, Optional, TypeVar, cast

from ttsds.util.distances import wasserstein_distance as _wasserstein_distance
from ttsds.util.distances import frechet_distance as _frechet_distance

# Type aliases for better readability
Distribution1D = np.ndarray
DistributionND = Tuple[np.ndarray, np.ndarray]  # (mean, covariance)
AnyDistribution = Union[Distribution1D, DistributionND]


class DistanceCalculator:
    """
    Class for computation of distribution distances.

    This class provides methods to compute various distance metrics between
    probability distributions, with support for both one-dimensional and
    multi-dimensional distributions.

    Attributes:
        logger (Optional[Callable[[str], None]]): Function for logging messages
        n_workers (int): Number of worker processes for parallel computation
    """

    def __init__(
        self, logger: Optional[Callable[[str], None]] = None, n_workers: int = 1
    ):
        """
        Initialize the DistanceCalculator.

        Args:
            logger: Optional logger function that takes a string message
            n_workers: Number of worker processes for parallel computation
        """
        self.logger = logger
        self.n_workers = n_workers

        if self.logger:
            mode = "parallel" if n_workers > 1 else "sequential"
            self.logger(
                f"Initialized DistanceCalculator in {mode} mode with {n_workers} workers"
            )

    def _log(self, message: str) -> None:
        """
        Log a message if a logger is available.

        Args:
            message: The message to log
        """
        if self.logger:
            self.logger(message)

    def _compute_wasserstein_distance(
        self, args: Tuple[Distribution1D, Distribution1D]
    ) -> float:
        """
        Compute Wasserstein distance between two one-dimensional distributions.

        Args:
            args: Tuple containing (distribution1, distribution2)

        Returns:
            float: Wasserstein distance
        """
        x, y = args
        return _wasserstein_distance(x, y)

    def _compute_frechet_distance(
        self, args: Tuple[AnyDistribution, AnyDistribution]
    ) -> float:
        """
        Compute Fréchet distance between two multi-dimensional distributions.

        Args:
            args: Tuple containing (distribution1, distribution2)
                  Each distribution can be either samples or (mean, covariance) tuple

        Returns:
            float: Fréchet distance
        """
        x, y = args
        return _frechet_distance(x, y)

    def compute_distances(
        self,
        target_distribution: AnyDistribution,
        comparison_distributions: List[AnyDistribution],
        dimension_type: str = "N_DIMENSIONAL",
        names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute distances between target distribution and multiple comparison distributions.

        This method computes either Wasserstein distances (for one-dimensional data)
        or Fréchet distances (for multi-dimensional data) between the target
        distribution and each of the comparison distributions.

        Args:
            target_distribution: The target distribution to compare against
            comparison_distributions: List of distributions to compare with target
            dimension_type: Type of dimension, either "ONE_DIMENSIONAL" or "N_DIMENSIONAL"
            names: Optional list of names for comparison distributions (defaults to indices)

        Returns:
            Dict[str, float]: Dictionary mapping names (or indices) to distances

        Raises:
            ValueError: If dimension_type is not recognized
        """
        start_time = time.time()

        # Prepare distribution pairs
        distribution_pairs = [
            (target_distribution, dist) for dist in comparison_distributions
        ]

        # Choose distance function based on dimension type
        if dimension_type == "ONE_DIMENSIONAL":
            distance_func = self._compute_wasserstein_distance
            self._log(f"Computing {len(distribution_pairs)} Wasserstein distances")
        elif dimension_type == "N_DIMENSIONAL":
            distance_func = self._compute_frechet_distance
            self._log(f"Computing {len(distribution_pairs)} Fréchet distances")
        else:
            raise ValueError(f"Unknown dimension type: {dimension_type}")

        # Compute distances sequentially
        # TODO: Add parallel computation support if self.n_workers > 1
        distances = [distance_func(pair) for pair in distribution_pairs]

        # Map distances to names or indices
        if names is None:
            names = [str(i) for i in range(len(comparison_distributions))]

        result = {name: distance for name, distance in zip(names, distances)}

        elapsed = time.time() - start_time
        self._log(f"Computed {len(distances)} distances in {elapsed:.2f} seconds")

        return result
