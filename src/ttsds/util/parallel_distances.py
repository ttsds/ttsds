"""
This module contains a class for computation of distribution distances.
"""

import numpy as np
from scipy import linalg
import time
from typing import Tuple, List, Dict, Any, Union, Callable

from ttsds.util.distances import wasserstein_distance as _wasserstein_distance
from ttsds.util.distances import frechet_distance as _frechet_distance


class DistanceCalculator:
    """
    Class for computation of distribution distances.
    """

    def __init__(self, logger: Callable = None):
        """
        Initialize the DistanceCalculator.

        Args:
            logger: Optional logger function.
        """
        self.logger = logger

        if self.logger:
            self.logger("Initialized DistanceCalculator in sequential mode")

    def _log(self, message: str):
        """Log a message if a logger is available."""
        if self.logger:
            self.logger(message)

    def _compute_wasserstein_distance(
        self, args: Tuple[np.ndarray, np.ndarray]
    ) -> float:
        """
        Compute Wasserstein distance between two distributions.

        Args:
            args: Tuple containing (distribution1, distribution2)

        Returns:
            float: Wasserstein distance
        """
        x, y = args
        return _wasserstein_distance(x, y)

    def _compute_frechet_distance(
        self, args: Tuple[Union[np.ndarray, Tuple], Union[np.ndarray, Tuple]]
    ) -> float:
        """
        Compute Frechet distance between two distributions.

        Args:
            args: Tuple containing (distribution1, distribution2)

        Returns:
            float: Frechet distance
        """
        x, y = args
        return _frechet_distance(x, y)

    def compute_distances(
        self,
        target_distribution: Union[np.ndarray, Tuple],
        comparison_distributions: List[Union[np.ndarray, Tuple]],
        dimension_type: str = "N_DIMENSIONAL",
        names: List[str] = None,
    ) -> Dict[str, float]:
        """
        Compute distances between target distribution and multiple comparison distributions.

        Args:
            target_distribution: The target distribution
            comparison_distributions: List of distributions to compare against
            dimension_type: Type of dimension, either "ONE_DIMENSIONAL" or "N_DIMENSIONAL"
            names: Optional list of names for comparison distributions

        Returns:
            Dict[str, float]: Dictionary mapping names (or indices) to distances
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
        else:
            distance_func = self._compute_frechet_distance
            self._log(f"Computing {len(distribution_pairs)} Frechet distances")

        # Compute distances sequentially
        distances = [distance_func(pair) for pair in distribution_pairs]

        # Map distances to names or indices
        if names is None:
            names = [str(i) for i in range(len(comparison_distributions))]

        result = {name: distance for name, distance in zip(names, distances)}

        elapsed = time.time() - start_time
        self._log(f"Computed {len(distances)} distances in {elapsed:.2f} seconds")

        return result
