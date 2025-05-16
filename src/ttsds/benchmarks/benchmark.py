"""
This file contains the Benchmark abstract class.
"""

from abc import ABC, abstractmethod
from enum import Enum
import hashlib
import json
from typing import List, Union, Optional, Dict, Tuple

import numpy as np

from ttsds.util.dataset import Dataset, DataDistribution
from ttsds.util.cache import cache, load_cache, check_cache, hash_md5
from ttsds.util.distances import wasserstein_distance, frechet_distance
from ttsds.util.parallel_distances import DistanceCalculator


class BenchmarkCategory(Enum):
    """
    Enum class for the different categories of benchmarks.
    """

    GENERIC = 1
    PROSODY = 2
    ENVIRONMENT = 3
    SPEAKER = 4
    INTELLIGIBILITY = 5


class BenchmarkDimension(Enum):
    """
    Enum class for the different dimensions of benchmarks.
    """

    ONE_DIMENSIONAL = 1
    N_DIMENSIONAL = 2


class DeviceSupport(Enum):
    """
    Enum class for the different device support of benchmarks.
    """

    CPU = 1
    GPU = 2


class Benchmark(ABC):
    """
    Abstract class for a benchmark.
    """

    def __init__(
        self,
        name: str,
        category: BenchmarkCategory,
        dimension: BenchmarkDimension,
        description: str,
        version: Optional[str] = None,
        supported_devices: List[DeviceSupport] = [DeviceSupport.CPU],
        **kwargs,
    ):
        self.name = name
        self.key = name.lower().replace(" ", "_")
        self.category = category
        self.dimension = dimension
        self.description = description
        self.version = version
        self.kwargs = kwargs
        self.supported_devices = supported_devices
        self.device = "cpu"
        self.logger = None  # Will be set by BenchmarkSuite

    def log(self, message: str):
        """
        Log a message using the suite's logger if available.

        Args:
            message (str): The message to log
        """
        if self.logger:
            self.logger(f"[{self.category.name}] [{self.name}] {message}")

    def set_logger(self, logger_func):
        """
        Set the logger function for this benchmark.

        Args:
            logger_func: A function that takes a message string as input
        """
        self.logger = logger_func

    def get_distribution(self, dataset: Union[Dataset, DataDistribution]) -> np.ndarray:
        """
        Abstract method to get the distribution of the benchmark.
        If the benchmark is one-dimensional, the method should return a
        numpy array with the values of the benchmark for each sample in the dataset.
        If the benchmark is n-dimensional, the method should return a numpy array
        with the values of the benchmark for each sample in the dataset, where each
        row corresponds to a sample and each column corresponds to a dimension of the benchmark.
        """
        ds_hash = hash_md5(dataset)
        benchmark_hash = hash_md5(self)
        cache_name = f"benchmarks/{self.name}/{ds_hash}_{benchmark_hash}"

        # Check cache and return if available
        if check_cache(cache_name):
            self.log(f"Cache hit for {dataset.name}")
            result = load_cache(cache_name)
            if result is not None:
                return result
        else:
            self.log(f"Cache miss for {dataset.name}")

        if check_cache(cache_name + "_mu") and check_cache(cache_name + "_sig"):
            self.log(f"Cache hit for mu/sigma distribution of {dataset.name}")
            mu = load_cache(cache_name + "_mu")
            sig = load_cache(cache_name + "_sig")
            if mu is not None and sig is not None:
                return (mu, sig)

        # Handle data distribution objects
        if (
            isinstance(dataset, DataDistribution)
            and self.dimension == BenchmarkDimension.N_DIMENSIONAL
        ):
            self.log(
                f"Getting N-dimensional distribution from DataDistribution for {dataset.name}"
            )
            mu, sig = dataset.get_distribution(self.key)
            cache(mu, cache_name + "_mu")
            cache(sig, cache_name + "_sig")
            return (mu, sig)
        elif (
            isinstance(dataset, DataDistribution)
            and self.dimension == BenchmarkDimension.ONE_DIMENSIONAL
        ):
            self.log(
                f"Getting 1-dimensional distribution from DataDistribution for {dataset.name}"
            )
            distribution = dataset.get_distribution(self.key)
            cache(distribution, cache_name)
            return distribution

        # Calculate distribution
        try:
            self.log(f"Calculating distribution for {dataset.name}")
            distribution = self._get_distribution(dataset)
            self.log(f"Distribution calculation complete for {dataset.name}")
        except Exception as e:
            self.log(f"Error calculating distribution for {dataset.name}: {str(e)}")
            print("error with", dataset)
            raise e

        # Cache the result
        cache(distribution, cache_name)
        return distribution

    @abstractmethod
    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Abstract method to get the distribution of the benchmark.
        """
        raise NotImplementedError

    def to_device(self, device: str):
        """
        Move the benchmark to a device.
        """
        if device not in ["cpu", "cuda"]:
            raise ValueError("Invalid device")
        if self.supported_devices == [DeviceSupport.CPU]:
            if device == "cuda":
                raise ValueError("Benchmark does not support CUDA")
        self._to_device(device)

    def _to_device(self, device: str):
        """
        Abstract method to move the benchmark to a device.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.category.name}/{self.name}"

    def __repr__(self):
        return f"{self.category.name}/{self.name}"

    def __hash__(self) -> int:
        h = hashlib.md5()
        h.update(self.name.encode())
        h.update(self.category.name.encode())
        h.update(self.dimension.name.encode())
        h.update(self.description.encode())
        if self.version is not None:
            h.update(self.version.encode())
        # convert the kwargs to strings
        kwargs_str = {
            k: str(v) if not isinstance(v, dict) else json.dumps(v, sort_keys=True)
            for k, v in self.kwargs.items()
        }
        h.update(json.dumps(kwargs_str, sort_keys=True).encode())
        return int(h.hexdigest(), 16)

    def compute_distance(
        self,
        one_dataset: Union[Dataset, DataDistribution],
        other_dataset: Union[Dataset, DataDistribution],
        distance_calculator: Optional[DistanceCalculator] = None,
    ) -> float:
        """
        Compute the distance between the distributions of the benchmark in two datasets.

        Args:
            one_dataset: First dataset
            other_dataset: Second dataset
            distance_calculator: Optional distance calculator for parallel computation

        Returns:
            float: Distance between the distributions
        """
        one_distribution = self.get_distribution(one_dataset)
        other_distribution = self.get_distribution(other_dataset)

        if distance_calculator is not None:
            # Use the distance calculator
            result = distance_calculator.compute_distances(
                one_distribution,
                [other_distribution],
                dimension_type=self.dimension.name,
                names=["distance"],
            )
            return result["distance"]
        else:
            # Fall back to original implementation
            if self.dimension == BenchmarkDimension.ONE_DIMENSIONAL:
                return wasserstein_distance(one_distribution, other_distribution)
            elif self.dimension == BenchmarkDimension.N_DIMENSIONAL:
                return frechet_distance(one_distribution, other_distribution)
            else:
                raise ValueError("Invalid benchmark dimension")

    def compute_score(
        self,
        dataset: Dataset,
        reference_datasets: List[Dataset],
        noise_datasets: List[Dataset],
    ) -> Tuple[float, Tuple[str, str]]:
        """
        Compute the score of the benchmark on a dataset using parallel computation.

        Args:
            dataset: The dataset to compute the score for
            reference_datasets: List of reference datasets
            noise_datasets: List of noise datasets
            n_workers: Number of worker processes to use (default: auto)

        Returns:
            Tuple containing:
                - score (float): The computed score
                - Tuple of (closest_noise_name, closest_reference_name)
        """
        # Create a distance calculator with logging
        distance_calculator = DistanceCalculator(logger=self.log)

        # Step 1: Get the distribution for the target dataset
        target_distribution = self.get_distribution(dataset)

        # Step 2: Get all noise distributions
        self.log(
            f"Computing scores against {len(noise_datasets)} noise datasets for {dataset.name}"
        )
        noise_distributions = []
        noise_names = []

        for noise_ds in noise_datasets:
            self.log(f"Getting distribution for noise dataset: {noise_ds.name}")
            dist = self.get_distribution(noise_ds)
            noise_distributions.append(dist)
            noise_names.append(noise_ds.name)

        # Step 3: Compute all noise distances in parallel
        noise_results = distance_calculator.compute_distances(
            target_distribution,
            noise_distributions,
            dimension_type=self.dimension.name,
            names=noise_names,
        )

        # Step 4: Get all reference distributions
        self.log(
            f"Computing scores against {len(reference_datasets)} reference datasets for {dataset.name}"
        )
        reference_distributions = []
        reference_names = []

        for ref_ds in reference_datasets:
            self.log(f"Getting distribution for reference dataset: {ref_ds.name}")
            dist = self.get_distribution(ref_ds)
            reference_distributions.append(dist)
            reference_names.append(ref_ds.name)

        # Step 5: Compute all reference distances in parallel
        reference_results = distance_calculator.compute_distances(
            target_distribution,
            reference_distributions,
            dimension_type=self.dimension.name,
            names=reference_names,
        )

        # Find closest noise and reference
        closest_noise_name = min(noise_results, key=noise_results.get)
        closest_reference_name = min(reference_results, key=reference_results.get)

        # Get scores
        noise_score = noise_results[closest_noise_name]
        dataset_score = reference_results[closest_reference_name]

        # Calculate final score
        combined_score = dataset_score + noise_score
        score = (noise_score / combined_score) * 100

        # Log results
        self.log(f"Final score for {dataset.name}: {score:.2f}")
        self.log(f"Closest noise: {closest_noise_name} ({noise_score:.4f})")
        self.log(f"Closest reference: {closest_reference_name} ({dataset_score:.4f})")

        return (
            score,
            (
                closest_noise_name,
                closest_reference_name,
            ),
        )

    def log_progress(
        self, current: int, total: int, dataset_name: str, interval: int = 10
    ):
        """
        Log progress during distribution computation.

        Args:
            current: Current item being processed
            total: Total number of items to process
            dataset_name: Name of the dataset being processed
            interval: Only log every interval percent
        """
        # Only log at regular intervals to avoid flooding
        percent = int((current / total) * 100)
        if percent % interval == 0 and current > 0:
            self.log(
                f"Processing {dataset_name}: {percent}% complete ({current}/{total})"
            )
