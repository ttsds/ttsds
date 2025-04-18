"""
This file contains the Benchmark abstract class.
"""

from abc import ABC, abstractmethod
from enum import Enum
import hashlib
import json
from typing import List, Union, Optional

import numpy as np

from ttsds.util.dataset import Dataset, DataDistribution
from ttsds.util.cache import cache, load_cache, check_cache, hash_md5
from ttsds.util.distances import wasserstein_distance, frechet_distance


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
        if check_cache(cache_name):
            result = load_cache(cache_name)
            if result is not None:
                return result
        if check_cache(cache_name + "_mu") and check_cache(cache_name + "_sig"):
            mu = load_cache(cache_name + "_mu")
            sig = load_cache(cache_name + "_sig")
            if mu is not None and sig is not None:
                return (mu, sig)
        if (
            isinstance(dataset, DataDistribution)
            and self.dimension == BenchmarkDimension.N_DIMENSIONAL
        ):
            mu, sig = dataset.get_distribution(self.key)
            cache(mu, cache_name + "_mu")
            cache(sig, cache_name + "_sig")
            return (mu, sig)
        elif (
            isinstance(dataset, DataDistribution)
            and self.dimension == BenchmarkDimension.ONE_DIMENSIONAL
        ):
            distribution = dataset.get_distribution(self.key)
            cache(distribution, cache_name)
            return distribution
        distribution = self._get_distribution(dataset)
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
    ) -> float:
        """
        Compute the distance between the distributions of the benchmark in two datasets.
        """
        one_distribution = self.get_distribution(one_dataset)
        other_distribution = self.get_distribution(other_dataset)
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
    ) -> float:
        """
        Compute the score of the benchmark on a dataset.
        """
        noise_scores = []
        for noise_ds in noise_datasets:
            score = self.compute_distance(noise_ds, dataset)
            noise_scores.append(score)
        noise_scores = np.array(noise_scores)

        dataset_scores = []
        for ref_ds in reference_datasets:
            score = self.compute_distance(ref_ds, dataset)
            dataset_scores.append(score)
        dataset_scores = np.array(dataset_scores)

        closest_noise_idx = np.argmin(noise_scores)
        closest_dataset_idx = np.argmin(dataset_scores)

        noise_score = np.min(noise_scores)
        dataset_score = np.min(dataset_scores)
        combined_score = dataset_score + noise_score
        score = (noise_score / combined_score) * 100
        return (
            score,
            (
                noise_datasets[closest_noise_idx].name,
                reference_datasets[closest_dataset_idx].name,
            ),
        )
