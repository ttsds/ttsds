"""
Dataset classes for handling audio data from different sources.

This module provides a set of classes for working with audio datasets:
- Dataset: Abstract base class for all dataset types
- DirectoryDataset: Dataset from a directory of WAV files
- TarDataset: Dataset from a TAR archive containing WAV files
- WavListDataset: Dataset from explicit lists of WAV and text files
- DataDistribution: Helper class to compute distributions across datasets
"""

from abc import ABC, abstractmethod
from copy import deepcopy
import hashlib
from pathlib import Path
import tarfile
from typing import Tuple, List, Dict, Union, Optional, Any, Iterator, TypeVar, cast
import pickle
import gzip

import numpy as np
import librosa
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.console import Console

from ttsds.util.cache import cache, check_cache, load_cache, hash_md5

# Type variable for self-referencing return types
T = TypeVar("T", bound="Dataset")


class Dataset(ABC):
    """
    Abstract base class for all dataset types.

    This class defines the interface that all dataset implementations must follow
    and provides common functionality for dataset operations.

    Attributes:
        sample_rate (int): Sampling rate for audio in Hz
        wavs (List[Any]): List of audio file references
        has_text (bool): Whether this dataset has associated text
        texts (List[Any]): List of text file references (if has_text is True)
        sample_params (Dict[str, Any]): Parameters used for sampling
        name (str): Name of the dataset
        indices (Optional[np.ndarray]): Indices for sampling subset of data
        _progress (Optional[Progress]): Progress bar for dataset iteration
        _progress_task: Task ID for the progress bar
    """

    def __init__(self, name: str, sample_rate: int = 22050, has_text: bool = False):
        """
        Initialize a dataset.

        Args:
            name: Name of the dataset
            sample_rate: Sampling rate for audio in Hz
            has_text: Whether this dataset has associated text
        """
        self.sample_rate = sample_rate
        self.wavs: List[Any] = []
        self.has_text = has_text
        if has_text:
            self.texts: List[Any] = []
        self.sample_params: Dict[str, Optional[int]] = {
            "n": None,
            "seed": None,
        }
        self.name = name
        self.indices: Optional[np.ndarray] = None
        self._progress: Optional[Progress] = None
        self._progress_task = None

    def _setup_progress(
        self, total: int, description: str = "Processing dataset"
    ) -> None:
        """
        Setup progress tracking for dataset iteration.

        Args:
            total: Total number of items to process
            description: Description for the progress bar
        """
        if self._progress is None:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=Console(),
                refresh_per_second=1,
            )
            self._progress.start()
            self._progress_task = self._progress.add_task(description, total=total)

    def _update_progress(self, advance: int = 1) -> None:
        """
        Update the progress bar.

        Args:
            advance: Number of steps to advance
        """
        if self._progress and self._progress_task is not None:
            self._progress.update(self._progress_task, advance=advance)

    def _stop_progress(self) -> None:
        """
        Stop and cleanup the progress bar.
        """
        if self._progress:
            self._progress.stop()
            self._progress = None
            self._progress_task = None

    def __enter__(self) -> "Dataset":
        """
        Context manager entry.

        Returns:
            self: The dataset instance
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Context manager exit.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        self._stop_progress()

    def iter_with_progress(
        self, benchmark: "Benchmark"
    ) -> Iterator[Tuple[np.ndarray, Optional[str]]]:
        """
        Iterate over the dataset with a progress bar.

        Args:
            benchmark: The benchmark being run on this dataset

        Yields:
            Tuple containing audio data and optional text
        """
        # e.g. LibriTTS -> PROSODY -> MPM (with different colors for different categories)
        if benchmark.category.name == "PROSODY":
            color = "cyan"
        elif benchmark.category.name == "SPEAKER":
            color = "green"
        elif benchmark.category.name == "ENVIRONMENT":
            color = "blue"
        elif benchmark.category.name == "INTELLIGIBILITY":
            color = "magenta"
        elif benchmark.category.name == "GENERIC":
            color = "yellow"
        description = f"{self.name} -> {benchmark.category.name} -> {benchmark.name}"
        self._setup_progress(len(self), description)
        try:
            for i in range(len(self)):
                yield self[i]
                self._update_progress()
        finally:
            self._stop_progress()

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[str]]:
        """
        Get a sample from the dataset.

        Args:
            idx: The index of the sample.

        Returns:
            Tuple containing:
                - np.ndarray: The audio data
                - Optional[str]: The text of the sample (None if has_text is False)
        """
        raise NotImplementedError

    def sample(self, n: int, seed: int = 42) -> T:
        """
        Sample n samples from the dataset.

        Args:
            n: The number of samples to sample.
            seed: The seed for the random number generator.

        Returns:
            A new dataset containing only the sampled items
        """
        rng = np.random.default_rng(seed)
        self.indices = rng.choice(len(self), size=n, replace=False)
        self.sample_params = {"n": n, "seed": seed}
        return self

    def __repr__(self) -> str:
        """
        String representation of the dataset.

        Returns:
            str: String representation including name and sample info
        """
        sample_info = ""
        if self.sample_params["n"] is not None:
            sample_info = f", samples={self.sample_params['n']} (seed={self.sample_params['seed']})"
        return f"{self.name}{sample_info}"


class DirectoryDataset(Dataset):
    """
    A dataset class for a directory containing WAV files and corresponding text files.

    This dataset loads audio data from WAV files in a directory and optionally
    the corresponding text from matching text files.

    Attributes:
        root_dir (Path): Root directory containing the audio files
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        sample_rate: int = 22050,
        has_text: bool = False,
        text_suffix: str = ".txt",
        name: Optional[str] = None,
    ):
        """
        Initialize a directory dataset.

        Args:
            root_dir: Path to the directory containing WAV files
            sample_rate: Sampling rate for audio in Hz
            has_text: Whether to load text files
            text_suffix: Suffix for text files
            name: Optional name for the dataset (defaults to directory name)

        Raises:
            ValueError: If root_dir is None
        """
        if name is not None:
            super().__init__(name, sample_rate, has_text)
        else:
            if root_dir is None:
                raise ValueError("root_dir must be provided.")
            super().__init__(Path(root_dir).name, sample_rate, has_text)
        if root_dir is None:
            raise ValueError("root_dir must be provided.")
        self.root_dir = Path(root_dir)
        # we assume that the root directory contains
        wavs, texts = [], []
        for wav_file in Path(root_dir).rglob("*.wav"):
            wavs.append(wav_file)
            if has_text:
                text = wav_file.with_suffix(text_suffix)
                texts.append(text)
        self.wavs = wavs
        if has_text:
            self.texts = texts

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        if self.indices is not None:
            return len(self.indices)
        return len(self.wavs)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[str]]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple containing:
                - np.ndarray: Audio data
                - Optional[str]: Text data (None if has_text is False)
        """
        if self.indices is not None:
            idx = self.indices[idx]
        wav, sr = self.wavs[idx], self.sample_rate
        str_root = hash_md5(str(self.root_dir)) + "_" + hash_md5(str(wav))
        wav_str = f"{str_root}_{sr}"
        if check_cache(wav_str):
            try:
                audio = load_cache(wav_str)
            except Exception as e:
                print(f"Error loading cache for {wav_str}: {e}")
                audio, _ = librosa.load(wav, sr=self.sample_rate)
                cache(audio, wav_str)
        else:
            audio, _ = librosa.load(wav, sr=self.sample_rate)
            cache(audio, wav_str)
        if self.has_text:
            with open(self.texts[idx], "r", encoding="utf-8") as f:
                text = f.read().replace("\n", "")
        if audio.shape[0] < 16000:
            print(f"(Almost) Empty audio file: {wav}, padding with zeros.")
            audio = np.pad(audio, (0, 16000 - audio.shape[0]))
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        if self.has_text:
            return audio, text
        return audio, None

    def __hash__(self) -> int:
        """
        Compute a hash value for the dataset.

        Returns:
            int: Hash value
        """
        h = hashlib.md5()
        h.update(str(self.__class__).encode())
        h.update(str(self.root_dir).encode())
        h.update(str(self.sample_params["n"]).encode())
        h.update(str(self.sample_params["seed"]).encode())
        h.update(str(self.has_text).encode())
        return int(h.hexdigest(), 16)

    def __repr__(self) -> str:
        """
        String representation of the directory dataset.

        Returns:
            str: String representation
        """
        sample_info = ""
        if self.sample_params["n"] is not None:
            sample_info = f", samples={self.sample_params['n']} (seed={self.sample_params['seed']})"
        return f"({self.root_dir.name}{sample_info})"


class TarDataset(Dataset):
    """
    A dataset class for a TAR archive containing WAV files and corresponding text files.

    This dataset loads audio data from WAV files in a TAR archive and optionally
    the corresponding text from matching text files within the archive.

    Attributes:
        root_tar (str): Path to the TAR archive
        path_prefix (Optional[str]): Prefix for paths within the archive
    """

    def __init__(
        self,
        root_tar: Optional[str] = None,
        sample_rate: int = 22050,
        has_text: bool = False,
        text_suffix: str = ".txt",
        path_prefix: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize a TAR archive dataset.

        Args:
            root_tar: Path to the TAR archive
            sample_rate: Sampling rate for audio in Hz
            has_text: Whether to load text files
            text_suffix: Suffix for text files
            path_prefix: Prefix for paths within the archive
            name: Optional name for the dataset

        Raises:
            ValueError: If root_tar is None
        """
        if name is not None:
            super().__init__(name, sample_rate, has_text)
        else:
            super().__init__(Path(root_tar).name, sample_rate, has_text)
        if root_tar is None:
            raise ValueError("root_tar must be provided.")
        self.root_tar = root_tar
        self.root_dir = Path(root_tar).name
        self.tar = tarfile.open(root_tar)
        wavs, texts = [], []
        for member in self.tar.getmembers():
            if member.name.endswith(".wav"):
                wav_file = Path(member.name)
                wavs.append(wav_file)
            if has_text:
                text_file = Path(member.name).with_suffix(text_suffix)
                texts.append(text_file)
        self.wavs = wavs
        if has_text:
            self.texts = texts
        self.path_prefix = path_prefix

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return len(self.wavs)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        if self.indices is not None:
            idx = self.indices[idx]
        wav, sr = self.wavs[idx], self.sample_rate
        wav_str = f"{Path(self.root_tar).name}_{wav}_{sr}"
        wav_str = wav_str.replace(".", "_")
        wav_str = wav_str.replace("/", "_")
        if check_cache(wav_str):
            audio = load_cache(wav_str)
        else:
            if self.path_prefix is not None:
                wav = self.path_prefix + str(wav)
            else:
                wav = str(wav)
            wav_file = self.tar.extractfile(wav)
            audio, _ = librosa.load(wav_file, sr=self.sample_rate)
            cache(audio, wav_str)
        if self.has_text:
            if self.path_prefix is not None:
                text_f = self.path_prefix + str(self.texts[idx])
            else:
                text_f = str(self.texts[idx])
            text_file = self.tar.extractfile(text_f)
            try:
                text = text_file.read().decode("utf-8")
            except UnicodeDecodeError:
                text = ""
                print(f"Error reading text file: {text_f}")
        if audio.shape[0] == 0:
            print(f"Empty audio file: {wav}, padding with zeros.")
            audio = np.zeros(16000)
        else:
            # remove silence at beginning and end
            audio, _ = librosa.effects.trim(audio)
        if self.has_text:
            return audio, text
        return audio, None

    def __hash__(self) -> int:
        h = hashlib.md5()
        h.update(str(self.__class__).encode())
        h.update(str(self.root_tar).encode())
        h.update(str(self.sample_params["n"]).encode())
        h.update(str(self.sample_params["seed"]).encode())
        h.update(str(self.path_prefix).encode())
        return int(h.hexdigest(), 16)

    def __repr__(self) -> str:
        sample_info = ""
        if self.sample_params["n"] is not None:
            sample_info = f", samples={self.sample_params['n']} (seed={self.sample_params['seed']})"
        return f"({Path(self.root_tar).name}{sample_info})"


class WavListDataset(Dataset):
    """
    A dataset class for a list of wav files and corresponding text files.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        has_text: bool = False,
        wavs: List[Path] = None,
        texts: List[Path] = None,
        name: str = None,
    ):
        if name is not None:
            super().__init__(name, sample_rate)
        else:
            super().__init__("WavListDataset", sample_rate)
        self.wavs = [w.resolve() for w in wavs]
        # sort
        idx = np.argsort([str(w) for w in self.wavs])
        self.wavs = [self.wavs[i] for i in idx]
        if has_text:
            self.texts = [t.resolve() for t in texts]
            self.texts = [self.texts[i] for i in idx]

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return len(self.wavs)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        if self.indices is not None:
            idx = self.indices[idx]
        wav, sr = self.wavs[idx], self.sample_rate
        wav_str = f"{wav}_{sr}"
        if check_cache(wav_str):
            audio = load_cache(wav_str)
        else:
            audio, _ = librosa.load(wav, sr=self.sample_rate)
            cache(audio, wav_str)
        if self.has_text:
            with open(self.texts[idx], "r", encoding="utf-8") as f:
                text = f.read().replace("\n", "")
        if audio.shape[0] == 0:
            print(f"Empty audio file: {wav}, padding with zeros.")
            audio = np.zeros(16000)
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        if self.has_text:
            return audio, text
        return audio, None

    def __hash__(self) -> int:
        h = hashlib.md5()
        h.update(str(self.__class__).encode())
        h.update(str(self.sample_params["n"]).encode())
        h.update(str(self.sample_params["seed"]).encode())
        h.update(str(self.wavs).encode())
        if self.has_text:
            h.update(str(self.texts).encode())
        h.update(str(self.has_text).encode())
        return int(h.hexdigest(), 16)

    def __repr__(self) -> str:
        sample_info = ""
        if self.sample_params["n"] is not None:
            sample_info = f", samples={self.sample_params['n']} (seed={self.sample_params['seed']})"
        return f"({self.name}{sample_info})"


class DataDistribution:
    def __init__(
        self,
        dataset: Dataset = None,
        benchmarks: Dict[str, "Benchmark"] = None,
        name: str = None,
    ):
        if name is not None:
            self.name = name
        elif dataset is not None:
            self.name = dataset.name
        self.benchmarks = benchmarks
        self.benchmark_results = {}
        if dataset is not None:
            self.dataset = dataset
            self.run()

    def run(self):
        for benchmark in self.benchmarks:
            print(f"Running {benchmark} on {self.dataset.root_dir}")
            bench = self.benchmarks[benchmark]
            dist = bench.get_distribution(self.dataset)
            if bench.dimension.name == "N_DIMENSIONAL":
                # compute mu and sigma and store as tuple
                mu = np.mean(dist, axis=0)
                sigma = np.cov(dist, rowvar=False)
                dist = (mu, sigma)
            self.benchmark_results[benchmark] = dist

    def _run_benchmark(self, benchmark: "Benchmark"):
        dist = benchmark.get_distribution(self.dataset)
        if benchmark.dimension.name == "N_DIMENSIONAL":
            # compute mu and sigma and store as tuple
            mu = np.mean(dist, axis=0)
            sigma = np.cov(dist, rowvar=False)
            dist = (mu, sigma)
        return dist

    def get_distribution(self, benchmark_name: str) -> np.ndarray:
        if benchmark_name not in self.benchmark_results:
            self.run()
        return self.benchmark_results[benchmark_name]

    def to_pickle(self, path: str):
        with gzip.open(path, "wb") as f:
            pickle.dump(self.benchmark_results, f)

    @staticmethod
    def from_pickle(path: str):
        with gzip.open(path, "rb") as f:
            benchmark_results = pickle.load(f)
        obj = DataDistribution()
        obj.benchmark_results = benchmark_results
        name = Path(path).name
        if "." in name:
            name = name.split(".")[0]
        obj.name = name
        return obj

    def __repr__(self) -> str:
        return self.dataset.__repr__()
