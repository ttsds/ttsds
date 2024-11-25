"""
The `DirectoryDataset` class is a dataset class for a directory containing wav files and corresponding text files.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
import hashlib
from pathlib import Path
import tarfile
from typing import Tuple, List, Dict, Union
import pickle
import gzip

import numpy as np
import librosa
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

from ttsds.util.cache import cache, check_cache, load_cache, hash_md5


class Dataset(ABC):
    """
    Abstract class for a dataset.
    """

    def __init__(self, name, sample_rate: int = 22050, has_text: bool = True):
        self.sample_rate = sample_rate
        self.wavs = []
        self.has_text = has_text
        if has_text:
            self.texts = []
        self.sample_params = {
            "n": None,
            "seed": None,
        }
        self.name = name
        self.indices = None

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, str], np.ndarray]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[np.ndarray, str]: The audio and text of the sample.
            np.ndarray: The audio of the sample, if the dataset does not have text.
        """
        raise NotImplementedError

    def sample(self, n: int, seed: int = 42) -> "DirectoryDataset":
        """
        Sample n samples from the dataset.

        Args:
            n (int): The number of samples to sample.
            seed (int): The seed for the random number generator.

        Returns:
            DirectoryDataset: A sampled dataset.
        """
        rng = np.random.default_rng(seed)
        self.indices = rng.choice(len(self), size=n, replace=False)
        self.sample_params = {"n": n, "seed": seed}
        return self


class DirectoryDataset(Dataset):
    """
    A dataset class for a directory containing
    with wav files and corresponding text files.
    """

    def __init__(
        self,
        root_dir: str = None,
        sample_rate: int = 22050,
        has_text: bool = True,
        text_suffix: str = ".txt",
        name: str = None,
    ):
        if name is not None:
            super().__init__(name, sample_rate, has_text)
        else:
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
        if self.indices is not None:
            return len(self.indices)
        return len(self.wavs)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
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
        return audio

    def __hash__(self) -> int:
        h = hashlib.md5()
        h.update(str(self.__class__).encode())
        h.update(str(self.root_dir).encode())
        h.update(str(self.sample_params["n"]).encode())
        h.update(str(self.sample_params["seed"]).encode())
        h.update(str(self.has_text).encode())
        return int(h.hexdigest(), 16)

    def __repr__(self) -> str:
        return f"({self.root_dir.name})"


class TarDataset(Dataset):
    """
    A dataset class for a tar file containing
    with wav files and corresponding text files.
    """

    def __init__(
        self,
        root_tar: str = None,
        sample_rate: int = 22050,
        has_text: bool = True,
        text_suffix: str = ".txt",
        path_prefix: str = None,
        name: str = None,
    ):
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
        return audio

    def __hash__(self) -> int:
        h = hashlib.md5()
        h.update(str(self.__class__).encode())
        h.update(str(self.root_tar).encode())
        h.update(str(self.sample_params["n"]).encode())
        h.update(str(self.sample_params["seed"]).encode())
        h.update(str(self.path_prefix).encode())
        return int(h.hexdigest(), 16)

    def __repr__(self) -> str:
        return f"({Path(self.root_tar).name})"


class WavListDataset(Dataset):
    """
    A dataset class for a list of wav files and corresponding text files.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        has_text: bool = True,
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
        return audio

    def __hash__(self) -> int:
        h = hashlib.md5()
        h.update(str(self.__class__).encode())
        h.update(str(self.sample_params["n"]).encode())
        h.update(str(self.sample_params["seed"]).encode())
        h.update(str(self.wavs).encode())
        h.update(str(self.texts).encode())
        h.update(str(self.has_text).encode())
        return int(h.hexdigest(), 16)

    def __repr__(self) -> str:
        return f"({self.name})"


class DataDistribution:
    def __init__(
        self,
        dataset: Dataset = None,
        benchmarks: Dict[str, "Benchmark"] = None,
        name: str = None,
        multiprocessing: bool = False,
        n_processes: int = 1,
    ):
        if name is not None:
            self.name = name
        elif dataset is not None:
            self.name = dataset.name
        self.benchmarks = benchmarks
        self.benchmark_results = {}
        self.multiprocessing = multiprocessing
        self.n_processes = n_processes
        if dataset is not None:
            self.dataset = dataset
            self.run()

    def run(self):
        if not self.multiprocessing:
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
        else:
            benchmark_key_values = list(self.benchmarks.items())
            values = list(self.benchmarks.values())
            keys = list(self.benchmarks.keys())
            print(
                f"Running {len(values)} benchmarks on {self.dataset.root_dir} using {self.n_processes} processes"
            )
            with ThreadPoolExecutor(max_workers=self.n_processes) as executor:
                results = list(executor.map(self._run_benchmark, values))
            for i, benchmark in enumerate(keys):
                self.benchmark_results[benchmark] = results[i]

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
