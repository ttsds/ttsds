"""
The `DirectoryDataset` class is a dataset class for a directory containing wav files and corresponding text files.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
import hashlib
from pathlib import Path
import tarfile
from typing import Tuple, List, Dict
import pickle
import gzip

import numpy as np
import librosa

from ttsds.util.cache import cache, check_cache, load_cache, hash_md5


class Dataset(ABC):
    """
    Abstract class for a dataset.
    """

    def __init__(self, name, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.wavs = []
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
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[Path, Path, str]: A tuple containing the wav file and text file.
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

    def __init__(self, root_dir: str = None, sample_rate: int = 22050):
        super().__init__(Path(root_dir).name, sample_rate)
        if root_dir is None:
            raise ValueError("root_dir must be provided.")
        self.root_dir = Path(root_dir)
        # we assume that the root directory contains
        wavs, texts = [], []
        for wav_file in Path(root_dir).rglob("*.wav"):
            wavs.append(wav_file)
            text = wav_file.with_suffix(".txt")
            texts.append(text)
        self.wavs = wavs
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
            audio = load_cache(wav_str)
        else:
            audio, _ = librosa.load(wav, sr=self.sample_rate)
            cache(audio, wav_str)
        with open(self.texts[idx], "r", encoding="utf-8") as f:
            text = f.read().replace("\n", "")
        if audio.shape[0] == 0:
            print(f"Empty audio file: {wav}, padding with zeros.")
            audio = np.zeros(16000)
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        return audio, text

    def __hash__(self) -> int:
        h = hashlib.md5()
        h.update(str(self.__class__).encode())
        h.update(str(self.root_dir).encode())
        h.update(str(self.sample_params["n"]).encode())
        h.update(str(self.sample_params["seed"]).encode())
        h.update(str(True).encode())
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
        text_suffix: str = ".txt",
        path_prefix: str = None,
    ):
        super().__init__(Path(root_tar).name, sample_rate)
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
                text_file = Path(member.name).with_suffix(text_suffix)
                texts.append(text_file)
        self.wavs = wavs
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
        return audio, text

    def __hash__(self) -> int:
        h = hashlib.md5()
        h.update(str(self.__class__).encode())
        h.update(str(self.root_tar).encode())
        h.update(str(self.sample_params["n"]).encode())
        h.update(str(self.sample_params["seed"]).encode())
        h.update(str(True).encode())
        return int(h.hexdigest(), 16)

    def __repr__(self) -> str:
        return f"({Path(self.root_tar).name})"


DEFAULT_BENCHMARKS = [
    "hubert",
    "wav2vec2",
    "wavlm",
    "wav2vec2_wer",
    "whisper_wer",
    "mpm",
    "pitch",
    "wespeaker",
    "dvector",
    "hubert_token",
    "voicefixer",
    "wada_snr",
]


class DataDistribution:
    def __init__(
        self,
        dataset: Dataset = None,
        benchmark_dict: Dict[str, "Benchmark"] = None,
        benchmarks: List[str] = DEFAULT_BENCHMARKS,
        name: str = None,
    ):
        if name is not None:
            self.name = name
        elif dataset is not None:
            self.name = dataset.name
        self.benchmarks = benchmarks
        if benchmark_dict is not None:
            self.benchmark_objects = {
                benchmark: benchmark_dict[benchmark] for benchmark in benchmarks
            }
        self.benchmark_results = {}
        if dataset is not None:
            self.dataset = dataset
            self.run()

    def run(self):
        for benchmark in self.benchmark_objects:
            print(f"Running {benchmark} on {self.dataset.root_dir}")
            bench = self.benchmark_objects[benchmark]
            dist = bench.get_distribution(self.dataset)
            if bench.dimension.name == "N_DIMENSIONAL":
                # compute mu and sigma and store as tuple
                mu = np.mean(dist, axis=0)
                sigma = np.cov(dist, rowvar=False)
                dist = (mu, sigma)
            self.benchmark_results[benchmark] = dist

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
