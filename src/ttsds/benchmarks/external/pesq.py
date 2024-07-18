from pesq import pesq
from tqdm import tqdm
import librosa
import numpy as np

from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsds.util.dataset import Dataset


class PESQBenchmark(Benchmark):
    """
    Benchmark class for the Hubert benchmark.
    """

    def __init__(
        self,
        reference_dataset: Dataset,
        sample_rate: int = 16000,
    ):
        super().__init__(
            name="PESQ",
            category=BenchmarkCategory.EXTERNAL,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="The PESQ benchmark.",
        )
        self.reference_dataset = reference_dataset
        self.sample_rate = sample_rate

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the Hubert benchmark.

        Args:
            dataset (DirectoryDataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the Hubert benchmark.
        """
        wavs = [
            (wav, txt)
            for wav, txt, _ in tqdm(
                dataset, desc=f"loading wavs for {self.name} {dataset}"
            )
        ]
        wavs_ref = [
            (wav, txt)
            for wav, txt, _ in tqdm(
                self.reference_dataset,
                desc=f"loading wavs for {self.name} {self.reference_dataset}",
            )
        ]
        embeddings = []
        scores = []
        sr = dataset.sample_rate
        sr_ref = self.reference_dataset.sample_rate
        for (wav, txt), (wav_ref, txt_ref) in zip(wavs, wavs_ref):
            if sr != self.sample_rate:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
            if sr_ref != self.sample_rate:
                wav_ref = librosa.resample(
                    wav_ref, orig_sr=sr_ref, target_sr=self.sample_rate
                )
            if txt != txt_ref:
                raise ValueError(f"Text mismatch between {txt} and {txt_ref}")
            score = pesq(self.sample_rate, wav_ref, wav, "wb")
            scores.append(score)
        return scores
