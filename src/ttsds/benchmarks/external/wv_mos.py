from tqdm import tqdm
from wvmos import get_wvmos
import tempfile
import soundfile as sf
import numpy as np

from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsds.util.dataset import Dataset


class WVMOSBenchmark(Benchmark):
    """
    Benchmark class for the Hubert benchmark.
    """

    def __init__(
        self,
    ):
        super().__init__(
            name="WVMOS",
            category=BenchmarkCategory.EXTERNAL,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="The WVMOS benchmark.",
        )
        self.model = get_wvmos()

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the WVMOS benchmark.

        Args:
            dataset (DirectoryDataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the WVMOS benchmark.
        """
        scores = []
        for wav, txt in tqdm(dataset, desc=f"computing scores for {self.name}"):
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                sf.write(f.name, wav, dataset.sample_rate)
                score = self.model.calculate_one(f.name)
            scores.append(score)
        return np.array(scores)
