import tempfile

from tqdm import tqdm
import numpy as np
from allosaurus.app import read_recognizer
import soundfile as sf


from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsds.util.dataset import Dataset


class AllosaurusBenchmark(Benchmark):
    """
    Benchmark class for the allosaurus benchmark.
    """

    def __init__(
        self,
    ):
        super().__init__(
            name="Allosaurus",
            category=BenchmarkCategory.PROSODY,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="The phone length of Allosaurus.",
            version="0.0.1",
        )
        self.model = read_recognizer()

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the pitch benchmark.

        Args:
            dataset (Dataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the pitch benchmark.
        """
        durations = []
        for wav, _ in tqdm(
            dataset, desc=f"computing allosaurus phone durations for {self.name}"
        ):
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                sf.write(f.name, wav, dataset.sample_rate)
                result = self.model.recognize(f.name, timestamp=True)
                if len(result.strip()) == 0:
                    durations.append(0)
                else:
                    start_times = [
                        float(x.split()[0]) for x in result.strip().split("\n")
                    ]
                    # calculate time between start times
                    diff = np.diff(start_times)
                    if len(diff) == 0:
                        durations.append(0)
                    else:
                        durations.append(
                            float(len(diff)) / (len(wav) / dataset.sample_rate)
                        )
        return np.array(durations)
