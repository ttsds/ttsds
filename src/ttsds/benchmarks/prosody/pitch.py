from tqdm import tqdm
import numpy as np

from ttsds.util.measures import PitchMeasure
from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsds.util.dataset import Dataset


class PitchBenchmark(Benchmark):
    """
    Benchmark class for the pitch benchmark.
    """

    def __init__(
        self,
    ):
        super().__init__(
            name="Pitch",
            category=BenchmarkCategory.PROSODY,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="The pitch of the audio.",
        )
        self.pitch_measure = PitchMeasure()

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the pitch benchmark.

        Args:
            dataset (Dataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the pitch benchmark.
        """
        pitches = []
        for wav, _ in tqdm(dataset, desc=f"computing pitches for {self.name}"):
            pitch = self.pitch_measure(wav, np.array([1000]))["measure"]
            pitches.append(pitch)
        pitches = np.concatenate(pitches)
        return pitches
