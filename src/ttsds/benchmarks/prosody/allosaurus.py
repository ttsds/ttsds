import tempfile


import numpy as np
from allosaurus.app import read_recognizer
import soundfile as sf


from ttsds.benchmarks.benchmark import (
    Benchmark,
    BenchmarkCategory,
    BenchmarkDimension,
    DeviceSupport,
)
from ttsds.util.dataset import Dataset


class AllosaurusSRBenchmark(Benchmark):
    """
    Benchmark class for the Allosaurus speaking rate benchmark.
    """

    def __init__(
        self,
    ):
        super().__init__(
            name="Allosaurus SR",
            category=BenchmarkCategory.PROSODY,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="Uses Allosaurus phone durations to calculate speaking rate.",
            version="0.0.1",
        )
        self.model = read_recognizer()

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the speaking rate benchmark.

        Args:
            dataset (Dataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the speaking rate benchmark.
        """
        speaking_rates = []
        for wav, _ in dataset.iter_with_progress(self):
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                sf.write(f.name, wav, dataset.sample_rate)
                result = self.model.recognize(f.name, timestamp=True)
                if len(result.strip()) == 0:
                    speaking_rates.append(0)
                else:
                    # Count the number of phones and divide by audio duration
                    num_phones = len(result.strip().split("\n"))
                    audio_duration = len(wav) / dataset.sample_rate
                    speaking_rates.append(num_phones / audio_duration)
        return np.array(speaking_rates)
