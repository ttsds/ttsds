import tempfile
import os

import wespeaker
from tqdm import tqdm
import numpy as np
import librosa
import soundfile as sf
from pyannote.audio import Model, Inference


from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsds.util.dataset import Dataset


class WeSpeakerBenchmark(Benchmark):
    """
    Benchmark class for the WeSpeaker benchmark.
    """

    def __init__(
        self,
        window_duration: float = 1.0,
        window_step: float = 0.5,
        measure_std: bool = False,
    ):
        super().__init__(
            name="WeSpeaker",
            category=BenchmarkCategory.SPEAKER,
            dimension=BenchmarkDimension.N_DIMENSIONAL,
            description="The speaker embeddings using WeSpeaker.",
            window_duration=window_duration,
            window_step=window_step,
            measure_std=measure_std,
        )
        self.model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
        self.inference = Inference(
            self.model, window="sliding", duration=window_duration, step=window_step
        )
        self.measure_std = measure_std

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the WeSpeaker benchmark.

        Args:
            dataset (Dataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the WeSpeaker benchmark.
        """
        embeddings = []
        for wav, _ in tqdm(dataset, desc=f"computing embeddings for {self.name}"):
            if dataset.sample_rate != 16000:
                wav = librosa.resample(
                    wav, orig_sr=dataset.sample_rate, target_sr=16000
                )
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                sf.write(f.name, wav, 16000)
                embedding = self.inference(f.name)
                embedding = [x[1] for x in embedding]
            if self.measure_std:
                embedding = np.std(embedding, axis=0)
                embeddings.append(embedding)
            else:
                embeddings.extend(embedding)
        embeddings = np.vstack(embeddings)
        return embeddings
