import re

from transformers import Wav2Vec2Processor, Wav2Vec2Model

import torch
import numpy as np
import librosa

from ttsds.benchmarks.benchmark import (
    Benchmark,
    BenchmarkCategory,
    BenchmarkDimension,
    DeviceSupport,
)
from ttsds.util.dataset import Dataset


class Wav2Vec2ActivationsBenchmark(Benchmark):
    """
    Benchmark class for extracting activations from the Wav2Vec2 model.
    """

    def __init__(
        self,
        wav2vec2_model: str = "facebook/wav2vec2-base-960h",
    ):
        super().__init__(
            name="Wav2Vec2 Activations",
            category=BenchmarkCategory.INTELLIGIBILITY,
            dimension=BenchmarkDimension.N_DIMENSIONAL,
            description="Extracted feature activations from the Wav2Vec2 model.",
            wav2vec2_model=wav2vec2_model,
            supported_devices=[DeviceSupport.CPU, DeviceSupport.GPU],
        )
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model)
        self.model = Wav2Vec2Model.from_pretrained(wav2vec2_model)
        self.model.eval()
        self.device = "cpu"

    def _to_device(self, device: str):
        """
        Move the model to the given device.

        Args:
            device (str): The device to move the model to.
        """
        self.model.to(device)
        self.device = device

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Extract activations from the Wav2Vec2 model for the given dataset.

        Args:
            dataset (Dataset): The dataset to extract activations from.

        Returns:
            np.ndarray: The extracted activations of shape (n, m),
                        where n is the number of samples and m is the feature dimension.
        """
        activations = []
        for wav, _ in dataset.iter_with_progress(self):
            if dataset.sample_rate != 16000:
                wav = librosa.resample(
                    wav, orig_sr=dataset.sample_rate, target_sr=16000
                )
            input_values = self.processor(
                wav, return_tensors="pt", sampling_rate=16000
            ).input_values
            input_values = input_values.to(self.device)
            with torch.no_grad():
                outputs = self.model(input_values)
                # Extract the last hidden state
                features = outputs.last_hidden_state
                # Pool across the time dimension (e.g., mean pooling)
                pooled_features = features.mean(dim=1).squeeze().cpu().numpy()
            activations.append(pooled_features)
        return np.stack(activations)
