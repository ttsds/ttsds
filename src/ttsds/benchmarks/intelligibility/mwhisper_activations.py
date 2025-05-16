import re
import tempfile

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import numpy as np
import librosa
import soundfile as sf

from ttsds.benchmarks.benchmark import (
    Benchmark,
    BenchmarkCategory,
    BenchmarkDimension,
    DeviceSupport,
)
from ttsds.util.dataset import Dataset


class MWhisperActivationsBenchmark(Benchmark):
    """
    Benchmark class for extracting activations from the Whisper model.
    """

    def __init__(
        self,
        whisper_model: str = "small",
    ):
        super().__init__(
            name="mWhisper Activations",
            category=BenchmarkCategory.INTELLIGIBILITY,
            dimension=BenchmarkDimension.N_DIMENSIONAL,
            description="Extracted feature activations from the Whisper model.",
            whisper_model=whisper_model,
            supported_devices=[DeviceSupport.CPU, DeviceSupport.GPU],
            version="1.3.0",
        )
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small"
        )
        self.device = "cpu"
        self.model.to(self.device)

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
        Extract activations from the Whisper model for the given dataset.

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
            input_features = self.processor(
                wav, sampling_rate=16000, return_tensors="pt"
            ).input_features
            input_features = input_features.to(self.device)
            with torch.no_grad():
                # Extract the encoder outputs (activations) directly
                outputs = self.model.generate(
                    input_features=input_features,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
                outputs = [x[-1].mean(dim=1) for x in outputs.encoder_hidden_states]
                outputs = torch.stack(outputs)
                outputs = outputs.squeeze(1).squeeze(1)
                pooled_features = outputs.mean(dim=0).squeeze(0)
            activations.append(pooled_features.cpu().numpy())
        return np.vstack(activations)
