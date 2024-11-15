import re
import tempfile

import torch
from transformers import WhisperProcessor, WhisperModel
from tqdm import tqdm
import numpy as np
import librosa
import soundfile as sf

from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension, DeviceSupport
from ttsds.util.dataset import Dataset


class WhisperActivationsBenchmark(Benchmark):
    """
    Benchmark class for extracting activations from the Whisper model.
    """

    def __init__(
        self,
        whisper_model: str = "small.en",
    ):
        super().__init__(
            name="Whisper Activations",
            category=BenchmarkCategory.INTELLIGIBILITY,
            dimension=BenchmarkDimension.N_DIMENSIONAL,
            description="Extracted feature activations from the Whisper model.",
            whisper_model=whisper_model,
            supported_devices=[DeviceSupport.CPU, DeviceSupport.GPU],
        )
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
        self.model = WhisperModel.from_pretrained("openai/whisper-small.en")
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
        for wav, _ in tqdm(dataset, desc=f"Extracting activations for {self.name}"):
            if dataset.sample_rate != 16000:
                wav = librosa.resample(wav, orig_sr=dataset.sample_rate, target_sr=16000)
            input_features = self.processor(
                wav, 
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features 
            input_features = input_features.to(self.device)
            decoder_input_ids = torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id
            decoder_input_ids = decoder_input_ids.to(self.device)
            with torch.no_grad():
                # Extract the encoder outputs (activations) directly
                outputs = self.model(input_features, decoder_input_ids=decoder_input_ids)
                pooled_features = outputs.last_hidden_state[0].mean(dim=0)
            activations.append(pooled_features.cpu().numpy())
        return np.vstack(activations)