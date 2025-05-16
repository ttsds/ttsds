from typing import Union

import numpy as np
import torch
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

import librosa

from ttsds.benchmarks.benchmark import (
    Benchmark,
    BenchmarkCategory,
    BenchmarkDimension,
    DeviceSupport,
)
from ttsds.util.dataset import Dataset


class WavLMBenchmark(Benchmark):
    """
    Benchmark class for the WavLM benchmark.
    """

    def __init__(
        self,
        wavlm_model: str = "microsoft/wavlm-base-plus",
        wavlm_layer: Union[int, str] = 11,
    ):
        super().__init__(
            name="WavLM",
            category=BenchmarkCategory.GENERIC,
            dimension=BenchmarkDimension.N_DIMENSIONAL,
            description="WavLM hidden states.",
            wavlm_model=wavlm_model,
            wavlm_layer=wavlm_layer,
            supported_devices=[DeviceSupport.CPU, DeviceSupport.GPU],
        )
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        self.model = WavLMModel.from_pretrained(wavlm_model)
        self.model_layer = wavlm_layer
        self.device = "cpu"

    def _to_device(self, device: str):
        """
        Move the model to the given device.

        Args:
            device (str): The device to move the model to.
        """
        self.model.to(device)
        self.device = device

    def get_embedding(self, wav, sr) -> np.ndarray:
        """
        Get the embedding of a wav file.

        Args:
            wav (np.ndarray): The wav file.

        Returns:
            np.ndarray: The embedding of the wav file.
        """
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        input_values = self.processor(
            wav, return_tensors="pt", sampling_rate=sr
        ).input_values
        input_values = input_values.to(self.device)
        with torch.no_grad():
            features = self.model(input_values, output_hidden_states=True).hidden_states
        if isinstance(self.model_layer, int):
            features = features[self.model_layer].detach().cpu().numpy()[0]
        else:
            layer_num = len(features)
            features_new = []
            for i in range(layer_num):
                features_new.append(features[i].detach().cpu().numpy()[0])
            features = np.stack(features_new, axis=-1)
            features = features.reshape(features.shape[0], -1)
        return features

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the WavLM benchmark.

        Args:
            dataset (DirectoryDataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the WavLM benchmark.
        """
        embeddings = []
        for wav, _ in dataset.iter_with_progress(self):
            embeddings.append(self.get_embedding(wav, dataset.sample_rate))
        return np.vstack(embeddings)
