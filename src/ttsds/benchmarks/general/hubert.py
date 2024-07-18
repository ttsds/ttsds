from typing import Union

import numpy as np
import torch
from transformers import Wav2Vec2Processor, HubertModel
from tqdm import tqdm
import librosa

from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsds.util.dataset import Dataset


class HubertBenchmark(Benchmark):
    """
    Benchmark class for the Hubert benchmark.
    """

    def __init__(
        self,
        hubert_model: str = "facebook/hubert-base-ls960",
        hubert_layer: Union[int, str] = 7,
    ):
        super().__init__(
            name="Hubert",
            category=BenchmarkCategory.OVERALL,
            dimension=BenchmarkDimension.N_DIMENSIONAL,
            description="Hubert hidden states.",
            hubert_model=hubert_model,
            hubert_layer=hubert_layer,
        )
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        )
        self.model = HubertModel.from_pretrained(hubert_model)
        self.model_layer = hubert_layer

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
        Get the distribution of the Hubert benchmark.

        Args:
            dataset (DirectoryDataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the Hubert benchmark.
        """
        wavs = [
            wav
            for wav, _ in tqdm(dataset, desc=f"loading wavs for {self.name} {dataset}")
        ]
        embeddings = []
        for wav in tqdm(wavs):
            embeddings.append(self.get_embedding(wav, dataset.sample_rate))
        return np.vstack(embeddings)
