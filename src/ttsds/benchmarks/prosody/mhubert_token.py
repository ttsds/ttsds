from typing import Union, List
import importlib.resources

import numpy as np
import torch
from transformers import Wav2Vec2Processor, HubertModel
import librosa
from sklearn.cluster import KMeans

from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsds.util.dataset import Dataset
from ttsds.util.cache import cache, load_cache, check_cache
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    ProgressColumn,
)
from rich.console import Console


class MHubert147TokenSRBenchmark(Benchmark):
    """
    Benchmark class for the Hubert benchmark. This benchmark is based on the speaking rate, rather than the individual tokens.
    """

    def __init__(
        self,
        cluster_datasets: List[Dataset],
        hubert_model: str = "utter-project/mHuBERT-147",
        hubert_layer: Union[int, str] = 7,
        cluster_num: int = 100,
        cluster_seed: int = 42,
    ):
        super().__init__(
            name="mHubert-147 Token SR",
            category=BenchmarkCategory.PROSODY,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="Uses mHubert-147 tokens to calculate speaking rate.",
            version="0.0.2",
            hubert_model=hubert_model,
            hubert_layer=hubert_layer,
            cluster_num=cluster_num,
            cluster_seed=cluster_seed,
            cluster_datasets=cluster_datasets,
        )
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        )
        self.model = HubertModel.from_pretrained(hubert_model)
        self.model_layer = hubert_layer
        self.kmeans = self.create_clusters(cluster_num, cluster_seed, cluster_datasets)

    def create_clusters(
        self, cluster_num: int, cluster_seed: int, cluster_datasets: List[Dataset]
    ) -> KMeans:
        """
        Create clusters for the Hubert benchmark.
        """
        cache_id = self.__hash__()
        if check_cache(cache_id):
            cluster_centres = load_cache(cache_id)
            kmeans = KMeans(n_clusters=cluster_num, random_state=cluster_seed)
            dummy = np.zeros((100, 768))
            kmeans.fit(dummy)
            kmeans.cluster_centers_ = cluster_centres
            return kmeans

        # Collect all wavs first
        wavs = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=Console(),
        ) as progress:
            # Outer progress for datasets
            dataset_task = progress.add_task(
                "[cyan]Creating hubert clusters...",
                total=len(cluster_datasets),
            )

            for dataset in cluster_datasets:
                progress.update(dataset_task)

                # Inner progress for wavs in current dataset
                wav_task = progress.add_task(
                    f"[green]Loading wavs from {dataset.name}...",
                    total=len(dataset),
                )

                for wav, _ in dataset:
                    wavs.append((wav, dataset.sample_rate))
                    progress.advance(wav_task)

                progress.remove_task(wav_task)
                progress.advance(dataset_task)

        # Process embeddings with progress
        embeddings = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=Console(),
        ) as progress:
            embedding_task = progress.add_task(
                "[yellow]Computing embeddings...", total=len(wavs)
            )

            for wav, sr in wavs:
                embeddings.append(self.get_embedding(wav, sr))
                progress.advance(embedding_task)

        embeddings = np.vstack(embeddings)
        kmeans = KMeans(n_clusters=cluster_num, random_state=cluster_seed).fit(
            embeddings
        )
        cache(kmeans.cluster_centers_, cache_id)
        return kmeans

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
        features = features[self.model_layer].detach().cpu().numpy()[0]
        return features

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the Hubert benchmark.

        Args:
            dataset (DirectoryDataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the Hubert benchmark.
        """
        lengths = []
        for wav, _ in dataset.iter_with_progress(self):
            wav_lengths = []
            wav_emb = self.get_embedding(wav, dataset.sample_rate)
            cluster = self.kmeans.predict(wav_emb)
            # the lengths are the number of times each cluster is repeated in a row
            current_length = 1
            for i in range(1, len(cluster)):
                if cluster[i] == cluster[i - 1]:
                    current_length += 1
                else:
                    wav_lengths.append(current_length)
                    current_length = 1
            wav_lengths.append(current_length)
            lengths.append(len(wav_lengths) / (len(wav) / dataset.sample_rate))
        return np.array(lengths)
