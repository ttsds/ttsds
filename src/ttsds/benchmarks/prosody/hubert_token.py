from typing import Union, List
import importlib.resources

import numpy as np
import torch
from transformers import Wav2Vec2Processor, HubertModel
from tqdm import tqdm
import librosa
from sklearn.cluster import KMeans
import requests

from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsds.util.dataset import Dataset, TarDataset
from ttsds.util.cache import cache, load_cache, check_cache, hash_md5, CACHE_DIR


class HubertTokenBenchmark(Benchmark):
    """
    Benchmark class for the Hubert benchmark.
    """

    def __init__(
        self,
        hubert_model: str = "facebook/hubert-base-ls960",
        hubert_layer: Union[int, str] = 7,
        cluster_num: int = 100,
        cluster_seed: int = 42,
        cluster_datasets: List[Dataset] = None,
    ):
        super().__init__(
            name="Hubert Token",
            category=BenchmarkCategory.PROSODY,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="Hubert hidden states.",
            hubert_model=hubert_model,
            hubert_layer=hubert_layer,
            cluster_num=cluster_num,
            cluster_seed=cluster_seed,
            cluster_datasets=cluster_datasets,
        )
        if cluster_datasets is None:
            TEST_DS_URL = "https://www.openslr.org/resources/60/test-clean.tar.gz"
            # download to cache
            TEST_DS_PATH = CACHE_DIR / "test-clean.tar.gz"
            if not TEST_DS_PATH.exists():
                print(f"downloading {TEST_DS_URL} to {TEST_DS_PATH} for HubertTokenBenchmark")
                with open(TEST_DS_PATH, "wb") as f:
                    f.write(requests.get(TEST_DS_URL).content)

            cluster_datasets = [
                TarDataset(TEST_DS_PATH, text_suffix=".normalized.txt", path_prefix="./").sample(
                    100
                )
            ]
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        )
        self.model = HubertModel.from_pretrained(hubert_model)
        self.model_layer = hubert_layer
        self.kmeans = self.create_clusters(cluster_num, cluster_seed, cluster_datasets)

    def create_clusters(
        self, cluster_num: int, cluster_seed: int, cluster_datasets: Dataset
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
        wavs = []
        for ds in tqdm(cluster_datasets, desc=f"loading wavs for {self.name}"):
            wavs.extend(
                [
                    (wav, ds.sample_rate)
                    for wav, _ in tqdm(ds, desc=f"loading wavs for {self.name} {ds}")
                ]
            )
        embeddings = []
        print(len(wavs))
        for wav in tqdm(wavs):
            embeddings.append(self.get_embedding(wav[0], wav[1]))
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
        wavs = [
            wav
            for wav, _ in tqdm(dataset, desc=f"loading wavs for {self.name} {dataset}")
        ]
        lengths = []
        for wav in tqdm(wavs):
            wav_emb = self.get_embedding(wav, dataset.sample_rate)
            cluster = self.kmeans.predict(wav_emb)
            # the lengths are the number of times each cluster is repeated in a row
            current_length = 1
            for i in range(1, len(cluster)):
                if cluster[i] == cluster[i - 1]:
                    current_length += 1
                else:
                    lengths.append(current_length)
                    current_length = 1
            lengths.append(current_length)
        return np.array(lengths)
