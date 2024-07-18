import torch
import numpy as np
import librosa
from tqdm import tqdm

from ttsds.util.mpm import MaskedProsodyModel
from ttsds.util.measures import (
    PitchMeasure,
    EnergyMeasure,
    VoiceActivityMeasure,
)
from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsds.util.dataset import Dataset


class MPMBenchmark(Benchmark):
    """
    Benchmark class for the Masked Prosody Model (MPM) benchmark.
    """

    def __init__(
        self,
        mpm_model: str = "cdminix/masked_prosody_model",
        mpm_layer: int = 7,
    ):
        super().__init__(
            name="MPM",
            category=BenchmarkCategory.PROSODY,
            dimension=BenchmarkDimension.N_DIMENSIONAL,
            description="The hidden states of the Masked Prosody Model (MPM).",
            mpm_model=mpm_model,
            mpm_layer=mpm_layer,
        )
        self.model = MaskedProsodyModel.from_pretrained(mpm_model)
        self.model_layer = mpm_layer
        self.model.eval()
        self.pitch_measure = PitchMeasure()
        self.energy_measure = EnergyMeasure()
        self.voice_activity_measure = VoiceActivityMeasure()
        self.pitch_min = 50
        self.pitch_max = 300
        self.energy_min = 0
        self.energy_max = 0.2
        self.vad_min = 0
        self.vad_max = 1
        self.bins = torch.linspace(0, 1, 128)
        self.mpm_layer = mpm_layer

    def get_embedding(self, wav, sr) -> np.ndarray:
        """
        Get the embedding of a wav file.

        Args:
            wav (np.ndarray): The wav file.

        Returns:
            np.ndarray: The embedding of the wav file.
        """
        features = self.model(wav, sr)
        features = features[self.model_layer].detach().cpu().numpy()[0]
        return features

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the MPM benchmark.

        Args:
            dataset (Dataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the MPM benchmark.
        """
        embeddings = []
        for wav, _ in tqdm(
            dataset, desc="loading masked prosody model representations"
        ):
            if dataset.sample_rate != 22050:
                wav = librosa.resample(
                    wav, orig_sr=dataset.sample_rate, target_sr=22050
                )
            pitch = self.pitch_measure(wav, np.array([1000]))["measure"]
            energy = self.energy_measure(wav, np.array([1000]))["measure"]
            vad = self.voice_activity_measure(wav, np.array([1000]))["measure"]
            pitch = torch.tensor(pitch)
            energy = torch.tensor(energy)
            vad = torch.tensor(vad)
            pitch[torch.isnan(pitch)] = -1000
            energy[torch.isnan(energy)] = -1000
            vad[torch.isnan(vad)] = -1000
            pitch = torch.clip(pitch, self.pitch_min, self.pitch_max)
            energy = torch.clip(energy, self.energy_min, self.energy_max)
            vad = torch.clip(vad, self.vad_min, self.vad_max)
            pitch = pitch / (self.pitch_max - self.pitch_min)
            energy = energy / (self.energy_max - self.energy_min)
            vad = vad / (self.vad_max - self.vad_min)
            pitch = torch.bucketize(pitch, self.bins)
            energy = torch.bucketize(energy, self.bins)
            vad = torch.bucketize(vad, torch.linspace(0, 1, 2))
            min_len = min(len(pitch), len(energy), len(vad))
            pitch = pitch[:min_len]
            energy = energy[:min_len]
            vad = vad[:min_len]
            model_input = torch.stack([pitch, energy, vad]).unsqueeze(0)
            with torch.no_grad():
                reprs = self.model(model_input, return_layer=self.mpm_layer)[
                    "representations"
                ]
            reprs = reprs.detach().cpu().numpy()[0]
            embeddings.append(reprs)
        embeddings = np.vstack(embeddings)
        return embeddings
