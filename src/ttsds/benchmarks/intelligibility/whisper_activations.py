import re
import tempfile

import torch
import whisper
from tqdm import tqdm
import numpy as np
import librosa
import soundfile as sf

from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
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
        )
        self.model = whisper.load_model(whisper_model)
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
                wav = librosa.resample(
                    wav, orig_sr=dataset.sample_rate, target_sr=16000
                )
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                sf.write(f.name, wav, 16000)
                if self.device == 'cpu':
                    fp16 = False
                else:
                    fp16 = True
                with torch.no_grad():
                    # Extract the encoder outputs (activations) directly
                    audio = whisper.load_audio(f.name)
                    mel = whisper.log_mel_spectrogram(audio).to(self.device)
                    try:
                        encoder_out = self.model.encoder(mel)
                    except:
                        print(mel.shape, "mel failed, retrying")
                        encoder_out = self.model.encoder(mel)
                    # Pooling across the time dimension (e.g., mean pooling)
                    pooled_features = encoder_out.mean(dim=1).cpu().numpy()
                activations.append(pooled_features.squeeze())
        return np.stack(activations)