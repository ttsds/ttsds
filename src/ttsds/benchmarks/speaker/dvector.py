import importlib.resources
import tempfile

import torch
import torch.nn as nn
import torchaudio
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import MelSpectrogram

import numpy as np

with importlib.resources.path("ttsds", "dvector") as dp:
    dvector_pt = dp / "dvector.pt"

# wav_tensor, sample_rate = torchaudio.load("example.wav")
# mel_tensor = wav2mel(wav_tensor, sample_rate)  # shape: (frames, mel_dim)
# emb_tensor = dvector.embed_utterance(mel_tensor)  # shape: (emb_dim)

from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsds.util.dataset import Dataset


class Wav2Mel(nn.Module):
    """Transform audio file into mel spectrogram tensors."""

    def __init__(
        self,
        sample_rate: int = 16000,
        norm_db: float = -3.0,
        sil_threshold: float = 1.0,
        sil_duration: float = 0.1,
        fft_window_ms: float = 25.0,
        fft_hop_ms: float = 10.0,
        f_min: float = 50.0,
        n_mels: int = 40,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.norm_db = norm_db
        self.sil_threshold = sil_threshold
        self.sil_duration = sil_duration
        self.fft_window_ms = fft_window_ms
        self.fft_hop_ms = fft_hop_ms
        self.f_min = f_min
        self.n_mels = n_mels
        self.sox_effects = SoxEffects(sample_rate, norm_db, sil_threshold, sil_duration)
        self.log_melspectrogram = LogMelspectrogram(
            sample_rate, fft_window_ms, fft_hop_ms, f_min, n_mels
        )

    def forward(self, wav_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav_tensor = self.sox_effects(wav_tensor, sample_rate)
        mel_tensor = self.log_melspectrogram(wav_tensor)
        return mel_tensor


class SoxEffects(nn.Module):
    """Transform waveform tensors."""

    def __init__(
        self,
        sample_rate: int,
        norm_db: float,
        sil_threshold: float,
        sil_duration: float,
    ):
        super().__init__()
        self.effects = [
            ["channels", "1"],  # convert to mono
            ["rate", f"{sample_rate}"],  # resample
            ["norm", f"{norm_db}"],  # normalize to -3 dB
        ]

    def forward(self, wav_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav_tensor, _ = apply_effects_tensor(wav_tensor, sample_rate, self.effects)
        return wav_tensor


class LogMelspectrogram(nn.Module):
    """Transform waveform tensors into log mel spectrogram tensors."""

    def __init__(
        self,
        sample_rate: int,
        fft_window_ms: float,
        fft_hop_ms: float,
        f_min: float,
        n_mels: int,
    ):
        super().__init__()
        self.melspectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=int(sample_rate * fft_hop_ms / 1000),
            n_fft=int(sample_rate * fft_window_ms / 1000),
            f_min=f_min,
            n_mels=n_mels,
        )

    def forward(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        mel_tensor = self.melspectrogram(wav_tensor).squeeze(0).T  # (time, n_mels)
        return torch.log(torch.clamp(mel_tensor, min=1e-9))


class DVectorBenchmark(Benchmark):
    """
    Benchmark class for evaluating speaker identity using the DVector model.

    This benchmark extracts speaker embeddings from audio samples using a pretrained
    DVector model. DVector is a speaker verification system that can distinguish
    between different speakers by creating a compact representation (embedding)
    of the speaker's voice characteristics.

    The benchmark extracts embeddings from sliding windows of audio and can either
    use the mean embedding per utterance or measure the standard deviation of
    embeddings across windows in an utterance, depending on the `measure_std` parameter.

    Attributes:
        wav2mel (Wav2Mel): Mel spectrogram extractor.
        dvector (torch.jit.ScriptModule): Pretrained DVector model.
        window_duration (float): Duration of each window in seconds.
        window_step (float): Step size between consecutive windows in seconds.
        measure_std (bool): Whether to measure standard deviation of embeddings across windows.
    """

    def __init__(
        self,
        window_duration: float = 1.0,
        window_step: float = 0.5,
        measure_std: bool = False,
    ):
        """
        Initialize the DVector benchmark.

        Args:
            window_duration: Duration of each window in seconds for processing speech.
                Longer windows capture more context but require more computation.
                Default is 1.0 second.
            window_step: Step size between consecutive windows in seconds.
                Smaller step sizes result in more overlapping windows and more embeddings.
                Default is 0.5 second.
            measure_std: If True, calculate the standard deviation of embeddings
                across windows in an utterance, which measures speaker consistency.
                If False, return all embeddings from all windows.
                Default is False.
        """
        super().__init__(
            name="DVector",
            category=BenchmarkCategory.SPEAKER,
            dimension=BenchmarkDimension.N_DIMENSIONAL,
            description="Speaker embeddings using DVector for speaker verification.",
            window_duration=window_duration,
            window_step=window_step,
            measure_std=measure_std,
        )
        self.wav2mel = Wav2Mel()
        self.dvector = torch.jit.load(dvector_pt).eval()
        self.window_duration = window_duration
        self.window_step = window_step
        self.measure_std = measure_std

    def _get_distribution(self, dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract DVector embeddings from audio samples in the dataset.

        This method processes each audio sample in the dataset, extracting
        sliding windows and computing DVector embeddings for each window.
        Depending on the configuration, it either returns all embeddings
        or computes the standard deviation of embeddings across windows.

        Args:
            dataset: The dataset containing audio samples to process.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - Mean of embeddings across all samples
                - Covariance matrix of embeddings
        """
        embeddings = []
        for wav, _ in dataset.iter_with_progress(self):
            window_length = int(self.window_duration * dataset.sample_rate)
            hop_length = int(self.window_step * dataset.sample_rate)
            utt_embeddings = []
            for i in range(0, len(wav), hop_length):
                wav_part = wav[i : i + window_length]
                if len(wav_part) > window_length // 2:
                    wav_tensor = torch.tensor(wav_part).float().unsqueeze(0)
                    mel_tensor = self.wav2mel(wav_tensor, dataset.sample_rate)
                    with torch.no_grad():
                        emb_tensor = self.dvector.embed_utterance(mel_tensor)
                    utt_embeddings.append(emb_tensor)
            if len(utt_embeddings) > 1 and self.measure_std:
                utt_embeddings = torch.stack(utt_embeddings)
                utt_embeddings = utt_embeddings.std(dim=0)
                embeddings.append(utt_embeddings)
            elif len(utt_embeddings) > 0 and not self.measure_std:
                embeddings.extend(utt_embeddings)

        # Convert to numpy array and calculate mean and covariance
        embeddings = np.vstack(embeddings)
        mean = np.mean(embeddings, axis=0)
        cov = np.cov(embeddings, rowvar=False)

        return mean, cov
