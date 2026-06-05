import tempfile
from typing import Optional

import numpy as np
import librosa
import torch

from ttsds.benchmarks.benchmark import (
    Benchmark,
    BenchmarkCategory,
    BenchmarkDimension,
    DeviceSupport,
)
from ttsds.util.dataset import Dataset


# Defer the `voicerestore` import until the benchmark is actually instantiated.
# `voicerestore` pulls in BigVGAN which calls `_from_pretrained` with kwargs
# (proxies, resume_download) that newer huggingface_hub releases reject. We
# don't want that to break `import ttsds.ttsds` for callers who never run the
# environment benchmarks.
_VR_IMPORT_ERROR: Optional[Exception] = None
_ShortAudioRestorer = None
_get_mel_spectrogram = None


def _load_voicerestore() -> None:
    global _ShortAudioRestorer, _get_mel_spectrogram, _VR_IMPORT_ERROR
    if _ShortAudioRestorer is not None or _VR_IMPORT_ERROR is not None:
        return
    try:
        from voicerestore.restore import ShortAudioRestorer
        from voicerestore.bigvgan import get_mel_spectrogram
    except Exception as exc:
        _VR_IMPORT_ERROR = exc
        return
    _ShortAudioRestorer = ShortAudioRestorer
    _get_mel_spectrogram = get_mel_spectrogram


class VoiceRestoreBenchmark(Benchmark):
    """
    Benchmark class for the VoiceFixer benchmark.
    """

    def __init__(
        self,
    ):
        super().__init__(
            name="VoiceRestore",
            category=BenchmarkCategory.ENVIRONMENT,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="The PESQ of VoiceRestore.",
            supported_devices=[DeviceSupport.CPU, DeviceSupport.GPU],
        )
        _load_voicerestore()
        if _VR_IMPORT_ERROR is not None:
            raise ImportError(
                "voicerestore could not be imported; the VoiceRestore "
                "benchmark is unavailable on this dep stack. Either install a "
                "compatible voicerestore/huggingface_hub combo or drop the "
                "'voicerestore' key from the benchmarks dict passed to "
                "BenchmarkSuite."
            ) from _VR_IMPORT_ERROR
        self.vr = _ShortAudioRestorer()
        self.device = "cpu"

    def _to_device(self, device: str):
        """
        Move the model to the given device.

        Args:
            device (str): The device to move the model to.
        """
        self.vr.model.to(device)
        self.device = device

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Compute the Word Error Rate (WER) distribution of the VoiceFixer model.

        Args:
            dataset (Dataset): The dataset to compute the WER on.

        Returns:
            float: The Word Error Rate (WER) distribution of the VoiceFixer model.
        """
        diffs = []
        for wav, _ in dataset.iter_with_progress(self):
            if dataset.sample_rate != 16000:
                wav = librosa.resample(
                    wav, orig_sr=dataset.sample_rate, target_sr=16000
                )
            # take random 2 seconds
            if len(wav) > 32000:
                start = np.random.randint(0, len(wav) - 32000)
                wav = wav[start : start + 32000]
            wav = torch.from_numpy(wav).to(self.device).unsqueeze(0)
            processed_mel = _get_mel_spectrogram(
                wav, self.vr.model.bigvgan_model.h
            ).to(self.device)
            # Restore audio
            restored_mel = self.vr.model.voice_restore.sample(
                processed_mel.transpose(1, 2),
                steps=4,
                cfg_strength=0.5,
            )
            restored_mel = restored_mel.squeeze(0).transpose(0, 1)
            restored_mel = restored_mel.cpu().numpy()
            processed_mel = processed_mel.cpu().numpy()
            mse = np.mean((restored_mel - processed_mel) ** 2)
            diffs.append(mse)
        return np.array(diffs)
