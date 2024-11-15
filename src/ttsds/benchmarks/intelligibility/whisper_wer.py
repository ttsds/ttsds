import re
import tempfile

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
from jiwer import wer
import torch
import numpy as np
import librosa
import soundfile as sf

from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension, DeviceSupport
from ttsds.util.dataset import Dataset
from ttsds.util.cache import hash_md5


class WhisperWERBenchmark(Benchmark):
    """
    Benchmark class for the Whisper Word Error Rate (WER) benchmark.
    """

    def __init__(
        self,
        whisper_model: str = "small.en",
    ):
        super().__init__(
            name="Whisper WER",
            category=BenchmarkCategory.INTELLIGIBILITY,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="The Word Error Rate (WER) of Whisper.",
            whisper_model=whisper_model,
            supported_devices=[DeviceSupport.CPU, DeviceSupport.GPU],
        )
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")
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
        Compute the Word Error Rate (WER) distribution of the Whisper model.

        Args:
            dataset (Dataset): The dataset to compute the WER on.

        Returns:
            float: The Word Error Rate (WER) distribution of the Whisper model.
        """
        wers = []
        for wav, gt_transcript in tqdm(dataset, desc=f"computing WER for {self.name}"):
            if dataset.sample_rate != 16000:
                wav = librosa.resample(wav, orig_sr=dataset.sample_rate, target_sr=16000)
            input_features = self.processor(wav, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(self.device)
            predicted_ids = self.model.generate(input_features)
            pred_transcript = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            pred_transcript = re.sub(r"[^\w\s]", "", pred_transcript)
            gt_transcript = re.sub(r"[^\w\s]", "", gt_transcript)
            pred_transcript = re.sub(r"\s+", " ", pred_transcript)
            gt_transcript = re.sub(r"\s+", " ", gt_transcript)
            pred_transcript = pred_transcript.strip().lower()
            gt_transcript = gt_transcript.strip().lower()
            if len(gt_transcript) == 0:
                wers.append(1.0)
            else:
                wers.append(wer(gt_transcript, pred_transcript))
        return np.array(wers)
