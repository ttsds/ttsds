import re
import tempfile

import whisper
from tqdm import tqdm
from jiwer import wer
import torch
import numpy as np
import librosa
import soundfile as sf

from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsds.util.dataset import Dataset


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
        )
        self.model = whisper.load_model(whisper_model)

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
                wav = librosa.resample(
                    wav, orig_sr=dataset.sample_rate, target_sr=16000
                )
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                sf.write(f.name, wav, 16000)
                pred_transcript = self.model.transcribe(f.name)["text"]
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
