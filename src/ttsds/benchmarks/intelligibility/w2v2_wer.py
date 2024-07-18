import re

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from tqdm import tqdm
from jiwer import wer
import torch
import numpy as np
import librosa

from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsds.util.dataset import Dataset


class Wav2Vec2WERBenchmark(Benchmark):
    """
    Benchmark class for the Wav2Vec2 Word Error Rate (WER) benchmark.
    """

    def __init__(
        self,
        wav2vec2_model: str = "facebook/wav2vec2-base-960h",
    ):
        super().__init__(
            name="Wav2Vec2 WER",
            category=BenchmarkCategory.INTELLIGIBILITY,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="The Word Error Rate (WER) of Wav2Vec2.",
            wav2vec2_model=wav2vec2_model,
        )
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model)
        self.model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)
        self.model.eval()

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Compute the Word Error Rate (WER) distribution of the Wav2Vec2 model.

        Args:
            dataset (Dataset): The dataset to compute the WER on.

        Returns:
            float: The Word Error Rate (WER) distribution of the Wav2Vec2 model.
        """
        wers = []
        for wav, gt_transcript in tqdm(dataset, desc=f"computing WER for {self.name}"):
            if dataset.sample_rate != 16000:
                wav = librosa.resample(
                    wav, orig_sr=dataset.sample_rate, target_sr=16000
                )
            input_values = self.processor(
                wav, return_tensors="pt", sampling_rate=16000
            ).input_values
            with torch.no_grad():
                logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            pred_transcript = self.processor.batch_decode(predicted_ids)[0]
            pred_transcript = re.sub(r"[^\w\s]", "", pred_transcript)
            gt_transcript = re.sub(r"[^\w\s]", "", gt_transcript)
            pred_transcript = re.sub(r"\s+", " ", pred_transcript).lower()
            gt_transcript = re.sub(r"\s+", " ", gt_transcript).lower()
            wers.append(wer(gt_transcript, pred_transcript))
        return np.array(wers)
