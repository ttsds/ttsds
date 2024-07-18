import tempfile

from pesq import pesq
from voicefixer import VoiceFixer
from simple_hifigan import Synthesiser
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsds.util.dataset import Dataset


class VoiceFixerBenchmark(Benchmark):
    """
    Benchmark class for the VoiceFixer benchmark.
    """

    def __init__(
        self,
    ):
        super().__init__(
            name="VoiceFixer",
            category=BenchmarkCategory.ENVIRONMENT,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="The phone counts of VoiceFixer.",
        )
        self.model = VoiceFixer()
        self.synthesiser = Synthesiser()

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Compute the Word Error Rate (WER) distribution of the VoiceFixer model.

        Args:
            dataset (Dataset): The dataset to compute the WER on.

        Returns:
            float: The Word Error Rate (WER) distribution of the VoiceFixer model.
        """
        mel_diffs = []
        for wav, _ in tqdm(dataset, desc=f"computing noise for {self.name}"):
            if dataset.sample_rate != 16000:
                wav = librosa.resample(
                    wav, orig_sr=dataset.sample_rate, target_sr=16000
                )
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                # take random 2 seconds
                if len(wav) > 32000:
                    start = np.random.randint(0, len(wav) - 32000)
                    wav = wav[start : start + 32000]
                sf.write(f.name, wav, 16000)
                with tempfile.NamedTemporaryFile(suffix=".wav") as f_out:
                    self.model.restore(f.name, f_out.name)
                    wav_out, _ = librosa.load(f_out.name, sr=16000)
            wav = wav / (np.max(np.abs(wav)) + 1e-5)
            wav_out = wav_out / np.max(np.abs(wav_out))
            mel = self.synthesiser.wav_to_mel(wav, 16000)[0].T
            mel_out = self.synthesiser.wav_to_mel(wav_out, 16000)[0].T
            if mel_out.shape[0] > mel.shape[0]:
                mel_out = mel_out[: mel.shape[0]]
            elif mel_out.shape[0] < mel.shape[0]:
                mel = mel[: mel_out.shape[0]]
            mel_diff = mel_out
            # check if there is any nan
            if np.isnan(mel_diff).any():
                print("nan found, skip")
                continue
            # convert back to wav
            mel_diff = self.synthesiser(mel_diff.T)[0]
            mel_diff = mel_diff / np.max(np.abs(mel_diff) + 1e-5)
            mel_diff = librosa.resample(mel_diff, orig_sr=22050, target_sr=16000)
            # calculate the difference
            try:
                mel_diff = pesq(16000, wav, mel_diff, "wb")
                mel_diffs.append(mel_diff)
            except:
                mel_diffs.append(0)
        return np.array(mel_diffs)
