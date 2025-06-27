"""Environment benchmarks for evaluating speech in noisy environments."""

from ttsds.benchmarks.environment.voicerestore import VoiceRestoreBenchmark
from ttsds.benchmarks.environment.wada_snr import WadaSNRBenchmark

__all__ = ["VoiceRestoreBenchmark", "WadaSNRBenchmark"]
