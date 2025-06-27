"""General benchmarks for evaluating speech quality metrics."""

from .hubert import HubertBenchmark
from .wav2vec2 import Wav2Vec2Benchmark
from .wav2vec2_xlsr import Wav2Vec2XLSRBenchmark
from .wavlm import WavLMBenchmark
from .mhubert import MHubert147Benchmark

__all__ = [
    "HubertBenchmark",
    "Wav2Vec2Benchmark",
    "Wav2Vec2XLSRBenchmark",
    "WavLMBenchmark",
    "MHubert147Benchmark",
]
