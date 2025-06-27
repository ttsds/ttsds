"""Prosody benchmarks for evaluating speech rhythm, intonation, and timing."""

from ttsds.benchmarks.prosody.mpm import MPMBenchmark
from ttsds.benchmarks.prosody.pitch import PitchBenchmark
from ttsds.benchmarks.prosody.hubert_token import HubertTokenSRBenchmark
from ttsds.benchmarks.prosody.mhubert_token import MHubert147TokenSRBenchmark
from ttsds.benchmarks.prosody.allosaurus import AllosaurusSRBenchmark

__all__ = [
    "MPMBenchmark",
    "PitchBenchmark",
    "HubertTokenSRBenchmark",
    "MHubert147TokenSRBenchmark",
    "AllosaurusSRBenchmark",
]
