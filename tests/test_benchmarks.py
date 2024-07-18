# SPDX-FileCopyrightText: 2024-present Christoph Minixhofer <christoph.minixhofer@gmail.com>
#
# SPDX-License-Identifier: MIT
import importlib.resources

import numpy as np

from ttsdb.benchmarks.general.mfcc import MFCCBenchmark
from ttsdb.benchmarks.general.hubert import HubertBenchmark
from ttsdb.benchmarks.intelligibility.w2v2_wer import Wav2Vec2WERBenchmark
from ttsdb.benchmarks.intelligibility.whisper_wer import WhisperWERBenchmark
from ttsdb.benchmarks.environment.voicefixer import VoiceFixerBenchmark
from ttsdb.benchmarks.environment.wada_snr import WadaSNRBenchmark
from ttsdb.benchmarks.phonetics.allosaurus import AllosaurusBenchmark
from ttsdb.benchmarks.prosody.mpm import MPMBenchmark
from ttsdb.benchmarks.prosody.mfa import MFADurationBenchmark
from ttsdb.util.dataset import TarDataset

with importlib.resources.path("ttsdb", "data") as data_path:
    test_dataset = data_path / "libritts_test.tar.gz"
    dev_dataset = data_path / "libritts_dev.tar.gz"
    noise_dataset = data_path / "noise.tar.gz"
    test_tar_dataset = TarDataset(test_dataset).sample(100)
    dev_tar_dataset = TarDataset(dev_dataset).sample(100)
    noise_tar_dataset = TarDataset(noise_dataset).sample(100)


# def test_mfcc_compute_distance():
#     benchmark = MFCCBenchmark()
#     result = benchmark.compute_distance(test_tar_dataset, dev_tar_dataset)
#     assert int(np.round(result, 0)) in [241, 242]


# def test_mfcc_compute_score():
#     benchmark = MFCCBenchmark()
#     result = benchmark.compute_score(dev_tar_dataset)
#     assert float(np.round(result[0], 1)) in [99.8, 99.9]


# def test_hubert_compute_distance():
#     benchmark = HubertBenchmark()
#     result = benchmark.compute_distance(test_tar_dataset, dev_tar_dataset)
#     assert float(np.round(result, 1)) in [2.1, 2.2]


# def test_hubert_compute_score():
#     benchmark = HubertBenchmark()
#     result = benchmark.compute_score(dev_tar_dataset)
#     assert float(np.round(result[0], 1)) in [98.6, 98.7]


# def test_intelligibility_w2v2():
#     benchmark = Wav2Vec2WERBenchmark()
#     result = benchmark.compute_distance(test_tar_dataset, dev_tar_dataset)
#     assert float(np.round(result, 1)) == 2.8


# def test_intelligibility_w2v2_score():
#     benchmark = Wav2Vec2WERBenchmark()
#     result = benchmark.compute_score(dev_tar_dataset)
#     assert int(np.round(result[0], 0)) == 87


# def test_intelligibility_whisper():
#     benchmark = WhisperWERBenchmark()
#     result = benchmark.compute_distance(test_tar_dataset, dev_tar_dataset)
#     assert float(np.round(result, 1)) == 2.8


# def test_environment_voicefixer():
#     benchmark = VoiceFixerBenchmark()
#     result = benchmark.compute_distance(test_tar_dataset, dev_tar_dataset)
#     assert float(np.round(result, 1)) in [5.8, 5.9]


# def test_environment_voicefixer_score():
#     benchmark = VoiceFixerBenchmark()
#     result = benchmark.compute_score(dev_tar_dataset)
#     assert float(np.round(result[0], 1)) in [98.4, 98.5]

# def test_environment_wada():
#     benchmark = WadaSNRBenchmark()
#     result = benchmark.compute_distance(test_tar_dataset, dev_tar_dataset)
#     assert float(np.round(result, 1)) in [2.0, 2.1]


# def test_environment_wada_score():
#     benchmark = WadaSNRBenchmark()
#     result = benchmark.compute_score(dev_tar_dataset)
#     assert float(np.round(result[0], 1)) in [87.7, 87.8]


# def test_phonetics_allosaurus():
#     benchmark = AllosaurusBenchmark()
#     result = benchmark.compute_distance(test_tar_dataset, dev_tar_dataset)
#     assert float(np.round(result, 1)) in [0.0, 0.1]


# def test_phonetics_allosaurus_score():
#     benchmark = AllosaurusBenchmark()
#     result = benchmark.compute_score(dev_tar_dataset)
#     assert float(np.round(result[0], 1)) in [97.9, 98.0]


def test_prosody_mfa():
    benchmark = MFADurationBenchmark()
    result = benchmark.compute_distance(test_tar_dataset, dev_tar_dataset)
    print(result)
    assert float(np.round(result, 1)) in [3.3, 3.4]


def test_prosody_mfa_score():
    benchmark = MFADurationBenchmark()
    result = benchmark.compute_score(dev_tar_dataset)
    print(result)
    assert float(np.round(result[0], 1)) in [98.5, 98.6]


# def test_prosody_mpm():
#     benchmark = MPMBenchmark()
#     result = benchmark.compute_distance(test_tar_dataset, dev_tar_dataset)
#     assert float(np.round(result, 1)) in [3.3, 3.4]


# def test_prosody_mpm_score():
#     benchmark = MPMBenchmark()
#     result = benchmark.compute_score(dev_tar_dataset)
#     assert float(np.round(result[0], 1)) in [98.5, 98.6]
