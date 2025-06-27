---
layout: default
title: Benchmarks Reference
---

# Benchmarks Reference

This page documents the benchmark classes available in TTSDS. Each benchmark evaluates a specific aspect of speech quality.

## Benchmark Categories

TTSDS organizes benchmarks into five main categories:

1. **Speaker**: Evaluates speaker identity preservation
2. **Intelligibility**: Measures speech recognition performance
3. **Prosody**: Assesses speech rhythm and intonation
4. **Generic**: Self-supervised representations
5. **Environment**: Noise robustness evaluation

## Benchmark Base Class

All benchmarks inherit from the abstract `Benchmark` class:

```python
from ttsds.benchmarks import Benchmark, BenchmarkCategory, BenchmarkDimension
```

This base class provides common functionality for distribution calculation, distance computation, and score normalization.

## Available Benchmarks

TTSDS includes the following benchmark implementations:

### Speaker Benchmarks

These benchmarks evaluate how well a TTS system preserves speaker identity.

#### DVectorBenchmark

```python
from ttsds.benchmarks.speaker.dvector import DVectorBenchmark
```

Evaluates speaker identity preservation using the d-vector speaker embedding model.

#### WeSpeakerBenchmark

```python
from ttsds.benchmarks.speaker.wespeaker import WeSpeakerBenchmark
```

Uses the [WeSpeaker](https://github.com/wenet-e2e/wespeaker) model for embeddings.

### Prosody Benchmarks

These benchmarks evaluate aspects of pitch and duration.

#### MPMBenchmark

```python
from ttsds.benchmarks.prosody.mpm import MPMBenchmark
```

Evaluates pitch distribution using the [Masked Prosody Model](https://arxiv.org/abs/2506.02584)

#### PitchBenchmark

```python
from ttsds.benchmarks.prosody.pitch import PitchBenchmark
```

Evaluates pitch distribution similarity (Pitch is extracted using [PyWorld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder))

#### HubertTokenSRBenchmark

```python
from ttsds.benchmarks.prosody.hubert_token import HubertTokenSRBenchmark
```

Uses HuBERT model tokens for speaking rate.

#### MHubert147TokenSRBenchmark

```python
from ttsds.benchmarks.prosody.mhubert_token import MHubert147TokenSRBenchmark
```

Uses multilingual HuBERT model tokens for speaking rate.

#### AllosaurusSRBenchmark

```python
from ttsds.benchmarks.prosody.allosaurus import AllosaurusSRBenchmark
```

Uses the Allosaurus phoneme recognizer for speaking rate.

### Generic Benchmarks

These benchmarks evaluate self-supervised representations.

#### HubertBenchmark

```python
from ttsds.benchmarks.generic.hubert import HubertBenchmark
```

Uses HuBERT model for speech representation.

#### Wav2Vec2Benchmark

```python
from ttsds.benchmarks.generic.wav2vec2 import Wav2Vec2Benchmark
```

Uses Wav2Vec 2.0 model for speech representation.

#### WavLMBenchmark

```python
from ttsds.benchmarks.generic.wavlm import WavLMBenchmark
```

Uses WavLM model for speech representation.

#### MHubert147Benchmark

```python
from ttsds.benchmarks.generic.mhubert import MHubert147Benchmark
```

Uses multilingual HuBERT model for speech representation.

#### Wav2Vec2XLSRBenchmark

```python
from ttsds.benchmarks.generic.wav2vec2_xlsr import Wav2Vec2XLSRBenchmark
```

Uses Wav2Vec 2.0 XLSR model for speech representation.

### Intelligibility Benchmarks

These benchmarks evaluate how well speech can be recognized and understood using the ASR head activations of the models.

#### Wav2Vec2ActivationsBenchmark

```python
from ttsds.benchmarks.intelligibility.w2v2_activations import Wav2Vec2ActivationsBenchmark
```

Uses Wav2Vec 2.0 model activations to assess speech intelligibility.

#### Wav2Vec2XLSRActivationsBenchmark

```python
from ttsds.benchmarks.intelligibility.w2v2_xlsr_activations import Wav2Vec2XLSRActivationsBenchmark
```

Uses Wav2Vec 2.0 XLSR model activations for multilingual speech intelligibility assessment.

#### WhisperActivationsBenchmark

```python
from ttsds.benchmarks.intelligibility.whisper_activations import WhisperActivationsBenchmark
```

Uses Whisper model activations to assess speech intelligibility.

#### MWhisperActivationsBenchmark

```python
from ttsds.benchmarks.intelligibility.mwhisper_activations import MWhisperActivationsBenchmark
```

Uses multilingual Whisper model activations for speech intelligibility assessment.

### Environment Benchmarks

These benchmarks evaluate speech robustness in various environments.

#### VoiceRestoreBenchmark

```python
from ttsds.benchmarks.environment.voicerestore import VoiceRestoreBenchmark
```

The SNR according to the difference between the original and restored speech.

#### WadaSNRBenchmark

```python
from ttsds.benchmarks.environment.wada_snr import WadaSNRBenchmark
```

Uses Wada SNR algorithm to assess speech quality in noise.

## Creating Custom Benchmarks

You can create custom benchmarks by extending the `Benchmark` base class:

```python
from ttsds.benchmarks import Benchmark, BenchmarkCategory, BenchmarkDimension

class CustomBenchmark(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(
            name="Custom",
            category=BenchmarkCategory.GENERIC,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="A custom benchmark",
            **kwargs
        )
    
    def _get_distribution(self, dataset):
        # Implement distribution calculation
        features = []
        for audio, text in dataset.iter_with_progress(self):
            # Process each audio sample
            # ...
            features.append(processed_feature)
        return np.array(features)
    
    def _to_device(self, device):
        # Implement device moving logic if needed
        pass
```

## Using Benchmarks Directly

You can use individual benchmarks directly without the BenchmarkSuite:

```python
from ttsds.benchmarks.speaker.dvector import DVectorBenchmark
from ttsds.util.dataset import DirectoryDataset

# Initialize datasets
dataset = DirectoryDataset("path/to/dataset", name="tts_output")
reference = DirectoryDataset("path/to/reference", name="reference")
noise = DirectoryDataset("path/to/noise", name="noise")

# Initialize benchmark
benchmark = DVectorBenchmark()

# Calculate score
score, (noise_name, ref_name) = benchmark.compute_score(
    dataset,
    [reference],
    [noise]
)
print(f"DVector Benchmark Score: {score:.4f}") 