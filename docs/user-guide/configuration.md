---
layout: default
title: Configuration
---

# Configuration

This guide explains the various configuration options available in TTSDS to customize your benchmarking setup.

## BenchmarkSuite Parameters

The `BenchmarkSuite` class accepts several parameters to customize its behavior:

```python
from ttsds import BenchmarkSuite, BENCHMARKS
from ttsds.util.dataset import DirectoryDataset
from ttsds.benchmarks.benchmark import BenchmarkCategory

suite = BenchmarkSuite(
    datasets=[DirectoryDataset("path/to/dataset", name="my_dataset")],
    reference_datasets=[DirectoryDataset("path/to/reference", name="reference")],
    noise_datasets=[DirectoryDataset("path/to/noise", name="noise")],  # Optional: only if you want to use your own distractor noise
    category_weights={  # Optional: customize category importance
        BenchmarkCategory.SPEAKER: 0.25,
        BenchmarkCategory.INTELLIGIBILITY: 0.25,
        BenchmarkCategory.PROSODY: 0.25,
        BenchmarkCategory.GENERIC: 0.25,
        BenchmarkCategory.ENVIRONMENT: 0.0,
    },
    benchmarks=BENCHMARKS,
    write_to_file="results.csv",  # Optional: save results to CSV
    skip_errors=True,  # Optional: skip failed benchmarks
    include_environment=False,  # Optional: exclude environment benchmarks
    multilingual=False,  # Optional: enable multilingual evaluation (this uses the multilingual benchmarks from the TTSDS2 paper)
    benchmark_kwargs=None,  # Optional: specify benchmark kwargs
    device="cpu",  # Optional: specify device to run on
    cache_dir="~/.ttsds_cache",  # Optional: specify cache directory
    n_workers=1,  # Optional: number of parallel workers
)
```

## Dataset Configuration

The `DirectoryDataset` class allows you to configure how your speech datasets are loaded:

```python
from ttsds.util.dataset import DirectoryDataset

# Basic dataset with default settings
dataset = DirectoryDataset("path/to/dataset", name="my_dataset")

# Dataset with advanced configuration
dataset = DirectoryDataset(
    root_dir="path/to/dataset",
    name="my_dataset",
    sample_rate=22050,  # Target sample rate
    has_text=False,  # Deprecated, text is no longer used
)
```

## Category Weights

You can adjust the importance of different evaluation categories using the `category_weights` parameter:

```python
from ttsds.benchmarks.benchmark import BenchmarkCategory

# Equal weights (default)
category_weights = {
    BenchmarkCategory.SPEAKER: 0.2,
    BenchmarkCategory.INTELLIGIBILITY: 0.2,
    BenchmarkCategory.PROSODY: 0.2,
    BenchmarkCategory.GENERIC: 0.2,
    BenchmarkCategory.ENVIRONMENT: 0.2,
}

# Custom weights (excluding environment)
category_weights = {
    BenchmarkCategory.SPEAKER: 0.3,
    BenchmarkCategory.INTELLIGIBILITY: 0.3,
    BenchmarkCategory.PROSODY: 0.3,
    BenchmarkCategory.GENERIC: 0.1,
    BenchmarkCategory.ENVIRONMENT: 0.0,
}
```

## Benchmark Selection

You can selectively run specific benchmarks:

```python
# Run only specific benchmark classes
from ttsds.benchmarks.speaker import DVectorBenchmark
from ttsds.benchmarks.prosody import PitchBenchmark

custom_benchmarks = {
    "dvector": DVectorBenchmark,
    "pitch": PitchBenchmark
}

suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    benchmarks=custom_benchmarks
)
```

## Parallel Processing

To speed up benchmarking with parallel processing:

```python
# Use 4 worker processes
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    n_workers=4
)
```

## Multilingual Support

For multilingual evaluation:

```python
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    multilingual=True
)
```

This enables additional language-specific benchmarks and evaluation criteria.


## Quick Links

- [Installation](user-guide/installation.md)
- [Quick Start](user-guide/quickstart.md)
- [*Configuration*](user-guide/configuration.md)
- [Benchmarks](reference/benchmarks.md)
- [Paper (TTSDS1)](https://arxiv.org/abs/2407.12707)
- [Paper (TTSDS2)](https://arxiv.org/abs/2506.19441)
- [Website](https://ttsdsbenchmark.com)