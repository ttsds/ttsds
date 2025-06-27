---
layout: default
title: Configuration
---

# Configuration

This guide explains the various configuration options available in TTSDS to customize your benchmarking setup.

## BenchmarkSuite Parameters

The `BenchmarkSuite` class accepts several parameters to customize its behavior:

```python
from ttsds import BenchmarkSuite
from ttsds.util.dataset import Dataset
from ttsds.benchmarks.benchmark import BenchmarkCategory

suite = BenchmarkSuite(
    datasets=[Dataset("path/to/dataset", name="my_dataset")],
    reference_datasets=[Dataset("path/to/reference", name="reference")],
    noise_datasets=None,  # Optional: for environment benchmarks
    category_weights={  # Optional: customize category importance
        BenchmarkCategory.SPEAKER: 0.25,
        BenchmarkCategory.INTELLIGIBILITY: 0.25,
        BenchmarkCategory.PROSODY: 0.25,
        BenchmarkCategory.GENERIC: 0.25,
        BenchmarkCategory.ENVIRONMENT: 0.0,
    },
    write_to_file="results.csv",  # Optional: save results to CSV
    skip_errors=True,  # Optional: skip failed benchmarks
    include_environment=False,  # Optional: exclude environment benchmarks
    multilingual=False,  # Optional: enable multilingual evaluation
    benchmark_classes=None,  # Optional: specify benchmark classes to run
    benchmark_names=None,  # Optional: specify benchmark names to run
    progress_bar=True,  # Optional: show progress bar
    results_folder=None,  # Optional: folder for additional result files
    cache_folder=None,  # Optional: override default cache location
    cache=True,  # Optional: enable or disable caching
    num_workers=1,  # Optional: number of parallel workers
)
```

## Dataset Configuration

The `Dataset` class allows you to configure how your speech datasets are loaded:

```python
from ttsds.util.dataset import Dataset

# Basic dataset with default settings
dataset = Dataset("path/to/dataset", name="my_dataset")

# Dataset with advanced configuration
dataset = Dataset(
    path="path/to/dataset",
    name="my_dataset",
    extensions=[".wav", ".flac"],  # File extensions to include
    recursive=True,  # Search subdirectories recursively
    min_duration=1.0,  # Minimum audio duration in seconds
    max_duration=30.0,  # Maximum audio duration in seconds
    limit=None,  # Maximum number of files to use
    shuffle=True,  # Shuffle the file order
    random_seed=42,  # Random seed for shuffling
    sample_rate=16000,  # Target sample rate
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
from ttsds.benchmarks.speaker import SPKBenchmark
from ttsds.benchmarks.prosody import PitchBenchmark

suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    benchmark_classes=[SPKBenchmark, PitchBenchmark]
)

# Or specify benchmark names
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    benchmark_names=["SPK", "Pitch"]
)
```

## Caching

TTSDS uses caching to speed up repeated benchmark runs. You can configure this behavior:

```python
# Disable caching
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    cache=False
)

# Specify a custom cache folder
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    cache_folder="/path/to/custom/cache"
)
```

You can also set the cache directory using an environment variable:

```bash
export TTSDS_CACHE_DIR=/path/to/cache
```

## Parallel Processing

To speed up benchmarking with parallel processing:

```python
# Use 4 worker processes
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    num_workers=4
)
```

## Environment Variables

TTSDS respects the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `TTSDS_CACHE_DIR` | Directory for cached data | `~/.cache/ttsds` |
| `TTSDS_RESULTS_DIR` | Directory for saving results | Current directory |
| `TTSDS_NUM_WORKERS` | Number of worker processes | `1` |

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

## Next Steps

- Explore [advanced usage](advanced.md) examples
- Learn about the [API reference](../reference/benchmarks.md) 