---
layout: default
title: Quick Start
---

# Quick Start Guide

This guide will help you get started with TTSDS quickly. We'll show you how to set up and run a basic benchmark for evaluating Text-to-Speech systems.

## Basic Example

Here's a simple example to get started with TTSDS:

```python
from ttsds import BenchmarkSuite
from ttsds.util.dataset import DirectoryDataset

# Initialize datasets
datasets = [
    DirectoryDataset("path/to/your/dataset", name="your_dataset")
]
reference_datasets = [
    DirectoryDataset("path/to/reference/dataset", name="reference")
]

# Create benchmark suite
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    write_to_file="results.csv",  # Optional: save results to CSV
    skip_errors=True,  # Optional: skip failed benchmarks
    include_environment=False,  # Optional: exclude environment benchmarks
)

# Run benchmarks
results = suite.run()

# Get aggregated results with weighted scores
aggregated = suite.get_aggregated_results()
print(aggregated)
```

## Understanding the Datasets

The datasets should be directories containing wav files. Since TTSDS is a distributional score, the wav files do not need to include the same content, and the number of files can vary between datasets. However, results are best when the speaker identities are the same.

## Custom Category Weights

You can customize the importance of different evaluation categories:

```python
from ttsds.benchmarks.benchmark import BenchmarkCategory

suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    category_weights={
        BenchmarkCategory.SPEAKER: 0.25,
        BenchmarkCategory.INTELLIGIBILITY: 0.25,
        BenchmarkCategory.PROSODY: 0.25,
        BenchmarkCategory.GENERIC: 0.25,
        BenchmarkCategory.ENVIRONMENT: 0.0,
    },
)
```

## Multilingual Support

TTSDS supports multilingual evaluation:

```python
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    multilingual=True,
)
```

## Progress Display

When running benchmarks, TTSDS provides a real-time progress display showing:
- Overall progress
- Per-benchmark completion status
- Estimated time remaining
- Error messages (if any)

## Understanding Results

After running the benchmark, you'll get:

1. **Individual benchmark scores**: Detailed scores for each benchmark
2. **Category scores**: Aggregated scores for each category
3. **Overall score**: Weighted average across all categories

## Saving Results

You can save benchmark results to a CSV file for further analysis:

```python
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    write_to_file="results.csv",
)
```

## Next Steps

- Learn about [advanced configuration options](configuration.md)
- Explore the [API reference](../reference/benchmarks.md)
- See [examples](advanced.md) of more complex use cases 