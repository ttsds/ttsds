---
layout: default
title: TTSDS Main Module
---

# TTSDS Main Module

This page documents the main module of TTSDS.

## BenchmarkSuite

```python
from ttsds import BenchmarkSuite
```

The `BenchmarkSuite` class is the main entry point for evaluating TTS systems. It manages the execution of multiple benchmarks and aggregates the results.

### Parameters

- `datasets`: List of datasets to evaluate
- `reference_datasets`: List of reference datasets for comparison
- `noise_datasets`: Optional list of noise datasets for normalization (default: predefined noise datasets)
- `benchmarks`: Dictionary mapping benchmark names to benchmark classes (default: predefined benchmarks)
- `category_weights`: Dictionary mapping benchmark categories to weights (default: equal weights for all categories except environment)
- `skip_errors`: Whether to skip errors during benchmark execution (default: False)
- `write_to_file`: Path to save results as CSV (default: None)
- `benchmark_kwargs`: Additional keyword arguments for benchmarks (default: {})
- `device`: Device to run benchmarks on ('cpu' or 'cuda') (default: 'cpu')
- `cache_dir`: Directory to store cached results (default: None)
- `include_environment`: Whether to include environment benchmarks (default: False)
- `multilingual`: Whether to use multilingual benchmarks (default: False)
- `n_workers`: Number of parallel workers for distance computation (default: number of CPU cores)

### Methods

#### `run()`

Run all benchmarks and collect results.

```python
suite = BenchmarkSuite(datasets=datasets, reference_datasets=reference_datasets)
results = suite.run()
```

Returns a pandas DataFrame containing benchmark results.

#### `get_aggregated_results()`

Get aggregated results with scores for each category and overall score.

```python
aggregated = suite.get_aggregated_results()
print(f"Overall score: {aggregated[aggregated['benchmark_category'] == 'OVERALL']['score_mean'].values[0]}")
print(f"Speaker score: {aggregated[aggregated['benchmark_category'] == 'SPEAKER']['score_mean'].values[0]}")
```

Returns a pandas DataFrame with aggregated scores by benchmark category.

#### `aggregate_df(df)`

Aggregate benchmark results by category and dataset.

```python
df = suite.database
aggregated = suite.aggregate_df(df)
```

#### `stop()`

Stop the benchmark suite and clean up resources.

```python
suite.stop()
```

## Exported Functionality

The `ttsds` package exports:

```python
from ttsds import BenchmarkSuite, __version__
```

## Example Usage

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
    write_to_file="results.csv",
)

# Run benchmarks
results = suite.run()

# Get aggregated results
aggregated = suite.get_aggregated_results()
print(f"Overall score: {aggregated[aggregated['benchmark_category'] == 'OVERALL']['score_mean'].values[0]}")
``` 