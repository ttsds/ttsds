---
layout: default
title: Advanced Usage
---

# Advanced Usage

This guide demonstrates more advanced use cases and techniques for working with TTSDS.

## Comparing Multiple TTS Systems

This example shows how to evaluate and compare multiple TTS systems:

```python
from ttsds import BenchmarkSuite
from ttsds.util.dataset import Dataset

# Define reference dataset (real speech)
reference = Dataset("path/to/real_speech", name="reference")

# Define TTS system outputs
tts_systems = [
    Dataset("path/to/system1_output", name="system1"),
    Dataset("path/to/system2_output", name="system2"),
    Dataset("path/to/system3_output", name="system3"),
]

# Run benchmarks for each system
results = {}
for tts_system in tts_systems:
    suite = BenchmarkSuite(
        datasets=[tts_system],
        reference_datasets=[reference],
        write_to_file=f"results_{tts_system.name}.csv",
    )
    suite.run()
    results[tts_system.name] = suite.get_aggregated_results()

# Compare results
for system_name, result in results.items():
    print(f"System: {system_name}")
    print(f"Overall score: {result['overall']:.4f}")
    print(f"Speaker score: {result['speaker']:.4f}")
    print(f"Intelligibility score: {result['intelligibility']:.4f}")
    print(f"Prosody score: {result['prosody']:.4f}")
    print(f"Generic score: {result['generic']:.4f}")
    print(f"Environment score: {result['environment']:.4f}")
    print("-" * 40)
```

## Custom Benchmark Selection

You can run specific benchmarks based on your needs:

```python
from ttsds import BenchmarkSuite
from ttsds.benchmarks.speaker import SPKBenchmark
from ttsds.benchmarks.prosody import PitchBenchmark, DurationBenchmark
from ttsds.benchmarks.intelligibility import STTBenchmark

# Run only speaker and prosody benchmarks
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    benchmark_classes=[SPKBenchmark, PitchBenchmark, DurationBenchmark],
)

# Or run specific benchmark names
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    benchmark_names=["SPK", "Pitch", "Duration", "STT"],
)
```

## Working with Cached Results

TTSDS caches intermediate results to speed up repeated runs. Here's how to manage the cache:

```python
from ttsds import BenchmarkSuite
from ttsds.util.cache import clear_cache

# Run with caching (default)
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    cache=True,
)

# Clear the entire cache
clear_cache()

# Clear cache for a specific dataset
clear_cache(dataset_name="your_dataset")

# Run without caching
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    cache=False,
)
```

## Custom Result Analysis

You can access and analyze the detailed benchmark results:

```python
from ttsds import BenchmarkSuite
import pandas as pd
import matplotlib.pyplot as plt

# Run the benchmark
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
)
suite.run()

# Get detailed results
detailed_results = suite.results

# Convert to DataFrame for analysis
results_df = pd.DataFrame(detailed_results)

# Calculate statistics
mean_scores = results_df.groupby('benchmark_category')['score'].mean()
std_scores = results_df.groupby('benchmark_category')['score'].std()

# Visualize results
plt.figure(figsize=(10, 6))
mean_scores.plot(kind='bar', yerr=std_scores, capsize=5)
plt.title('Benchmark Scores by Category')
plt.ylabel('Score')
plt.xlabel('Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('benchmark_results.png')
plt.show()
```

## Multilingual Evaluation

For evaluating TTS systems across multiple languages:

```python
from ttsds import BenchmarkSuite
from ttsds.util.dataset import Dataset

# Define reference datasets for different languages
reference_en = Dataset("path/to/english_reference", name="reference_en")
reference_fr = Dataset("path/to/french_reference", name="reference_fr")
reference_de = Dataset("path/to/german_reference", name="reference_de")

# Define TTS system outputs for different languages
tts_en = Dataset("path/to/english_tts", name="tts_en")
tts_fr = Dataset("path/to/french_tts", name="tts_fr")
tts_de = Dataset("path/to/german_tts", name="tts_de")

# Run multilingual benchmark
suite = BenchmarkSuite(
    datasets=[tts_en, tts_fr, tts_de],
    reference_datasets=[reference_en, reference_fr, reference_de],
    multilingual=True,
    write_to_file="multilingual_results.csv",
)
suite.run()
results = suite.get_aggregated_results()

# Analyze results by language
en_results = [r for r in suite.results if "tts_en" in r['dataset']]
fr_results = [r for r in suite.results if "tts_fr" in r['dataset']]
de_results = [r for r in suite.results if "tts_de" in r['dataset']]

# Calculate language-specific scores
def calculate_language_score(lang_results):
    if not lang_results:
        return 0
    return sum(r['score'] for r in lang_results) / len(lang_results)

print(f"English score: {calculate_language_score(en_results):.4f}")
print(f"French score: {calculate_language_score(fr_results):.4f}")
print(f"German score: {calculate_language_score(de_results):.4f}")
```

## Parallel Processing

For faster evaluation with large datasets:

```python
import multiprocessing
from ttsds import BenchmarkSuite

# Determine optimal number of workers based on CPU cores
num_cores = multiprocessing.cpu_count()
num_workers = max(1, num_cores - 1)  # Leave one core free

# Run benchmark with parallel processing
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    num_workers=num_workers,
)
suite.run()
```

## Next Steps

- Explore the [API reference](../reference/benchmarks.md) for detailed information on each module
- Check out the [contributing guide](../contributing/development.md) if you'd like to enhance TTSDS 