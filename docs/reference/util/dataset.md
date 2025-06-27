---
layout: default
title: Dataset Utilities
---

# Dataset Utilities

TTSDS provides dataset utilities for loading and managing audio files for benchmarking.

## Dataset Classes

TTSDS includes several dataset classes for different sources:

- `Dataset`: Abstract base class for all dataset types
- `DirectoryDataset`: Dataset from a directory of WAV files
- `TarDataset`: Dataset from a TAR archive containing WAV files
- `WavListDataset`: Dataset from explicit lists of WAV and text files

### Abstract Base Class

```python
from ttsds.util.dataset import Dataset
```

This is an abstract base class that all other dataset types inherit from. It provides common functionality for dataset operations.

### DirectoryDataset

```python
from ttsds.util.dataset import DirectoryDataset

# Create a dataset from a directory
dataset = DirectoryDataset(
    root_dir="path/to/dataset",
    sample_rate=22050,
    has_text=False,
    text_suffix=".txt",
    name="my_dataset"
)
```

#### Parameters

- `root_dir`: Path to the directory containing WAV files
- `sample_rate`: Sampling rate for audio in Hz (default: 22050)
- `has_text`: Whether to load associated text files (default: False)
- `text_suffix`: Suffix for text files (default: ".txt")
- `name`: Name of the dataset (default: directory name)

### TarDataset

```python
from ttsds.util.dataset import TarDataset

# Create a dataset from a TAR archive
dataset = TarDataset(
    root_tar="path/to/archive.tar.gz",
    sample_rate=22050,
    has_text=False,
    text_suffix=".txt",
    path_prefix=None,
    name="my_dataset"
)
```

#### Parameters

- `root_tar`: Path to the TAR archive containing WAV files
- `sample_rate`: Sampling rate for audio in Hz (default: 22050)
- `has_text`: Whether to load associated text files (default: False)
- `text_suffix`: Suffix for text files (default: ".txt")
- `path_prefix`: Prefix for paths within the archive (default: None)
- `name`: Name of the dataset (default: archive name)

### WavListDataset

```python
from ttsds.util.dataset import WavListDataset
from pathlib import Path

# Create a dataset from lists of WAV and text files
wavs = [Path("file1.wav"), Path("file2.wav")]
texts = [Path("file1.txt"), Path("file2.txt")]

dataset = WavListDataset(
    sample_rate=22050,
    has_text=True,
    wavs=wavs,
    texts=texts,
    name="my_dataset"
)
```

#### Parameters

- `sample_rate`: Sampling rate for audio in Hz (default: 22050)
- `has_text`: Whether to load associated text files (default: False)
- `wavs`: List of Path objects pointing to WAV files
- `texts`: List of Path objects pointing to text files (if has_text is True)
- `name`: Name of the dataset (default: "WavListDataset")

## Common Methods

All dataset classes provide these common methods:

### `__len__()`

Returns the number of samples in the dataset.

```python
print(f"Dataset contains {len(dataset)} audio files")
```

### `__getitem__(idx)`

Retrieves a sample from the dataset.

```python
# Get the first sample
audio, text = dataset[0]
```

Returns a tuple containing:
- `audio`: NumPy array containing audio data
- `text`: Text content (or None if has_text is False)

### `sample(n, seed)`

Creates a sampled subset of the dataset.

```python
# Sample 10 items with seed 42
sampled_dataset = dataset.sample(10, seed=42)
```

### `iter_with_progress(benchmark)`

Iterates over the dataset with a progress bar.

```python
for audio, text in dataset.iter_with_progress(benchmark):
    # Process each sample
    pass
```

## DataDistribution Class

The `DataDistribution` class computes and stores benchmark distributions for datasets.

```python
from ttsds.util.dataset import DataDistribution
from ttsds.benchmarks import BENCHMARKS

# Create a distribution
distribution = DataDistribution(
    dataset=dataset,
    benchmarks=BENCHMARKS,
    name="my_distribution"
)

# Save distribution to a file
distribution.to_pickle("my_distribution.pkl.gz")

# Load a distribution from a file
loaded_distribution = DataDistribution.from_pickle("my_distribution.pkl.gz")
``` 