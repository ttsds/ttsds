---
layout: default
title: Code Style
---

# Code Style Guide

This guide outlines the coding style and best practices for contributing to TTSDS.

## General Principles

- **Readability**: Write code that is easy to read and understand
- **Simplicity**: Keep code simple and avoid unnecessary complexity
- **Consistency**: Follow established patterns in the codebase
- **Documentation**: Document your code thoroughly

## Python Style Guide

TTSDS follows [PEP 8](https://www.python.org/dev/peps/pep-0008/), the official style guide for Python code, with a few project-specific guidelines.

### Formatting

- Use 4 spaces for indentation (no tabs)
- Keep lines to a maximum of 100 characters
- Use blank lines to separate logical sections of code
- Use spaces around operators and after commas
- Use parentheses sparingly but when needed for clarity

```python
# Good
def calculate_score(features1, features2, weight=0.5):
    score = weight * metric1(features1, features2) + (1 - weight) * metric2(features1, features2)
    return score

# Bad - line too long, inconsistent spacing
def calculate_score(features1,features2,weight = 0.5):
    score=weight*metric1(features1, features2)+(1-weight)*metric2(features1, features2)
    return score
```

### Naming Conventions

- **Classes**: Use `CamelCase` for class names
- **Functions and Variables**: Use `snake_case` for function and variable names
- **Constants**: Use `UPPER_CASE` for constants
- **Private Methods and Variables**: Prefix with underscore (e.g., `_private_method`)
- **Benchmark Classes**: Use a descriptive name followed by "Benchmark" (e.g., `PitchBenchmark`)

```python
# Good
MAX_CACHE_SIZE = 1024

class PitchBenchmark:
    def __init__(self, extraction_method="praat"):
        self._extraction_method = extraction_method
    
    def extract_features(self, audio_paths):
        feature_list = []
        for path in audio_paths:
            features = self._extract_single(path)
            feature_list.append(features)
        return feature_list
    
    def _extract_single(self, path):
        # Implementation details
        pass

# Bad
class pitch_benchmark:
    def __init__(self, extractionMethod="praat"):
        self.extractionMethod = extractionMethod
    
    def ExtractFeatures(self, AudioPaths):
        FeatureList = []
        for Path in AudioPaths:
            Features = self.extract_single(Path)
            FeatureList.append(Features)
        return FeatureList
    
    def extract_single(self, path):
        # Implementation details
        pass
```

## Docstrings

Use Google-style docstrings for all modules, classes, methods, and functions:

```python
def calculate_score(features1, features2, weight=0.5):
    """Calculate a weighted score between two feature sets.
    
    This function computes a weighted average of two different metrics
    applied to the input features.
    
    Args:
        features1: Array of features from the first dataset.
        features2: Array of features from the second dataset.
        weight: Float between 0 and 1 that weights the contribution
            of the first metric. Default is 0.5 (equal weighting).
    
    Returns:
        float: The calculated score between 0 and 1, where higher
            values indicate greater similarity.
    
    Raises:
        ValueError: If weight is not between 0 and 1.
    """
    if not 0 <= weight <= 1:
        raise ValueError("Weight must be between 0 and 1.")
    
    score = weight * metric1(features1, features2) + (1 - weight) * metric2(features1, features2)
    return score
```

### Module Docstrings

Each module should have a docstring at the top describing its purpose:

```python
"""Speaker verification benchmarks for TTSDS.

This module contains benchmarks for evaluating how well a TTS system
preserves speaker identity characteristics compared to reference speech.
"""
```

## Type Hints

Use type hints to improve code clarity and enable static type checking:

```python
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

def extract_pitch(
    waveform: np.ndarray,
    sample_rate: int,
    method: str = "praat"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract pitch contour from audio.
    
    Args:
        waveform: Audio waveform.
        sample_rate: Sampling rate in Hz.
        method: Pitch extraction method. Default is "praat".
    
    Returns:
        Tuple containing:
            - times: Array of time points in seconds.
            - pitch: Array of pitch values in Hz.
    """
    # Implementation
    times = np.array([0.0, 0.01, 0.02])
    pitch = np.array([100.0, 120.0, 110.0])
    return times, pitch
```

## Imports

Organize imports in the following order, with a blank line between each group:

1. Standard library imports
2. Related third-party imports
3. Local application/library-specific imports

Within each group, imports should be alphabetically sorted:

```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
import torch
from scipy import signal

# Local
from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory
from ttsds.util.audio import load_audio
```

Use absolute imports within the package to avoid potential confusion:

```python
# Good
from ttsds.util.audio import load_audio

# Avoid relative imports unless necessary
from ..util.audio import load_audio
```

## Error Handling

Use explicit error handling with specific exception types:

```python
# Good
def load_feature_cache(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        logger.warning(f"Cache file not found: {path}")
        return None
    except (pickle.PickleError, EOFError) as e:
        logger.error(f"Error loading cache from {path}: {e}")
        return None

# Bad
def load_feature_cache(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    except:  # Too broad!
        return None
```

## Comments

- Use comments sparingly and only when necessary to explain complex logic
- Focus on writing self-documenting code with clear variable and function names
- Always keep comments up-to-date with code changes

```python
# Good: Comment explains the non-obvious aspect
# Use zero-padding to ensure frequency resolution matches the reference
fft_size = 2048  # Next power of 2 above max_length

# Bad: Comment just repeats what the code does
# Calculate mean
mean = np.mean(values)
```

## Automated Code Quality Tools

TTSDS uses several tools to maintain code quality:

### Black

We use [Black](https://black.readthedocs.io/) for code formatting with a 100-character line limit:

```bash
black --line-length 100 .
```

### isort

[isort](https://pycqa.github.io/isort/) is used to sort imports:

```bash
isort --profile black .
```

### flake8

[flake8](https://flake8.pycqa.org/) is used for linting:

```bash
flake8 --max-line-length 100 .
```

### mypy

[mypy](http://mypy-lang.org/) is used for type checking:

```bash
mypy .
```

## Best Practices

### Use Pathlib for File Operations

Prefer `pathlib.Path` over string operations for file paths:

```python
# Good
from pathlib import Path

cache_dir = Path.home() / ".cache" / "ttsds"
feature_file = cache_dir / f"{benchmark_name}_{dataset_name}.pkl"
feature_file.parent.mkdir(parents=True, exist_ok=True)

# Less good
import os

cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "ttsds")
feature_file = os.path.join(cache_dir, f"{benchmark_name}_{dataset_name}.pkl")
os.makedirs(os.path.dirname(feature_file), exist_ok=True)
```

### Avoid Global State

Minimize the use of global variables and state:

```python
# Good: State is encapsulated in a class
class BenchmarkSuite:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "ttsds"
        self.results = []
    
    def run(self):
        # Implementation
        pass

# Bad: Global state
CACHE_DIR = Path.home() / ".cache" / "ttsds"
RESULTS = []

def run_benchmarks():
    global RESULTS
    # Implementation that modifies RESULTS
    pass
```

### Use Context Managers

Use context managers for resource management:

```python
# Good
def save_features(features, path):
    with open(path, "wb") as f:
        pickle.dump(features, f)

# Bad
def save_features(features, path):
    f = open(path, "wb")
    pickle.dump(features, f)
    f.close()  # Could be missed if an exception occurs
```

## Next Steps

- Check out the [Development Guide](development.md) for workflow information
- Learn about [Testing](testing.md) guidelines 