---
layout: default
title: Benchmark Base Class
---

# Benchmark Base Class

The `Benchmark` class is the base class for all benchmarks in TTSDS. It defines the common interface and functionality that all benchmark implementations should follow.

## Class Definition

```python
from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory

class Benchmark:
    def __init__(self, name, category, **kwargs):
        """Initialize a benchmark.
        
        Args:
            name: Name of the benchmark
            category: Category of the benchmark (from BenchmarkCategory enum)
            **kwargs: Additional parameters
        """
        self.name = name
        self.category = category
        # Additional initialization
    
    def extract_features(self, audio_paths):
        """Extract features from audio files.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of extracted features
        """
        raise NotImplementedError("Subclasses must implement extract_features")
    
    def calculate_score(self, dataset_features, reference_features):
        """Calculate score based on feature comparison.
        
        Args:
            dataset_features: Features from the dataset being evaluated
            reference_features: Features from the reference dataset
            
        Returns:
            float: Score between 0 and 1
        """
        raise NotImplementedError("Subclasses must implement calculate_score")
```

## Benchmark Categories

The `BenchmarkCategory` enum defines the categories that benchmarks can belong to:

```python
from enum import Enum, auto

class BenchmarkCategory(Enum):
    SPEAKER = auto()
    INTELLIGIBILITY = auto()
    PROSODY = auto()
    GENERIC = auto()
    ENVIRONMENT = auto()
```

## Creating Custom Benchmarks

To create a custom benchmark, extend the `Benchmark` class and implement the required methods:

```python
class CustomBenchmark(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(
            name="Custom",
            category=BenchmarkCategory.GENERIC,
            **kwargs
        )
    
    def extract_features(self, audio_paths):
        # Implement feature extraction
        features = []
        for path in audio_paths:
            # Process each audio file
            # ...
            features.append(processed_features)
        return features
    
    def calculate_score(self, dataset_features, reference_features):
        # Implement score calculation
        # ...
        return score
``` 