---
layout: default
title: Testing
---

# Testing Guide

This guide outlines the testing practices for the TTSDS project. Thorough testing ensures reliability and maintains code quality as the project evolves.

## Testing Philosophy

The TTSDS project follows these testing principles:

1. **Test-Driven Development**: Write tests before implementing features when possible
2. **Comprehensive Coverage**: Aim for high test coverage across all modules
3. **Fast Execution**: Tests should run quickly to encourage frequent execution
4. **Realistic Scenarios**: Tests should reflect real-world usage

## Test Organization

Tests are organized in the `tests/` directory, mirroring the structure of the main package:

```
tests/
├── benchmarks/            # Tests for benchmark modules
│   ├── test_benchmark.py  # Tests for base benchmark class
│   ├── test_speaker.py    # Tests for speaker benchmarks
│   ├── test_prosody.py    # Tests for prosody benchmarks
│   └── ...
├── util/                  # Tests for utility modules
│   ├── test_audio.py      # Tests for audio utilities
│   ├── test_cache.py      # Tests for caching utilities
│   └── ...
├── test_ttsds.py          # Tests for main module
└── conftest.py            # Shared pytest fixtures
```

## Running Tests

TTSDS uses [pytest](https://docs.pytest.org/) as its testing framework. To run all tests:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/benchmarks/test_speaker.py

# Run a specific test
pytest tests/benchmarks/test_speaker.py::test_spk_benchmark_initialization
```

## Writing Tests

### Test File Naming

- Test files should be named `test_<module_name>.py`
- Test functions should be named `test_<function_or_feature_name>`

### Test Function Structure

Each test function should follow the Arrange-Act-Assert (AAA) pattern:

1. **Arrange**: Set up the test environment and inputs
2. **Act**: Execute the function or code being tested
3. **Assert**: Verify the results match expectations

```python
def test_calculate_score():
    # Arrange
    features1 = np.array([1.0, 2.0, 3.0])
    features2 = np.array([1.1, 2.1, 3.1])
    
    # Act
    score = calculate_score(features1, features2)
    
    # Assert
    assert 0 <= score <= 1
    assert np.isclose(score, 0.95, atol=0.1)
```

### Test Fixtures

Use pytest fixtures for common setup and test data:

```python
import pytest
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_audio_path():
    """Return path to a sample audio file."""
    return Path(__file__).parent / "data" / "sample.wav"

@pytest.fixture
def sample_features():
    """Return sample feature arrays."""
    return np.random.rand(10, 20)

def test_feature_extraction(sample_audio_path):
    features = extract_features(sample_audio_path)
    assert features.shape[0] > 0
    assert features.ndim == 2
```

Common fixtures should be defined in `conftest.py` to make them available to all tests.

## Types of Tests

TTSDS includes several types of tests:

### 1. Unit Tests

Test individual functions and classes in isolation:

```python
def test_js_divergence():
    """Test Jensen-Shannon divergence calculation."""
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.1, 0.4, 0.5])
    
    div = calculate_js_divergence(p, q)
    
    assert 0 <= div <= 1
    assert np.isclose(div, 0.33, atol=0.1)
```

### 2. Integration Tests

Test interactions between components:

```python
def test_benchmark_suite_integration():
    """Test BenchmarkSuite with real benchmarks."""
    dataset = DirectoryDataset("path/to/test_data", name="test")
    reference = DirectoryDataset("path/to/reference", name="reference")
    
    suite = BenchmarkSuite(
        datasets=[dataset],
        reference_datasets=[reference],
        benchmark_names=["dvector", "pitch"]
    )
    
    results = suite.run()
    
    assert len(results) == 2
    assert "dvector" in [r["benchmark"] for r in results]
    assert "pitch" in [r["benchmark"] for r in results]
```

### 3. Parametrized Tests

Use pytest's parametrization to test multiple inputs:

```python
@pytest.mark.parametrize("method", ["praat", "swipe", "world"])
def test_pitch_extraction_methods(sample_audio_path, method):
    """Test different pitch extraction methods."""
    waveform, sr = load_audio(sample_audio_path)
    
    times, pitch = extract_pitch(waveform, sr, method=method)
    
    assert len(times) == len(pitch)
    assert np.all(pitch >= 0)
```

### 4. Property Tests

Test properties that should hold for a wide range of inputs:

```python
@pytest.mark.parametrize("size", [10, 100, 1000])
def test_divergence_properties(size):
    """Test mathematical properties of divergence metrics."""
    # Generate random distributions
    p = np.random.random(size)
    p = p / p.sum()
    q = np.random.random(size)
    q = q / q.sum()
    
    # JS divergence should be symmetric
    js1 = calculate_js_divergence(p, q)
    js2 = calculate_js_divergence(q, p)
    assert np.isclose(js1, js2, atol=1e-10)
    
    # Identical distributions should have zero divergence
    js3 = calculate_js_divergence(p, p)
    assert np.isclose(js3, 0, atol=1e-10)
```

## Mocking

Use the `unittest.mock` module or pytest's monkeypatch fixture to isolate components:

```python
from unittest.mock import patch, MagicMock

def test_feature_caching():
    """Test that features are cached properly."""
    with patch("ttsds.util.cache.save_cache") as mock_save:
        benchmark = DVectorBenchmark(cache=True)
        benchmark.extract_features(["path/to/audio.wav"])
        
        # Check that the cache was saved
        assert mock_save.called
```

## Test Data

Store test audio files and other test data in the `tests/data` directory. Keep test files small but representative.

For audio files, consider:
- Using short clips (1-2 seconds)
- Including files with different characteristics (speech, silence, noise)
- Supporting multiple formats (WAV, FLAC, etc.)

## Code Coverage

Monitor test coverage to ensure all code paths are tested:

```bash
# Run tests with coverage
pytest --cov=ttsds

# Generate HTML coverage report
pytest --cov=ttsds --cov-report=html
```

Aim for at least 80% coverage for core functionality.

## Test Performance

Optimize tests for speed:

- Use small, synthetic data when possible
- Cache expensive computations
- Use parametrization instead of loops
- Skip or mark slow tests appropriately:

```python
@pytest.mark.slow
def test_large_dataset_processing():
    """Test processing of a large dataset (slow)."""
    # ...test implementation...
```

Run only fast tests during development:

```bash
pytest -k "not slow"
```

## Continuous Integration

TTSDS uses GitHub Actions for continuous integration. Tests are automatically run on:

- Push to main branch
- Pull requests
- Scheduled runs (weekly)

The CI configuration can be found in `.github/workflows/tests.yml`.

## Common Testing Patterns

### Testing Exceptions

Test that functions raise appropriate exceptions:

```python
def test_invalid_input_handling():
    """Test handling of invalid inputs."""
    with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
        calculate_score(np.array([1, 2]), np.array([1, 2]), weight=1.5)
```

### Testing Warnings

Test that functions issue appropriate warnings:

```python
def test_deprecated_feature_warning():
    """Test that deprecated features issue warnings."""
    with pytest.warns(DeprecationWarning, match="function is deprecated"):
        legacy_function()
```

## Next Steps

- Review the [Development Guide](development.md) for workflow information
- Check the [Code Style Guide](code-style.md) for coding standards