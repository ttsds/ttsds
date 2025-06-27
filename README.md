# TTSDS - Text-to-Speech Distribution Score

[![PyPI - Version](https://img.shields.io/pypi/v/ttsds.svg)](https://pypi.org/project/ttsds) [![Tests](https://raw.githubusercontent.com/ttsds/ttsds/main/docs/assets/img/tests.svg)](https://github.com/ttsds/ttsds/actions) [![Coverage](https://raw.githubusercontent.com/ttsds/ttsds/main/docs/assets/img/coverage.svg)](https://github.com/ttsds/ttsds/actions)

<img src="https://api.star-history.com/svg?repos=ttsds/ttsds&type=Date" style="width:45em">

TTSDS is a comprehensive benchmark for evaluating the quality of synthetic speech in Text-to-Speech (TTS) systems. It assesses multiple aspects of speech quality including prosody, speaker identity, and intelligibility by comparing synthetic speech with both real speech and noise datasets.

## Version 2.1.0

We are excited to release TTSDS 2.1.0!
TTSDS2 is multilingual and updated quarterly, with a new dataset every time: you can view the results at https://ttsdsbenchmark.com#leaderboard.



## Features

- **Multi-dimensional Evaluation**: Assess speech quality across different categories:
  - Prosody (e.g., pitch, speaking rate)
  - Speaker Identity (e.g., speaker verification)
  - Intelligibility (e.g., speech recognition)
  - Generic Features (e.g., embeddings)
  - Environment (e.g., noise robustness)

- **Weighted Scoring**: Customizable weights for different evaluation categories
- **Progress Tracking**: Real-time progress display with detailed statistics
- **Caching**: Efficient caching of intermediate results
- **Error Handling**: Robust error handling with optional skipping of failed benchmarks

## Installation

### System Requirements

```bash
# Required system packages
sudo apt-get install ffmpeg automake autoconf unzip sox gfortran subversion libtool
```

### Python Installation

```bash
# Basic installation
pip install ttsds
```

### Optional: Fairseq Installation

If you encounter dependency conflicts with fairseq, use this fork:
```bash
pip install git+https://github.com/MiniXC/fairseq-noconf
```

### Development Installation

For development, install with the dev extras:

```bash
# Clone the repository
git clone https://github.com/ttsds/ttsds.git
cd ttsds

# Install with development dependencies
pip install -e ".[dev]"
```

## Usage

### Basic Example

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

The datasets should be directories containing wav files. Since this is a distributional score, the wav files do not need to include the same content, and the number of files can vary between datasets. However, results are best when the speaker identities are the same.

### Custom Category Weights

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

### Multilingual

```python
suite = BenchmarkSuite(
    datasets=datasets,
    reference_datasets=reference_datasets,
    multilingual=True,
)
```

### Progress Display

The benchmark suite provides a real-time progress display showing:
- Overall progress
- Per-benchmark completion status
- Estimated time remaining
- Error messages (if any)

## Configuration

### Environment Variables

```bash
# Set cache directory (default: ~/.cache/ttsds)
export TTSDS_CACHE_DIR=/path/to/cache
```

### Benchmark Categories

- **Speaker**: Evaluates speaker identity preservation
- **Intelligibility**: Measures speech recognition performance
- **Prosody**: Assesses speech rhythm and intonation
- **Generic**: General speech quality metrics
- **Environment**: Noise robustness evaluation - this is excluded by default, set `include_environment=True` to include it.

## Results

The benchmark results include:
- Individual benchmark scores
- Category-wise aggregated scores
- Overall weighted score
- Time taken for each benchmark
- Reference and noise dataset information

Results can be saved to a CSV file for further analysis.

## Development

### Running Tests

TTSDS includes a comprehensive test suite covering its functionality:

```bash
# Run all tests
cd ttsds
./tests/run_tests.py

# Run specific test modules or classes
./tests/run_tests.py tests/unit/benchmarks/test_benchmark.py
./tests/run_tests.py tests/unit/test_ttsds.py::test_benchmark_suite_init

# Run with coverage report
./tests/run_tests.py --cov-report=html
```

The test suite uses pytest and includes:
- Unit tests for individual components
- Integration tests for the full system
- Test coverage reporting

### Documentation

The API documentation is automatically generated from docstrings using mkdocstrings:

```bash
# Build the documentation
pip install -e ".[dev]"
mkdocs build

# Serve the documentation locally
mkdocs serve
```

## Citation

```bibtex
@inproceedings{minixhofer2024ttsds,
  title={TTSDS-Text-to-Speech Distribution Score},
  author={Minixhofer, Christoph and Klejch, Ond{\v{r}}ej and Bell, Peter},
  booktitle={SLT},
  year={2024},
}

```


## License

`ttsds` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Links

- [Paper](https://arxiv.org/abs/2407.12707)
- [HuggingFace Space](https://huggingface.co/spaces/ttsds/benchmark)
- [Website](https://ttsdsbenchmark.com)
