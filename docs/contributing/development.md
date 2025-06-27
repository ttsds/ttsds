---
layout: default
title: Development Guide
---

# Development Guide

This guide provides information for developers who want to contribute to TTSDS.

## Setting Up the Development Environment

### Prerequisites

Before starting development, ensure you have the following tools installed:

- Python 3.7 or higher
- Git
- FFmpeg and other system dependencies (see [Installation](../user-guide/installation.md))

### Clone the Repository

First, fork the TTSDS repository on GitHub, then clone your fork:

```bash
git clone https://github.com/your-username/ttsds.git
cd ttsds
```

### Create a Virtual Environment

We recommend using a virtual environment for development:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n ttsds-dev python=3.8
conda activate ttsds-dev
```

### Install Dependencies

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

This will install TTSDS in development mode along with all the required development tools.

4. Set up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## Project Structure

The TTSDS project is organized as follows:

```
ttsds/
├── benchmarks/         # Benchmark implementations
│   ├── benchmark.py    # Base benchmark class
│   ├── speaker/        # Speaker benchmarks
│   ├── prosody/        # Prosody benchmarks
│   ├── intelligibility/ # Intelligibility benchmarks
│   ├── generic/        # Generic benchmarks
│   └── environment/    # Environment benchmarks
├── util/               # Utility functions
│   ├── audio.py        # Audio processing utilities
│   ├── cache.py        # Caching utilities
│   ├── dataset.py      # Dataset utilities
│   ├── features.py     # Feature extraction utilities
│   ├── progress.py     # Progress tracking utilities
│   ├── stats.py        # Statistical utilities
│   └── visualization.py # Visualization utilities
├── ttsds.py            # Main module
├── __init__.py         # Package initialization
└── __about__.py        # Package metadata
```

## Development Workflow

### 1. Create a Branch

Always create a new branch for your changes:

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Implement your changes, following the coding style guidelines.

### 3. Run Tests

Before submitting your changes, run the tests to ensure everything is working:

```bash
pytest
```

### 4. Add Documentation

Update or add documentation for your changes. Make sure to:

- Add docstrings to new functions, classes, and methods
- Update relevant documentation files
- Add examples if applicable

### 5. Submit a Pull Request

Once your changes are ready and tests pass, push your branch and create a pull request:

```bash
git push origin feature/your-feature-name
```

Then go to GitHub and create a pull request against the main repository.

## Adding a New Benchmark

To add a new benchmark to TTSDS, follow these steps:

1. Identify which category your benchmark belongs to (speaker, prosody, intelligibility, generic, or environment)
2. Create a new file in the appropriate category directory
3. Implement your benchmark by extending the `Benchmark` base class
4. Add tests for your benchmark

Here's an example of implementing a new benchmark:

```python
from ttsds.benchmarks.benchmark import Benchmark, BenchmarkCategory

class MyNewBenchmark(Benchmark):
    def __init__(self, param1=default1, param2=default2, **kwargs):
        super().__init__(
            name="MyNew",
            category=BenchmarkCategory.GENERIC,
            **kwargs
        )
        self.param1 = param1
        self.param2 = param2
    
    def extract_features(self, audio_paths):
        features = []
        for path in audio_paths:
            # Extract features from audio
            # ...
            features.append(extracted_features)
        return features
    
    def calculate_score(self, dataset_features, reference_features):
        # Calculate score based on feature comparison
        # ...
        return score
```

## Coding Style Guidelines

TTSDS follows these coding style guidelines:

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use Google-style docstrings
- Keep lines under 100 characters
- Use meaningful variable and function names
- Add type hints where possible

We use the following tools to enforce these guidelines:

- [Black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [flake8](https://flake8.pycqa.org/) for linting
- [mypy](http://mypy-lang.org/) for type checking

You can run these tools using:

```bash
# Format code
black .

# Sort imports
isort .

# Lint
flake8

# Type check
mypy .
```

## Testing Guidelines

We use [pytest](https://docs.pytest.org/) for testing. When adding tests:

- Place tests in the `tests/` directory
- Name test files with a `test_` prefix
- Name test functions with a `test_` prefix
- Use descriptive test names
- Write tests for positive and negative cases
- Use fixtures when appropriate

Example test:

```python
import pytest
from ttsds.benchmarks.speaker import DVectorBenchmark

def test_dvector_benchmark_initialization():
    benchmark = DVectorBenchmark(model_name="test_model")
    assert benchmark.name == "dvector"
    assert benchmark.category.name == "SPEAKER"
    assert benchmark.model_name == "test_model"

def test_dvector_benchmark_feature_extraction(sample_audio_paths):
    benchmark = DVectorBenchmark()
    features = benchmark.extract_features(sample_audio_paths)
    assert len(features) == len(sample_audio_paths)
    assert all(isinstance(f, np.ndarray) for f in features)
```

## Documentation Guidelines

Documentation is written in Markdown and built with MkDocs. When updating documentation:

- Keep the documentation up to date with code changes
- Use clear, concise language
- Provide examples for complex features
- Include both API documentation and usage guides

## Release Process

TTSDS follows semantic versioning (MAJOR.MINOR.PATCH). The release process is:

1. Update version in `__about__.py`
2. Update CHANGELOG.md
3. Create a release branch: `release/vX.Y.Z`
4. Create a pull request for the release
5. After approval, merge to main
6. Tag the release: `git tag vX.Y.Z`
7. Push the tag: `git push origin vX.Y.Z`
8. Build and publish to PyPI:
   ```bash
   python -m build
   python -m twine upload dist/*
   ```

## Getting Help

If you need help or have questions about contributing to TTSDS:

- Open an issue on GitHub
- Reach out to the maintainers
- Check the documentation for guidance

## Testing and Coverage

The project includes comprehensive testing infrastructure:

1. **Running Tests**: 
   ```bash
   python tests/run_tests.py
   ```

2. **Running All Tests (including speaker benchmarks)**:
   ```bash
   python tests/run_tests.py --include-all
   ```

3. **Coverage Reports**:
   ```bash
   python tests/run_tests.py --cov-report=html
   ```
   
4. **Test and Coverage Badges**:
   ```bash
   python scripts/generate_badges.py
   ```
   This generates badges for test status and coverage in `docs/assets/img/`.

5. **Current Status**:
   - All tests are passing
   - Current coverage is 31%
   - Tests use mock implementations to avoid dependencies
   - Badge generation is automated

When adding new features, aim for at least 80% test coverage for your code. For complex benchmarks that require external dependencies, use mock implementations in tests to avoid unnecessary dependencies.

See the [Testing](../testing.md) page for more details about the test suite.

## Code Style

TTSDS follows standard Python code style guidelines:

1. Use Black for code formatting:

```bash
pip install black
black src/ tests/
```

2. Follow PEP 8 naming conventions:
   - `snake_case` for functions and variables
   - `PascalCase` for classes
   - `UPPER_CASE` for constants

3. Write comprehensive docstrings in Google style

## Documentation

TTSDS uses [MkDocs](https://www.mkdocs.org/) with the [Material](https://squidfunk.github.io/mkdocs-material/) theme and [mkdocstrings](https://mkdocstrings.github.io/) for API documentation.

1. API documentation is generated automatically from docstrings
2. User guides and tutorials should be added manually to the `docs/` directory
3. Build the documentation:

```bash
mkdocs build
```

4. Preview the documentation:

```bash
mkdocs serve
```

## Release Process

1. Update version in `src/ttsds/__about__.py`
2. Update CHANGELOG.md
3. Create a new release on GitHub
4. GitHub Actions will automatically publish to PyPI

## Pre-commit Hooks

The project uses pre-commit hooks to enforce code quality standards:

1. **Code Formatting**: Black and isort automatically format your code
2. **Testing**: Tests are run automatically before each commit
3. **Badge Generation**: Test coverage and status badges are updated on push

To manually run all pre-commit hooks:

```bash
pre-commit run --all-files
```

## Test Coverage and Badges

The project uses badges to track test status and code coverage. These badges are automatically updated when running the badge generation script:

```bash
python scripts/generate_badges.py
```

The script:
1. Runs the test suite with coverage tracking
2. Generates SVG badges for test status and coverage percentage
3. Saves badges to `docs/assets/img/`
4. Updates the README with badge links (if needed)

See the [Testing](../testing.md) page for more details about the test suite.

## Documentation

The documentation is built using MkDocs with the Material theme. API documentation is automatically generated from docstrings using mkdocstrings.

To preview the documentation locally:

```bash
mkdocs serve
```

To build the documentation:

```bash
mkdocs build
```

The built documentation will be available in the `site` directory.

## Code Style

TTSDS follows a consistent code style as described in the [Coding Style](coding_style.md) guide. Key highlights:

- Use descriptive variable and function names
- Follow PEP 8 guidelines
- Include docstrings for all public modules, functions, classes, and methods
- Write clean, maintainable code

The pre-commit hooks help enforce code style rules.

## Dependency Management

When adding new dependencies:

1. Update `pyproject.toml` with the new dependency
2. If it's a development-only dependency, add it to the `[project.optional-dependencies]` section under `dev`
3. Document any system requirements in the README

## Release Process

1. Update version in `src/ttsds/__about__.py`
2. Update CHANGELOG.md with the new version and changes
3. Create a new GitHub release with the version number as the tag
4. The GitHub Actions workflow will automatically build and publish the package to PyPI 