# Testing

[![Tests](https://raw.githubusercontent.com/ttsds/ttsds/main/docs/assets/img/tests.svg)](https://github.com/ttsds/ttsds/actions)
[![Coverage](https://raw.githubusercontent.com/ttsds/ttsds/main/docs/assets/img/coverage.svg)](https://github.com/ttsds/ttsds/actions)

## Overview

TTSDS includes a comprehensive test suite to ensure code quality and prevent regressions. The testing framework is built with pytest and includes both unit tests and integration tests.

## Test Structure

The test structure mirrors the package structure:

```
tests/
├── conftest.py             # Common test fixtures
├── run_tests.py            # Test runner script
├── unit/                   # Unit tests
│   ├── benchmarks/         # Tests for benchmark classes
│   │   ├── speaker/        # Tests for speaker benchmarks
│   │   └── ...             # Other benchmark categories
│   ├── test_ttsds.py       # Tests for BenchmarkSuite
│   └── util/               # Tests for utility modules
│       ├── test_dataset.py # Tests for dataset utilities
│       └── ...             # Other utility tests
└── integration/            # Integration tests
    └── test_benchmark_suite.py  # End-to-end tests
```

## Running Tests

To run the test suite, use the `run_tests.py` script:

```bash
# Run all tests (excluding benchmarks/speaker by default)
python tests/run_tests.py

# Run with verbose output
python tests/run_tests.py -v

# Include all tests (including speaker benchmarks)
python tests/run_tests.py --include-all

# Run specific test modules
python tests/run_tests.py tests/unit/benchmarks/test_benchmark.py

# Run tests with a specific marker
python tests/run_tests.py -m "integration"
```

By default, tests that require extensive mocking (e.g., speaker benchmark tests) are skipped. Use the `--include-all` flag to run all tests.

## Current Test Status

As of the latest update:
- All tests are passing
- Current code coverage is 31%
- Speaker benchmark tests are mocked to avoid dependencies
- Utility tests have been improved to work without direct imports

## Test Coverage

Test coverage is automatically generated when running the tests. The coverage report is available in the `coverage_html` directory after running the tests.

To generate test coverage badges, use:

```bash
python scripts/generate_badges.py
```

This will:
1. Run the unit tests with coverage tracking
2. Generate badges for test status and coverage percentage
3. Save badges to `docs/assets/img/`
4. Update the README with badge links (if needed)

## Test Fixtures

The `conftest.py` file contains common test fixtures used across multiple test modules:

- `mock_dataset`: A mock dataset for testing
- `mock_reference_dataset`: A mock reference dataset
- `mock_noise_dataset`: A mock noise dataset
- `temp_cache_dir`: A temporary directory for caching during tests

## Mock Implementation Strategy

Many tests in TTSDS use mock implementations rather than relying on the actual implementations. This approach:

1. Reduces test dependencies on external libraries
2. Makes tests faster and more reliable
3. Allows testing components in isolation
4. Avoids circular import issues

For example, in `test_dataset.py` we use mock implementations of Dataset classes rather than importing the actual ones, which helps avoid dependency and import issues.

## Areas for Improvement

The test suite could be improved in the following areas:

1. **Increase coverage**: Current coverage is 31%, which could be improved by adding more tests.
2. **Test more utilities**: Add tests for remaining utility modules (distances.py, measures.py, etc.).
3. **Add more benchmark tests**: Create test modules for other benchmark categories.
4. **Add integration tests**: Create more comprehensive end-to-end tests.
5. **Add parametrized tests**: Use pytest's parametrize feature for more comprehensive testing.

## Continuous Integration

Tests are automatically run on GitHub Actions for every pull request and push to the main branch. The workflow is defined in `.github/workflows/tests.yml`.

## Writing Tests

When writing new tests:

1. Place tests in the appropriate directory based on the component being tested
2. Use the existing fixtures when possible
3. Mock external dependencies to avoid network calls and speed up tests
4. Follow the naming convention: `test_*.py` for test files and `test_*` for test functions
5. Include both positive and negative test cases
6. Test edge cases and error conditions

### Example Test

```python
def test_benchmark_initialization():
    """Test benchmark initialization with default parameters."""
    benchmark = MockBenchmark(
        name="Test Benchmark",
        category=BenchmarkCategory.GENERIC,
        dimension=BenchmarkDimension.ONE_DIMENSIONAL,
        description="A test benchmark",
    )
    
    # Check properties were set correctly
    assert benchmark.name == "Test Benchmark"
    assert benchmark.key == "test_benchmark"
    assert benchmark.category == BenchmarkCategory.GENERIC
    assert benchmark.dimension == BenchmarkDimension.ONE_DIMENSIONAL
    assert benchmark.description == "A test benchmark"
```

## Adding Tests

When adding new features to TTSDS, please ensure you also add corresponding tests:

1. For new benchmark implementations, use the template in `tests/unit/benchmarks/benchmark_test_template.py`
2. Aim for at least 80% code coverage for new features
3. Include both happy path and error cases in your tests
4. For complex benchmarks, use mock implementations to avoid dependencies

## Test Dependencies

Test-specific dependencies are listed in the `[project.optional-dependencies]` section of `pyproject.toml` under the `dev` extras. 